import torch
import os
import argparse
from transformers import AutoConfig, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="模型权重结构对比与转换工具")
    
    # 必需参数
    parser.add_argument("--config_dir", type=str, required=True, 
                        help="官方模型配置目录 (包含 config.json)")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="源断点文件路径 (例如 checkpoint_20000.pth)")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="转换后权重的保存路径 (例如 aligned_model.pth)")
    
    # 可选参数
    parser.add_argument("--trust_remote_code", action="store_true", 
                        help="是否信任远程代码 (加载 config 时)")
    parser.add_argument("--skip_compare", action="store_true", 
                        help="跳过对比步骤，只执行转换")
    
    return parser.parse_args()

def load_state_dict_safely(path):
    """
    兼容加载不同格式的 checkpoint (纯 state_dict 或包裹在字典中)
    """
    print(f"正在读取文件：{path}")
    checkpoint = torch.load(path, map_location="cpu")
    
    # 常见格式判断
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        else:
            # 假设整个文件就是 state_dict
            return checkpoint
    else:
        raise ValueError(f"未知的 checkpoint 格式：{type(checkpoint)}")

def compare_official_vs_checkpoint(config_dir, checkpoint_path, trust_remote_code=False):
    # 1. 获取官方模型的结构
    print(f"\n[对比] 正在从 {config_dir} 加载官方配置...")
    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=trust_remote_code)
    official_model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    official_state_dict = official_model.state_dict()

    # 2. 获取断点文件的结构
    checkpoint_state_dict = load_state_dict_safely(checkpoint_path)

    # 3. 开始对比输出
    official_keys = sorted(list(official_state_dict.keys()))
    checkpoint_keys = sorted(list(checkpoint_state_dict.keys()))

    print("\n" + "="*100)
    print(f"{'官方层名称 (Official)':<50} | {'断点层名称 (Checkpoint)':<50}")
    print(f"{'形状 (Shape)':<50} | {'形状 (Shape)':<50}")
    print("="*100)

    max_len = max(len(official_keys), len(checkpoint_keys))
    mismatch_count = 0
    
    for i in range(max_len):
        # 处理官方层信息
        if i < len(official_keys):
            off_k = official_keys[i]
            off_s = str(list(official_state_dict[off_k].shape))
        else:
            off_k = "---"
            off_s = "---"

        # 处理你的断点层信息
        if i < len(checkpoint_keys):
            ckpt_k = checkpoint_keys[i]
            ckpt_s = str(list(checkpoint_state_dict[ckpt_k].shape))
        else:
            ckpt_k = "---"
            ckpt_s = "---"

        if off_k != ckpt_k or off_s != ckpt_s:
            # 输出名称行
            print(f"{off_k:<50} | {ckpt_k:<50}")
            # 输出形状行
            print(f"{off_s:<50} | {ckpt_s:<50}")
            print("-" * 100)
            mismatch_count += 1

    # 统计信息
    print(f"\n[对比总结]:")
    print(f"官方模型总参数项：{len(official_keys)}")
    print(f"断点文件总参数项：{len(checkpoint_keys)}")
    if mismatch_count == 0:
        print("✅ 结构完全一致！")
    else:
        print(f"⚠️  发现 {mismatch_count} 处不一致 (包括名称或形状)")
    print("="*100 + "\n")

# def convert_checkpoint(old_ckpt_path):
#     """
#     加载旧权重，修改 key 名称，返回新的 state_dict
#     """
#     print(f"\n[转换] 正在处理权重：{old_ckpt_path}")
#     old_state_dict = load_state_dict_safely(old_ckpt_path)
    
#     new_state_dict = {}
#     modify_count = 0
#     for k, v in old_state_dict.items():
#         # 根据观察到的规律改名 (移除 warmup_linear.)
#         if "warmup_linear." in k:
#             new_key = k.replace("warmup_linear.", "")
#             modify_count += 1
#         else:
#             new_key = k
#         new_state_dict[new_key] = v
        
#     print(f"[转换] 共修改了 {modify_count} 个参数名称")
#     return new_state_dict

def convert_checkpoint(old_ckpt_path):
    """
    核心转换逻辑：
    1. 识别低秩分解 (ulinear, vlinear, s, warmup_linear) -> 合并为 weight
    2. 识别普通微调 (warmup_linear) -> 重命名为 weight
    3. 其他参数保持不变
    """
    print(f"\n[转换] 正在处理权重：{old_ckpt_path}")
    old_state_dict = load_state_dict_safely(old_ckpt_path)
    
    new_state_dict = {}
    
    # 用于暂存需要合并的层信息
    # 结构：{ base_module_name: { 'u': key, 'v': key, 's': key, 'residual': key } }
    merge_map = {}
    
    # 记录需要被跳过（因为会被合并）的原始 key
    keys_to_skip = set()

    # --- 第一步：扫描所有 key，识别模式 ---
    for k in old_state_dict.keys():
        is_special = False
        base_name = None
        
        # 识别 ulinear
        if k.endswith(".ulinear.weight"):
            base_name = k[:-len(".ulinear.weight")]
            if base_name not in merge_map: merge_map[base_name] = {}
            merge_map[base_name]['u'] = k
            is_special = True
            
        # 识别 vlinear
        elif k.endswith(".vlinear.weight"):
            base_name = k[:-len(".vlinear.weight")]
            if base_name not in merge_map: merge_map[base_name] = {}
            merge_map[base_name]['v'] = k
            is_special = True
            
        # 识别 s (注意 s 后面没有 .weight)
        elif k.endswith(".s"):
            base_name = k[:-len(".s")]
            if base_name not in merge_map: merge_map[base_name] = {}
            merge_map[base_name]['s'] = k
            is_special = True
            
        # 识别 warmup_linear (残差部分)
        elif k.endswith(".warmup_linear.weight"):
            base_name = k[:-len(".warmup_linear.weight")]
            if base_name not in merge_map: merge_map[base_name] = {}
            merge_map[base_name]['residual'] = k
            is_special = True
        
        if is_special:
            keys_to_skip.add(k)

    # --- 第二步：构建新字典 ---
    merge_count = 0
    rename_count = 0

    for k, v in old_state_dict.items():
        if k in keys_to_skip:
            continue
        new_state_dict[k] = v

    # --- 第三步：处理合并逻辑 ---
    for base_name, parts in merge_map.items():
        target_key = f"{base_name}.weight"
        
        # 情况 A: 低秩分解模式 (存在 U, V, S)
        if 'u' in parts and 'v' in parts and 's' in parts:
            u_tensor = old_state_dict[parts['u']]
            v_tensor = old_state_dict[parts['v']]
            s_tensor = old_state_dict[parts['s']].squeeze() # 确保 s 是 1 维
            
            # 获取残差，如果存在的话
            if 'residual' in parts:
                residual_tensor = old_state_dict[parts['residual']]
            else:
                # 理论上分解模式应该都有残差，如果没有则初始化为 0
                print(f"[警告] {base_name} 缺少残差部分，将仅使用低秩部分")
                residual_tensor = torch.zeros_like(torch.matmul(u_tensor * s_tensor, v_tensor))
            
            # 计算：(U * S) @ V + Residual
            # u_tensor: [dim_out, rank], s_tensor: [rank], v_tensor: [rank, dim_in]
            # u_tensor * s_tensor 利用广播机制将奇异值乘到 U 的列上
            low_rank_part = torch.matmul(u_tensor * s_tensor, v_tensor)
            final_weight = low_rank_part + residual_tensor
            
            new_state_dict[target_key] = final_weight
            merge_count += 1
            print(f"[合并] {base_name} (低秩分解还原)")
            
        # 情况 B: 普通模式 (仅存在 warmup_linear/residual)
        elif 'residual' in parts:
            residual_tensor = old_state_dict[parts['residual']]
            new_state_dict[target_key] = residual_tensor
            rename_count += 1
            print(f"[重命名] {parts['residual']} -> {target_key}")
            
        else:
            print(f"[警告] {base_name} 发现不完整的分解部分 {parts.keys()}，已跳过")

    print(f"[转换完成] 共合并低秩层：{merge_count} 个，重命名普通层：{rename_count} 个")
    return new_state_dict


def main():
    args = parse_args()

    # 1. 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"输出目录不存在，正在创建：{output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 2. 执行对比 (转换前)
    if not args.skip_compare:
        compare_official_vs_checkpoint(
            args.config_dir, 
            args.checkpoint_path, 
            trust_remote_code=args.trust_remote_code
        )

    # 3. 执行转换
    aligned_state_dict = convert_checkpoint(args.checkpoint_path)

    # 4. 保存新权重
    print(f"正在保存对齐后的权重到：{args.output_path}")
    torch.save(aligned_state_dict, args.output_path)
    print(f"✅ 成功保存。")

    # 5. 执行对比 (转换后验证)
    if not args.skip_compare:
        print("\n>>> 开始验证转换后的文件结构...")
        compare_official_vs_checkpoint(
            args.config_dir, 
            args.output_path, 
            trust_remote_code=args.trust_remote_code
        )

if __name__ == "__main__":
    main()
                       