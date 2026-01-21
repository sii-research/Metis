from .quant import *
import torch.nn as nn
import torch.nn.init as init
import transformer_engine.pytorch  as te

from functools import partial

import math
import uuid
from scipy.linalg import hadamard

def ensure_param_uuid(p: torch.nn.Parameter) -> str:
    if not hasattr(p, "_svd_uuid"):
        p._svd_uuid = uuid.uuid4().hex
    return p._svd_uuid


class LinearLowbitFunction(torch.autograd.Function):
    svd_input_history = {}
    svd_grad_output_history = {}
    svd_count = {}    
    
    q_forward_input = Cast2Fp4e2m1
    q_forward_weight = Cast2Fp4e2m1

    q_backward_input = Cast2Fp4e2m1
    q_backward_weight = Cast2Fp4e2m1
    q_backward_outputgrad = Cast2Fp4e2m1
    
    enable_nv_recipe = False
    
    activation_lowrank_niter = 0
    backward_lowrank_niter = 0
    
    enable_activation_svd = False
    activation_lowrank_svd = -1
    
    enable_backward_svd = False
    backward_lowrank_svd = -1
    # enable_backward_longtail = False
    
    activation_broadcast_dim = -1
    backward_broadcast_dim = -1
    gradacc_broadcast = False    
    gradacc_broadcast_steps = 1
    
    state = {}
    
    @staticmethod
    def _power_iter_svd(
        G32: torch.Tensor,   # [m, n], float32
        r: int,
        niter: int,
        key, 
        eps: float = 1e-8,
    ):
        """
        用 Dion 风格幂迭代近似 G32 的 rank-k SVD:
            G ≈ U S V^T

        首次调用（或 rank 变化）时，用 torch.svd_lowrank 做初始化：
            U, S, V = svd_lowrank(G32, q=k)
        并将 V 缓存到 state 里作为后续 step 的初始子空间。

        返回:
            U : [m, k]
            S : [k]
            V : [n, k]
        """
        m, n = G32.shape
        k = r
        
        
        state = LinearLowbitFunction.state
        V_prev = state.get(key, None)

        # ---- 情况 1：没有历史 V，或者形状不匹配 —— 回退到精度较高的 svd_lowrank ----
        if V_prev is None or V_prev.shape != (n, k):
            # print(key, (n, k))
            # if not (V_prev is None):
                # print(V_prev.shape)
            # torch.svd_lowrank 直接给出近似 U,S,V
            U_svd, S_svd, V_svd = torch.svd_lowrank(G32, q=k)  # U:[m,k], S:[k], V:[n,k]
            # 缓存右子空间作为后续 step 的初始迭代子空间
            state[key] = V_svd.detach()
            return U_svd, S_svd, V_svd

        # ---- 情况 2：有历史 V，用 Dion 风格幂迭代 refine ----
        V_prev = V_prev.to(device=G32.device, dtype=G32.dtype)
        V = V_prev
        U = None
        W = None

        for it in range(niter):
            # P = G V,  shape [m,k]
            P = G32 @ V

            # U = Orthonormalize(P)，这里直接用 QR
            U, _ = torch.linalg.qr(P, mode="reduced")  # [m,k]

            # W = G^T U, shape [n,k]
            W = G32.mT @ U

            if it < niter - 1:
                # 为下一轮迭代更新 V：列归一化 W
                col_norm = torch.linalg.norm(W, dim=0, keepdim=True)  # [1,k]
                V = W / (col_norm + eps)

        # 最后一次迭代后，从 W 中提取 S, V
        S = torch.linalg.norm(W, dim=0)           # [k]
        V_new = W / (S.unsqueeze(0) + eps)        # [n,k]，列单位范数

        # 缓存右子空间，供下一步使用
        state[key] = V_new.detach()

        return U, S, V_new
                
    
    @staticmethod
    def svd_quant(
        input_:torch.Tensor, quant_func, rank=60, niter=0, 
        use_power_iter=True, key=None,
        broadcast_dim=-1,
        gradacc_broadcast = False, load_history = False, history_id=0, history_list={}
    ):
                        
        if broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)
        else:
            cinput = input_
        
        original_shape = cinput.shape
        if len(original_shape) == 3:
            cinput = cinput.reshape(-1, original_shape[-1])
            input_ = input_.reshape(-1, original_shape[-1])
        
        # print(gradacc_broadcast, load_history)
        if gradacc_broadcast and load_history:
            ug = history_list[history_id][0]
            sg = history_list[history_id][1]
            vg = history_list[history_id][2]
            ker = history_list[history_id][3]     
            # print("load")       
        else:
            if use_power_iter:
                ug, sg, vg = LinearLowbitFunction._power_iter_svd(
                    cinput,
                    r=rank,
                    niter=niter,
                    key=key
                )
            else:
                ug, sg, vg = torch.svd_lowrank(
                    cinput, 
                    q=rank, 
                    niter=niter
                )
        
            vg = vg.T
            ug = ug.T                        

            ker = (ug.T @ torch.diag(sg) @ vg)
            if broadcast_dim >= 0:
                ker = ker.unsqueeze(broadcast_dim)            
            
            ug_scalar = quant_func.get_scalar(ug)
            vg_scalar = quant_func.get_scalar(vg)
            ug = quant_func.quant(ug, ug_scalar)
            ug = quant_func.rquant(ug, ug_scalar)
            
            vg = quant_func.quant(vg, vg_scalar)
            vg = quant_func.rquant(vg, vg_scalar)
            
            if gradacc_broadcast:
                history_list[history_id] = [ug, sg, vg, ker]

        input_res = input_ - ker
        input_res_scalar = quant_func.get_scalar(input_res)
        input_res = quant_func.quant(input_res, input_res_scalar)
        input_res = quant_func.rquant(input_res, input_res_scalar)        
        
        quant_func
        
        input_ = ug.T @ torch.diag(sg) @ vg
        if broadcast_dim >= 0:
            input_ = input_.unsqueeze(broadcast_dim)

        input_ = input_ + input_res 
        
        if len(original_shape) == 3:
            input_ = input_.view(original_shape[0], original_shape[1], -1)
        return input_
        

    @staticmethod
    def forward(ctx, input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, weight_uuid: str):        
        wdim = weight.shape[-1]
        idim = input_.shape[-1]
        if (not hasattr(LinearLowbitFunction, "h")) and LinearLowbitFunction.enable_nv_recipe:
            LinearLowbitFunction.hdim = 4096
            H_scipy = hadamard(LinearLowbitFunction.hdim)
            LinearLowbitFunction.h = torch.from_numpy(H_scipy).float().to(weight.device)                
        
        """grad acc related"""
        ctx.weight_uuid = weight_uuid  # stash for backward
        idweight = weight_uuid        
        if LinearLowbitFunction.svd_count.get(idweight) is None:
            LinearLowbitFunction.svd_count[idweight] = 0                                
        load_history = False 
        # if LinearLowbitFunction.svd_count[idweight] % LinearLowbitFunction.gradacc_broadcast_steps == 0:
        #     load_history = False
        """ """
        key = weight.id4svd
        if LinearLowbitFunction.enable_forward_svd_intime:
            weight = LinearLowbitFunction.svd_quant(
                weight, 
                quant_func=LinearLowbitFunction.q_forward_weight,
                rank=LinearLowbitFunction.forward_lowrank_svd_intime,
                niter=LinearLowbitFunction.activation_lowrank_niter,  
                key=None,              
                broadcast_dim=LinearLowbitFunction.activation_broadcast_dim,
                gradacc_broadcast=LinearLowbitFunction.gradacc_broadcast,
                load_history=load_history,
                history_id=idweight,
                history_list=LinearLowbitFunction.svd_input_history
            )
        
        
        if LinearLowbitFunction.enable_activation_svd:
            input_ = LinearLowbitFunction.svd_quant(
                input_, 
                quant_func=LinearLowbitFunction.q_forward_input,
                rank=LinearLowbitFunction.activation_lowrank_svd,
                niter=LinearLowbitFunction.activation_lowrank_niter,  
                key=(key, 0),              
                broadcast_dim=LinearLowbitFunction.activation_broadcast_dim,
                gradacc_broadcast=LinearLowbitFunction.gradacc_broadcast,
                load_history=load_history,
                history_id=idweight,
                history_list=LinearLowbitFunction.svd_input_history
            )
            input_scalar = LinearLowbitFunction.q_forward_input.get_scalar(input_)        
        else:
            if LinearLowbitFunction.enable_nv_recipe:
                input_ = input_ @ LinearLowbitFunction.h[: idim]
            input_scalar = LinearLowbitFunction.q_forward_input.get_scalar(input_)
            input_ = LinearLowbitFunction.q_forward_input.quant(input_, input_scalar)
            input_ = LinearLowbitFunction.q_forward_input.rquant(input_, input_scalar)
            if LinearLowbitFunction.enable_nv_recipe:
                input_ = input_ @ LinearLowbitFunction.h[: idim].mT / LinearLowbitFunction.hdim
    
        
        if LinearLowbitFunction.enable_nv_recipe:
            weight = weight @ LinearLowbitFunction.h[: wdim]
        weight_scalar = LinearLowbitFunction.q_forward_input.get_scalar(weight)
        if LinearLowbitFunction.enable_forward_svd_intime == False:
            weight = LinearLowbitFunction.q_forward_weight.quant(weight, weight_scalar)
            weight = LinearLowbitFunction.q_forward_weight.rquant(weight, weight_scalar)
        if LinearLowbitFunction.enable_nv_recipe:
            weight = weight @ LinearLowbitFunction.h[: wdim].mT / LinearLowbitFunction.hdim
        
        t = torch.zeros([1], dtype=torch.int64)
        t[0] = key 
        ctx.save_for_backward(
            input_, 
            weight, 
            input_scalar, 
            weight_scalar, 
            bias,
            t
        )
           
        output = torch.matmul(input_, weight.T)
        
        if bias is not None:
            output += bias
        
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_, weight, input_scalar, weight_scalar, bias, key = ctx.saved_tensors        
        key = int(key[0].item())
        
        grad_bias = grad_output.sum(dim=(0, 1)) if bias is not None else None
        
        grad_output_shape0 = grad_output.shape[0]
        grad_output_shape1 = grad_output.shape[1]
        grad_output_shape2 = grad_output.shape[2]

        
        if LinearLowbitFunction.enable_backward_svd:
            if LinearLowbitFunction.backward_lowrank_svd > 0:                
                """gradacc broadcast related"""
                idweight = ctx.weight_uuid                
                load_history = True 
                if LinearLowbitFunction.svd_count[idweight] % LinearLowbitFunction.gradacc_broadcast_steps == 0:
                    load_history = False
                """ """
                    
                grad_output = LinearLowbitFunction.svd_quant(
                    grad_output, 
                    quant_func=LinearLowbitFunction.q_backward_outputgrad,
                    rank=LinearLowbitFunction.backward_lowrank_svd,
                    niter=LinearLowbitFunction.backward_lowrank_niter,      
                    key=(key, 1),              
                    broadcast_dim=LinearLowbitFunction.backward_broadcast_dim,
                    gradacc_broadcast=LinearLowbitFunction.gradacc_broadcast,
                    load_history=load_history,
                    history_id=idweight,
                    history_list=LinearLowbitFunction.svd_grad_output_history
                )
                grad_output = grad_output.reshape(-1, grad_output.shape[-1]).T
                LinearLowbitFunction.svd_count[idweight] += 1
            else:
                ug, sg, vg = torch.linalg.svd(grad_output, full_matrices=False)
                ug_scalar = ug.abs().mean()
                vg_scalar = vg.abs().mean()
                
                grad_output = \
                    LinearLowbitFunction.q_backward_outputgrad(ug / ug_scalar) @ \
                    torch.diag(sg) @ \
                    LinearLowbitFunction.q_backward_outputgrad(vg / vg_scalar)

                grad_output *= ug_scalar * vg_scalar
        else:
            gdim = grad_output.shape[-1]
            if LinearLowbitFunction.enable_nv_recipe:
                grad_output = grad_output @ LinearLowbitFunction.h[: gdim]
            
            grad_output_scalar = LinearLowbitFunction.q_backward_outputgrad.get_scalar(grad_output)
            
            grad_output = LinearLowbitFunction.q_backward_outputgrad.quant(grad_output, grad_output_scalar)
            grad_output = LinearLowbitFunction.q_backward_outputgrad.rquant(grad_output, grad_output_scalar)
            if LinearLowbitFunction.enable_nv_recipe:
                grad_output = grad_output @ LinearLowbitFunction.h[: gdim].mT / LinearLowbitFunction.hdim
            
            grad_output = grad_output.reshape(-1, grad_output.shape[-1]).T
            
        grad_weight = torch.matmul(
            grad_output,
            input_.reshape(-1, input_.shape[-1])
        )
    
        grad_output = grad_output.T.reshape(grad_output_shape0, grad_output_shape1, grad_output_shape2)
        grad_input = torch.matmul(grad_output, weight)                    
        
        return grad_input, grad_weight, grad_bias, None

class LinearLowbit(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias=True,
        args=None, 
        device=None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=args.device if device is None else device)
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((out_features,), dtype=torch.float32, device=args.device if device is None else device)
            )
        else:
            self.bias = None
            
        # >>> assign persistent UUID to this Parameter <<<
        self.weight_uuid = ensure_param_uuid(self.weight)
        if self.bias is not None:
            ensure_param_uuid(self.bias)
            
        self.reset_parameters()
        self.weight.id4svd = id(self.weight)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        # return LinearLowbitFunction.apply(input, self.weight, self.bias)
        # >>> pass the UUID to the autograd.Function <<<
        return LinearLowbitFunction.apply(input, self.weight, self.bias, self.weight_uuid)

    pass

class BitLinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        args=None,
        bias=True
    ):
        super().__init__()
        if args.enable_forward_svd == False and args.enable_lowbit == True:
            if args.enable_te:
                self.warmup_linear = te.Linear(in_features, out_features, device=args.device)
            else:
                self.warmup_linear = LinearLowbit(in_features, out_features, bias=bias, args=args)
        else:
            self.warmup_linear = nn.Linear(in_features, out_features, bias=bias, device=args.device)
            init.kaiming_uniform_(self.warmup_linear.weight, a=math.sqrt(5))
            if bias:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.warmup_linear.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.warmup_linear.bias, -bound, bound)

        self.ulinear = None
        self.vlinear = None
        self.s = None

        LinearLowbitFunction.q_forward_input = quant_func[args.q_forward_input]
        LinearLowbitFunction.q_forward_weight = quant_func[args.q_forward_weight]
        LinearLowbitFunction.q_backward_input = quant_func[args.q_backward_input]
        LinearLowbitFunction.q_backward_weight = quant_func[args.q_backward_weight]
        LinearLowbitFunction.q_backward_outputgrad = quant_func[args.q_backward_outputgrad]
        
        LinearLowbitFunction.enable_backward_svd = args.enable_backward_svd
        LinearLowbitFunction.backward_lowrank_svd = args.backward_lowrank_svd
        LinearLowbitFunction.backward_lowrank_niter = args.backward_lowrank_niter
        
        LinearLowbitFunction.enable_forward_svd_intime = args.enable_forward_svd_intime
        LinearLowbitFunction.forward_lowrank_svd_intime = args.forward_lowrank_svd_intime
        
        LinearLowbitFunction.enable_activation_svd = args.enable_activation_svd
        LinearLowbitFunction.activation_lowrank_svd = args.activation_lowrank_svd
        LinearLowbitFunction.activation_lowrank_niter = args.activation_lowrank_niter                        
        
        LinearLowbitFunction.activation_broadcast_dim = args.activation_broadcast_dim
        LinearLowbitFunction.backward_broadcast_dim = args.backward_broadcast_dim
        
        LinearLowbitFunction.enable_nv_recipe = args.enable_nv_recipe
        
        LinearLowbitFunction.gradacc_broadcast = args.gradacc_broadcast
        LinearLowbitFunction.gradacc_broadcast_steps = args.gradacc_broadcast_steps        

        self.args = args
        self.is_svd_quant = False
        
        
        if args.forward_svd_warmup_steps <= 0 and args.enable_forward_svd:
            print("split")
            self.split()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_svd_quant:
            y = self.vlinear(x)
            y = torch.mul(self.s, y)
            y = self.ulinear(y)
            if self.args.forward_svd_rank > 0:
                y += self.warmup_linear(x)
            
            
        else:
            y = self.warmup_linear(x)
        
        return y
    
    @staticmethod
    def _init_telinear(w, weight):
        torch.nn.init.ones_(weight)
        weight.mul_(w)
    
    @torch.no_grad()
    def split(self):
        if not self.args.enable_forward_svd:
            return
        
        
        
        if not self.vlinear is None:
            u, s, v = torch.linalg.svd(
                self.ulinear.weight @ 
                torch.diag(self.s) @ 
                self.vlinear.weight, full_matrices=False)
            
            bias = self.ulinear.bias
            device = self.ulinear.weight.device
        else:
            device = self.warmup_linear.weight.device
            u, s, v = torch.linalg.svd(self.warmup_linear.weight, full_matrices=False)
            u = u.cuda(self.warmup_linear.weight.get_device())
            s = s.cuda(self.warmup_linear.weight.get_device())
            v = v.cuda(self.warmup_linear.weight.get_device())
            
            if not self.warmup_linear.bias is None:
                bias = self.warmup_linear.bias.to(device=device)
            else:
                bias = None
            w = self.warmup_linear.weight.to(device=device)
            # forward svd low rank
            if self.args.forward_svd_rank > 0:
                self.warmup_linear = LinearLowbit(
                    self.warmup_linear.weight.shape[1], 
                    self.warmup_linear.weight.shape[0],
                    bias=True if not bias is None else False, 
                    args=self.args,
                    # device=device
                )
                if not bias is None:
                    self.warmup_linear.bias.copy_(bias)
                self.warmup_linear.weight.copy_(
                    w - \
                    u[:,:self.args.forward_svd_rank] @ \
                    torch.diag(s[:self.args.forward_svd_rank]) @ \
                    v[:self.args.forward_svd_rank]
                )
            
            
            
        
        if self.args.enable_lowbit: 
            # nv fp8
            # ******************************************************************
            # self.ss = u @ s @ u.transpose()
            # with fp8_model_init(enabled=True):
            #     self.uvlinear = te.Linear(
            #         self.warmup_linear.weight.shape[1], 
            #         self.warmup_linear.weight.shape[0], 
            #         init_method=partial(BitLinear._init_telinear, u @ v), 
            #         bias=False, 
            #         device=self.device
            #     )
            
            if self.args.enable_te:
                self.vlinear = te.Linear(
                    v.shape[1], 
                    v.shape[0], 
                    init_method=partial(BitLinear._init_telinear, v), 
                    bias=False, 
                    device=self.device
                )
                self.ulinear = te.Linear(
                    u.shape[1], 
                    u.shape[0], 
                    init_method=partial(BitLinear._init_telinear, u), 
                    bias=False, 
                    device=self.device
                )
            # ******************************************************************
            
            elif self.args.forward_svd_rank > 0:
                # self.vlinear = LinearLowbit(
                #     v.shape[1], 
                #     self.args.forward_svd_rank, # v.shape[0] // 30, 
                #     bias=False, 
                #     args=self.args,
                #     # device=device
                # )
                self.vlinear = nn.Linear(
                    v.shape[1], 
                    self.args.forward_svd_rank, # v.shape[0] // 30, 
                    bias=False, 
                    # args=self.args,
                    device=device
                )
                self.ulinear = nn.Linear(
                    self.args.forward_svd_rank, # u.shape[1] // 30, 
                    u.shape[0], 
                    bias=False, 
                    device=device
                )
                self.vlinear.weight.copy_(v[: self.args.forward_svd_rank, :])
                self.ulinear.weight.copy_(u[:, : self.args.forward_svd_rank])
            else:
                self.vlinear = LinearLowbit(
                    v.shape[1], 
                    v.shape[0], # v.shape[0] // 30, 
                    bias=False, 
                    args=self.args,
                    # device=device
                )
                self.ulinear = nn.Linear(
                    u.shape[1], # u.shape[1] // 30, 
                    u.shape[0], 
                    bias=False,
                    device=device,
                )
                self.vlinear.weight.copy_(v)
                self.ulinear.weight.copy_(u)
            
            
            # forward svd low rank
            # if self.args.forward_svd_rank > 0 and not bias is None:
            #     self.ulinear.bias.copy_(bias)
        else:
            self.vlinear = nn.Linear(v.shape[1], v.shape[0], bias=False)
            self.ulinear = nn.Linear(u.shape[1], u.shape[0])

            
            self.vlinear.weight = nn.Parameter(v)
            self.ulinear.weight = nn.Parameter(u)
            if (not bias is None):
                self.ulinear.bias = nn.Parameter(
                    self.warmup_linear.bias.clone().cuda(self.warmup_linear.weight.get_device())
                )
        
        
        self.is_svd_quant = True
        
        if self.args.forward_svd_rank > 0:
            self.s = torch.nn.Parameter(s[:self.args.forward_svd_rank])
            
        else:
            self.s = torch.nn.Parameter(s)
            self.warmup_linear = None
