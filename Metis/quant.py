
import torch


class QuantFunc:
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        return x / s
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        return x * s


class WeightQuant(QuantFunc):
    @classmethod
    @torch.no_grad()
    def quant(cls, w, eps: float = 1e-6, bits = 1):
        
        abs_mean = w.abs().mean()
        abs_std  = w.abs().std()
        
        max_w = 2 * abs_std + eps
        q_range = max_w / (2 ** bits)
        w_quant = w / q_range
        
        w_quant = w_quant.round() / (2 ** bits)
        w_quant = w_quant.clamp(-1, 1) * abs_mean
    
        return w_quant

class Cast2Fp4e2m1(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 6 + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xsign = x.sign()
        x = x.abs() / (s / 2)
        
        
        x -= (x - 4).relu_() / 2 + (x - 8).relu_() / 4
        x.round_()
        x += (x - 4).relu_() + (x - 6).relu_() * 2      
        return x * xsign / 2
    
class Cast2Fp4e2m1Random(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 6 + 1e-6
    
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x:torch.Tensor, s: torch.Tensor):
        xsign = x.sign()
        x = x.abs() / (s / 2)
        
        x -= (x - 4).relu_() / 2 + (x - 8).relu_() / 4
        x += torch.rand_like(x) - 0.5
        x.round_()
        x += (x - 4).relu_() + (x - 6).relu_() * 2      
        return x * xsign / 2
        # return out * xsign

class Cast2Fp6e3m2(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 625 + 1e-7
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        x1 = (x / s).clamp(-625, 625).abs()
        x1 = (x1 ** (1 / 4)).to(torch.float8_e5m2).to(torch.float32)
        x1 = x1 ** 4

        return torch.sign(x) * x1

class Cast2Fp8e4m3(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 448 + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        return (x / s).to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)


class Cast2Fp32(QuantFunc):
    pass

class BlockQuantFunc(QuantFunc):
    block_shape = (1, 16)
                
    @classmethod
    @torch.no_grad()
    def _reshape(cls, x: torch.Tensor, s: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        s = s.view(rows // brows, 1, cols // bcols, 1)
        x = x.view(rows // brows, brows, cols // bcols, bcols)
        return x, s
    

class Cast2MXFp4e2m1Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        s = s.sign() * (2 ** ((s + 1e-127).log2().clamp_(-127, 127).round_()))
        
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.rquant(x, s).view(xshape)
    

class Cast2MXFp4e2m1BlockNOSR(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        s = s.sign() * (2 ** ((s + 1e-127).log2().clamp_(-127, 127).round_()))
        
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.rquant(x, s).view(xshape)

class Cast2NVFp4e2m1Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        smax = s.abs().max()
        s = s / (smax / 448)            
        s = s.to(dtype=torch.float8_e4m3fn).to(dtype=x.dtype)                    
        s *= smax / 448
        
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.rquant(x, s).view(xshape)
    
class Cast2NVFp4e2m1BlockNOSR(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        smax = s.abs().max()
        s = s / (smax / 448)
        s = s.to(dtype=torch.float8_e4m3fn).to(dtype=x.dtype)          
        s *= smax / 448
        
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.rquant(x, s).view(xshape)

class Cast2Fp4e2m1Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.rquant(x, s).view(xshape)
    
class Cast2Fp6e3m2Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 625 + 1e-7
        
        return x.to(dtype=torch.float16).to(dtype=torch.float32)
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp6e3m2.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp6e3m2.rquant(x, s).view(xshape)


class Cast2Fp8e4m3Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 448 + 1e-7
        
        return x.to(dtype=torch.float16).to(dtype=torch.float32)
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp8e4m3.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp8e4m3.rquant(x, s).view(xshape)

@torch.no_grad()
def cast_2_fp32(x):
    return x


class Cast2Hif4(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        qdim = -1
        x = x.reshape(-1, x.shape[-1])
        x = x.unflatten(qdim, (-1, 8, 2, 4))
        x_unsigned = torch.abs(x)
        sign = torch.sign(x)
        
        # compute initial shared exp 
        max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
        max_lv2 = torch.max(max_lv3, dim=qdim-1, keepdim=True)[0]
        max_lv1 = torch.max(max_lv2, dim=qdim-2, keepdim=True)[0]
        
        div7 = torch.ones_like(max_lv1) / 7.0
        div7 = div7.to(torch.bfloat16).to(x.dtype)
        scale_factor = max_lv1 * div7
        scale_factor = (scale_factor).to(torch.bfloat16).to(x.dtype).clip(min=2 ** (-48), max=49152) 
        ## change to tobf16(rint)
        e_sf = torch.floor(torch.log2(scale_factor))
        mant_sf = scale_factor / 2**e_sf * 2**7
        scale_factor = torch.round(mant_sf) / 2**7 * 2**e_sf

        
        # scale_factor to e6m2
        e_sf = torch.floor(torch.log2(scale_factor))
        scale_factor = torch.round(scale_factor * torch.exp2(2-e_sf)) * torch.exp2(e_sf-2)
        
        rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)
        scale_lv2 = (max_lv2 * rec_sf)
        scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())
        scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())
        
        return scale_lv2 * scale_lv3 / rec_sf
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        qdim = -1
        man_bits = 3
        x = x.reshape(-1, x.shape[-1])
        x = x.unflatten(qdim, (-1, 8, 2, 4))
        x_unsigned = torch.abs(x)
        sign = torch.sign(x)
        
        mant = x_unsigned / s
        mant = torch.floor(mant * 2**(man_bits - 1) + 0.5) / 2**(man_bits - 1)
        mant[mant>=2] = 2 - 2**(-man_bits+1)
        mant = sign * mant
        mant = mant.flatten(qdim-3, qdim)
        return  mant.view(xshape)#output of shape (dim1, (-1, 8, 2, 4))
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        # print(x.shape)
        qdim = -1
        xshape = x.shape
        x = x.reshape(-1, x.shape[-1])
        x = x.unflatten(qdim, (-1, 8, 2, 4))
        out = x * s
        out = out.flatten(qdim-3, qdim)
        return out.view(xshape)
    
    
class Cast2Hif4SR(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return Cast2Hif4.get_scalar(x)
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        qdim = -1
        man_bits = 3
        x = x.reshape(-1, x.shape[-1])
        x = x.unflatten(qdim, (-1, 8, 2, 4))
        x_unsigned = torch.abs(x)
        sign = torch.sign(x)
        
        mant = x_unsigned / s        
        q = 2**(man_bits - 1)
        # mant = torch.floor(mant * 2**(man_bits - 1) + 0.5) / 2**(man_bits - 1)
        #TODO: check SR
        mant = torch.floor(mant * q + torch.rand_like(mant * q)) / q
        mant[mant>=2] = 2 - 2**(-man_bits+1)
        mant = sign * mant
        mant = mant.flatten(qdim-3, qdim)
        return  mant.view(xshape)#output of shape (dim1, (-1, 8, 2, 4))  
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        return Cast2Hif4.rquant(x, s)
    
    
class Cast2Hif4SSR(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        qdim = -1
        x = x.reshape(-1, x.shape[-1])
        x = x.unflatten(qdim, (-1, 8, 2, 4))
        x_unsigned = torch.abs(x)
        sign = torch.sign(x)
        
        # compute initial shared exp 
        max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
        max_lv2 = torch.max(max_lv3, dim=qdim-1, keepdim=True)[0]
        max_lv1 = torch.max(max_lv2, dim=qdim-2, keepdim=True)[0]
        
        div7 = torch.ones_like(max_lv1) / 7.0
        div7 = div7.to(torch.bfloat16).to(x.dtype)
        scale_factor = max_lv1 * div7
        scale_factor = (scale_factor).to(torch.bfloat16).to(x.dtype).clip(min=2 ** (-48), max=49152) 
        ## change to tobf16(rint)
        e_sf = torch.floor(torch.log2(scale_factor))
        mant_sf = scale_factor / 2**e_sf * 2**7        
        scale_factor = torch.floor(mant_sf + torch.rand_like(mant_sf)) / 2**7 * 2**e_sf

        
        # scale_factor to e6m2
        e_sf = torch.floor(torch.log2(scale_factor))        
        scale_factor = torch.floor(scale_factor * torch.exp2(2-e_sf) + torch.rand_like(e_sf)) * torch.exp2(e_sf-2)
        
        rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)
        scale_lv2 = (max_lv2 * rec_sf)
        scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())
        scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())
        
        return scale_lv2 * scale_lv3 / rec_sf
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        qdim = -1
        man_bits = 3
        x = x.reshape(-1, x.shape[-1])
        x = x.unflatten(qdim, (-1, 8, 2, 4))
        x_unsigned = torch.abs(x)
        sign = torch.sign(x)
        
        mant = x_unsigned / s        
        q = 2**(man_bits - 1)
        # mant = torch.floor(mant * 2**(man_bits - 1) + 0.5) / 2**(man_bits - 1)        
        mant = torch.floor(mant * q + torch.rand_like(mant * q)) / q
        mant[mant>=2] = 2 - 2**(-man_bits+1)
        mant = sign * mant
        mant = mant.flatten(qdim-3, qdim)
        return  mant.view(xshape)#output of shape (dim1, (-1, 8, 2, 4))  
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        return Cast2Hif4.rquant(x, s)


quant_func = {
    "fp4e2m1": Cast2Fp4e2m1,
    "nvfp4e2m1b": Cast2NVFp4e2m1Block,
    "mxfp4e2m1b": Cast2MXFp4e2m1Block,    
    "mxfp4e2m1bnosr": Cast2MXFp4e2m1BlockNOSR,
    "nvfp4e2m1bnosr": Cast2NVFp4e2m1BlockNOSR,
    "hif4": Cast2Hif4,
    "hif4sr": Cast2Hif4SR,
    "hif4ssr": Cast2Hif4SSR,
    "fp6e3m2": Cast2Fp6e3m2,
    "fp6e3m2b": Cast2Fp6e3m2Block,
    "fp8e4m3": Cast2Fp8e4m3,
    "fp8e4m3b": Cast2Fp8e4m3Block,
    "fp32": Cast2Fp32,
    "1p58bit": WeightQuant,
}



if __name__ == "__main__":
    x = torch.load("/inspire/hdd/project/yunweiyuhuifu/p-shangli/quant/gpt/visual_ckpt/baseline/warmup_linear_weight.pt")
    print(x.shape)
    x = x.view(-1, 2048)
    # x = torch.randn([768, 768])
    # u, s, v = torch.linalg.svd(x)
    # s[0] *= 15
    # s[1] *= 8
    # s[2] *= 7
    # s[4] *= 6
    # x = u @ torch.diag(s) @ v
    
    print(x)
    s = Cast2NVFp4e2m1BlockNOSR.get_scalar(x)
    qx = Cast2NVFp4e2m1BlockNOSR.quant(x, s)
    qx = Cast2NVFp4e2m1BlockNOSR.rquant(qx, s)
    

    print((x - qx) / x.norm())
    me = x.mean(dim=0).repeat(1, x.shape[0]).view(x.shape[0], -1)
    me *= x.mean(dim=1, keepdim=True) 
    # me = x.mean(dim=0, keepdim=True).repeat(x.shape[0])
    x1 = x - me
    s = Cast2NVFp4e2m1BlockNOSR.get_scalar(x1)
    qx1 = Cast2NVFp4e2m1BlockNOSR.quant(x1, s)
    qx1 = Cast2NVFp4e2m1BlockNOSR.rquant(qx1, s)
    
    qx1 += me

    print((x - qx1).norm() / (x).norm())    
    # print(qx / s)