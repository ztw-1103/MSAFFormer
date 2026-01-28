import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops.layers.torch import Rearrange
from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(1, dim // reduction)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, mid, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        # x, pattn1: [B, C, H, W]
        x = x.unsqueeze(2)          # [B, C, 1, H, W]
        pattn1 = pattn1.unsqueeze(2)# [B, C, 1, H, W]
        x2 = torch.cat([x, pattn1], dim=2)  # [B, C, 2, H, W]
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class HAFM(nn.Module):
    """
    Haze-Aware Feature Modulator
    输入/输出: [B, C, H, W]  -> [B, C, H, W]（残差连接）
    组成:
      - 轻量 transmission 估计: t_hat ∈ (0,1)
      - 通道门控: GAP -> MLP -> sigmoid
      - 空间门控: concat(avg,max, 1-t_hat) -> 7x7 conv -> sigmoid
      - 高频增强: depthwise 3x3 高通 + 1x1 投影
      - 残差: y = x + conv( (1+gate_c)*x + gate_s*highfreq )
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(4, channels // reduction)

        # 1) 粗传输估计（仅用于特征调制）
        self.t_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 2) 通道门控 (CBAM风格, 但受 t_hat 影响)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid()
        )

        # 3) 空间门控（将 1 - t_hat 作为附加通道）
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3, bias=True),  # [avg, max, 1-t_hat]
            nn.Sigmoid()
        )

        # 4) 轻量高通分支（可学习深度可分离 + 投影）
        self.dw_hpf = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        with torch.no_grad():
            # 初始化为拉普拉斯近似核 (0,-1,0;-1,4,-1;0,-1,0)
            k = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
            k = k.view(1,1,3,3).repeat(channels,1,1,1)
            self.dw_hpf.weight.copy_(k)
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)

        # 5) 输出融合卷积
        self.out_conv = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        # 6) Layerscale（可选）
        self.gamma = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        # t_hat ∈ (0,1)，雾厚 ≈ 低 t
        t_hat = self.t_head(x)           # [B,1,H,W]
        inv_t = 1.0 - t_hat

        # 通道门控
        gate_c = self.mlp(self.gap(x))   # [B,C,1,1]

        # 空间门控：把 avg/max/1-t_hat 拼一起
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        gate_s = self.spatial(torch.cat([avg, mx, inv_t], dim=1))  # [B,1,H,W]

        # 高频增强分支
        hpf = self.proj(self.dw_hpf(x))

        # 融合 & 残差
        y = (1.0 + gate_c) * x + gate_s * hpf
        y = self.out_conv(y)

        return x + self.gamma * y, t_hat


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out
class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # x, y: [B, C, H, W] with same C
        initial = x + y
        cattn = self.ca(initial)         # [B, C, 1, 1] -> broadcast
        sattn = self.sa(initial)         # [B, 1, H, W] -> broadcast
        pattn1 = sattn + cattn           # [B, C, H, W]
        pattn2 = self.sigmoid(self.pa(initial, pattn1))  # [B, C, H, W]
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

class Fusion(nn.Module):
    def __init__(self, dim, reduction=8, height=2):
        super(Fusion, self).__init__()
        self.cga = CGAFusion(dim, reduction)
        self.sk = SKFusion(dim, height=height, reduction=reduction)

    def forward(self, x, y):

        out_cga = self.cga(x, y)          # [B, C, H, W]
        out = self.sk([out_cga, y])
        return out

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias



class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim, d=4):
        super().__init__()
        self.dw5  = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.dw7d = nn.Conv2d(dim, dim, 7, padding=3*d, dilation=d, groups=dim, bias=False)
        self.pw   = nn.Conv2d(dim, dim, 1, bias=True)
    def forward(self, x):
        attn = self.pw(self.dw7d(self.dw5(x)))
        return x * attn


class AEstimator(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//4, 1), nn.ReLU(True),
            nn.Conv2d(c//4, 3, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.head(x)  # [B,3,1,1]


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads):  # ← 移除 bias 形参
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm_mlp = LayerNorm(dim, LayerNorm_type)  # ★ 卷积分支单独的LN

        # dim -> 2*dim，再DW-Conv；SimpleGate回到dim
        self.conv1x1 = nn.Conv2d(dim, 2*dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv3x3 = nn.Conv2d(2*dim, 2*dim, kernel_size=3, stride=1, padding=1, groups=2*dim, bias=bias)
        self.sg = SimpleGate()

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        # ★ LayerScale：小系数更稳（也可设为1.0）
        self.gamma_attn = nn.Parameter(torch.ones(1) * 1.0)
        self.gamma_mlp  = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        # Self-Attention 残差
        x = x + self.gamma_attn * self.attn(self.norm1(x))

        # 卷积分支残差（用独立LN）
        residual = x
        y = self.conv1x1(self.norm_mlp(x))
        y = self.conv3x3(y)
        y = self.sg(y)
        x = residual + y

        # FFN 残差
        x = x + self.gamma_mlp * self.ffn(self.norm2(x))
        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)





class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


##########################################################################
class MSAFFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False
                 ):
        super(MSAFFormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[3])])

        # Decoder Level 3
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3 (8*dim -> 4*dim)
        self.proj_up4_3 = nn.Identity()
        self.fuse_l3 = Fusion(int(dim * 2 ** 2), reduction=8)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])

        # Decoder Level 2
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2 (4*dim -> 2*dim)
        self.proj_up3_2 = nn.Identity()
        self.fuse_l2 = Fusion(int(dim * 2 ** 1), reduction=8)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])

        # Decoder Level 1
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1 (2*dim -> 1*dim)
        self.fuse_l1 = Fusion(int(dim * 1), reduction=8)
        self.expand_l1 = nn.Conv2d(int(dim * 1), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])

        self.lka_l1 = LKA(int(dim * 2))
        self.hafm_l1 = HAFM(int(dim*2**1), reduction=8)

        self.a_head = AEstimator(int(dim * 2))
        self.phys_proj = nn.Conv2d(3, int(dim * 2), kernel_size=1, bias=True)


        nn.init.zeros_(self.phys_proj.weight)
        nn.init.zeros_(self.phys_proj.bias)

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.se_enc1 = SELayer(dim)
        self.se_enc2 = SELayer(int(dim * 2 ** 1))
        self.se_enc3 = SELayer(int(dim * 2 ** 2))

    def forward(self, inp_img):
        # Encoder
        inp_enc_level1 = self.patch_embed(inp_img)  # [B, 1*dim, H, W]
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level1 = self.se_enc1(out_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)  # [B, 2*dim, H/2, W/2]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2 = self.se_enc2(out_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)  # [B, 4*dim, H/4, W/4]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        out_enc_level3 = self.se_enc3(out_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)       # [B, 8*dim, H/8, W/8]
        latent = self.latent(inp_enc_level4)                # [B, 8*dim, H/8, W/8]

        # Decoder Level 3
        up_l3 = self.up4_3(latent)                          # [B, 4*dim, H/4, W/4]
        up_l3 = self.proj_up4_3(up_l3)                      #
        fused_l3 = self.fuse_l3(up_l3, out_enc_level3)      # [B, 4*dim, H/4, W/4]
        out_dec_level3 = self.decoder_level3(fused_l3)

        # Decoder Level 2
        up_l2 = self.up3_2(out_dec_level3)                  # [B, 2*dim, H/2, W/2]
        up_l2 = self.proj_up3_2(up_l2)
        fused_l2 = self.fuse_l2(up_l2, out_enc_level2)      # [B, 2*dim, H/2, W/2]
        out_dec_level2 = self.decoder_level2(fused_l2)

        # Decoder Level 1
        up_l1 = self.up2_1(out_dec_level2)                  # [B, 1*dim, H, W]
        fused_l1 = self.fuse_l1(up_l1, out_enc_level1)      # [B, 1*dim, H, W]
        inp_dec_level1 = self.expand_l1(fused_l1)           # [B, 2*dim, H, W]
        out_dec_level1 = self.decoder_level1(inp_dec_level1)


        out_dec_level1 = self.lka_l1(out_dec_level1)
        out_dec_level1, t_hat_l1 = self.hafm_l1(out_dec_level1)  # t_hat_l1: [B,1,H,W]


        A = self.a_head(out_dec_level1)  # [B,3,1,1]


        H, W = out_dec_level1.shape[-2:]
        if inp_img.shape[-2:] != (H, W):
            inp_resized = F.interpolate(inp_img, size=(H, W), mode='bilinear', align_corners=False)
        else:
            inp_resized = inp_img


        eps = 1e-3
        J_hint = (inp_resized - A * (1 - t_hat_l1)).clamp(0, 1) / t_hat_l1.clamp(eps, 1)
        phys_feat = 0.5 * self.phys_proj(J_hint - inp_resized)
        out_dec_level1 = out_dec_level1 + phys_feat

        # refinement
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1






if __name__ == '__main__':
    a =MSAFFormer()
    print(a)