"""
PC-RF velocity network: a conditional U-Net with sinusoidal time embedding,
FiLM conditioning, and a lightweight Sentinel-2 (ResNet) encoder.

forward(x_t, t, cond) -> velocity, where:
  x_t  : [B, C, H, W]   noisy/interpolated fine field
  t    : [B]            flow time in [0, 1]
  cond : [B, Cc, H, W]  conditioning image (upsampled coarse field, optionally
                        concatenated with the satellite embedding)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] in [0,1] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1)
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """Feature-wise linear modulation: y = gamma(c) * x + beta(c)."""
    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, feat_dim)
        self.to_shift = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g = self.to_scale(c)[:, :, None, None]
        b = self.to_shift(c)[:, :, None, None]
        return (1 + g) * x + b


class SatelliteEncoder(nn.Module):
    """Small ResNet-style encoder mapping Sentinel-2 bands+indices to an
    embedding at the target resolution. Output channels = embed_ch."""
    def __init__(self, in_ch: int, embed_ch: int = 32):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, embed_ch, 3, padding=1)
        self.blocks = nn.Sequential(
            nn.GroupNorm(8, embed_ch), nn.SiLU(),
            nn.Conv2d(embed_ch, embed_ch, 3, padding=1),
            nn.GroupNorm(8, embed_ch), nn.SiLU(),
            nn.Conv2d(embed_ch, embed_ch, 3, padding=1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = self.stem(s)
        return h + self.blocks(h)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.mha = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        y, _ = self.mha(y, y, y)
        return x + y.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    """Conditional U-Net velocity field.

    Args:
        in_ch:    channels of x_t (the fine field).
        cond_ch:  channels of the conditioning image.
        base:     base channel width.
        ch_mults: per-level channel multipliers.
        attn_at:  level indices (from the bottom) that use self-attention.
    """
    def __init__(self, in_ch: int, cond_ch: int, base: int = 64,
                 ch_mults=(1, 2, 4), attn_at=(2,), temb_dim: int = 256):
        super().__init__()
        self.temb_mlp = nn.Sequential(
            nn.Linear(base, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim)
        )
        self.temb_dim_in = base
        self.in_conv = nn.Conv2d(in_ch + cond_ch, base, 3, padding=1)
        self.film_in = FiLM(temb_dim, base)

        # ---- encoder ----
        chs = [base]
        ch = base
        self.downs = nn.ModuleList()
        for i, m in enumerate(ch_mults):
            out = base * m
            self.downs.append(nn.ModuleList([
                ResBlock(ch, out, temb_dim),
                AttnBlock(out) if i in attn_at else nn.Identity(),
            ]))
            chs.append(out)
            ch = out

        # ---- bottleneck ----
        self.mid1 = ResBlock(ch, ch, temb_dim)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, temb_dim)

        # ---- decoder ----
        self.ups = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mults))):
            out = base * m
            self.ups.append(nn.ModuleList([
                ResBlock(ch + chs.pop(), out, temb_dim),
                AttnBlock(out) if i in attn_at else nn.Identity(),
            ]))
            ch = out

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        temb = self.temb_mlp(sinusoidal_embedding(t, self.temb_dim_in))
        h = self.in_conv(torch.cat([x_t, cond], dim=1))
        h = self.film_in(h, temb)

        skips, sizes = [], []
        for res, attn in self.downs:
            h = attn(res(h, temb))
            skips.append(h)
            sizes.append(h.shape[-2:])
            h = self.pool(h)

        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)

        for (res, attn), skip, size in zip(self.ups, reversed(skips),
                                           reversed(sizes)):
            h = F.interpolate(h, size=size, mode="nearest")
            h = attn(res(torch.cat([h, skip], dim=1), temb))

        return self.out_conv(F.silu(self.out_norm(h)))
