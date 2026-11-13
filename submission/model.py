"""Small pixel-space class-conditional noise-prediction U-Net, trained from scratch."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps, dimension):
    half = dimension // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(conditioning_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(F.silu(condition)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(self.dropout(F.silu(h)))


class SpatialAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by attention_heads')
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(
            b, 3, self.heads, c // self.heads, height * width).unbind(1)
        weights = torch.softmax(torch.matmul(q.transpose(-1, -2), k) / math.sqrt(c // self.heads), dim=-1)
        h = torch.matmul(v, weights.transpose(-1, -2)).reshape(b, c, height, width)
        return x + self.proj(h)


class ConditionalUNet(nn.Module):
    def __init__(self, channels=(48, 96, 144, 192), blocks_per_level=2,
                 conditioning_dim=256, attention_heads=4, dropout=0.10, patch_size=1,
                 patch_detail=False):
        super().__init__()
        if len(channels) != 4 or any(c % 8 for c in channels):
            raise ValueError('Four channel levels divisible by 8 required')
        if conditioning_dim % 2:
            raise ValueError('conditioning_dim must be even')
        if type(patch_size) is not int or patch_size not in (1, 4, 8):
            raise ValueError('patch_size must be 1, 4, or 8')
        if patch_detail and patch_size == 1:
            raise ValueError('patch_detail requires a patch grid')
        self.config = dict(channels=list(channels), blocks_per_level=blocks_per_level,
                           conditioning_dim=conditioning_dim, attention_heads=attention_heads,
                           dropout=dropout, patch_size=patch_size, patch_detail=patch_detail)
        self.patch_size = patch_size
        # Use the same two spatial reductions for both patch grids. The deepest
        # transition still has a learned convolution, so 4×4 tokens can reach 1×1.
        self.spatial_reductions = 3 if patch_size == 1 else 2
        patch_channels = 3 * patch_size ** 2
        self.embedding_dim = conditioning_dim
        self.time_mlp = nn.Sequential(nn.Linear(conditioning_dim, conditioning_dim), nn.SiLU(),
                                      nn.Linear(conditioning_dim, conditioning_dim))
        self.class_embedding = nn.Embedding(3, conditioning_dim)  # 2 is null conditioning.
        self.input = nn.Conv2d(patch_channels, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        current = channels[0]
        for level, out_channels in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(ResidualBlock(current, out_channels, conditioning_dim, dropout))
                current = out_channels
            attention = SpatialAttention(current, attention_heads) if level >= 2 else nn.Identity()
            self.encoder.append(nn.ModuleDict({'blocks': blocks, 'attention': attention}))
            if level < 3:
                stride = 2 if level < self.spatial_reductions else 1
                self.downsample.append(nn.Conv2d(current, current, 3, stride=stride, padding=1))
        self.middle1 = ResidualBlock(current, current, conditioning_dim, dropout)
        self.middle_attention = SpatialAttention(current, attention_heads)
        self.middle2 = ResidualBlock(current, current, conditioning_dim, dropout)
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for level in reversed(range(4)):
            out_channels = channels[level]
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(ResidualBlock(current + out_channels, out_channels, conditioning_dim, dropout))
                current = out_channels
            attention = SpatialAttention(current, attention_heads) if level >= 2 else nn.Identity()
            self.decoder.append(nn.ModuleDict({'blocks': blocks, 'attention': attention}))
            if level > 0:
                self.upsample.append(nn.Conv2d(current, current, 3, padding=1))
        self.output_norm = nn.GroupNorm(8, current)
        self.output = nn.Conv2d(current, patch_channels, 3, padding=1)
        if patch_detail:
            # Retain local pixel structure as the packed branch learns whole-head context.
            self.detail_input = nn.Conv2d(6, 24, 3, padding=1)
            self.detail = nn.ModuleList([ResidualBlock(24, 24, conditioning_dim, dropout)
                                         for _ in range(2)])
            self.detail_output = nn.Conv2d(24, 3, 3, padding=1)
        else:
            self.detail_input = None

    def forward(self, x, timesteps, labels):
        condition = self.time_mlp(timestep_embedding(timesteps, self.embedding_dim)) + self.class_embedding(labels)
        packed = F.pixel_unshuffle(x, self.patch_size) if self.patch_size > 1 else x
        h = self.input(packed)
        skips = []
        for level, stage in enumerate(self.encoder):
            for block in stage['blocks']:
                h = block(h, condition)
                skips.append(h)
            h = stage['attention'](h)
            if level < 3:
                h = self.downsample[level](h)
        h = self.middle2(self.middle_attention(self.middle1(h, condition)), condition)
        for index, stage in enumerate(self.decoder):
            for block in stage['blocks']:
                h = block(torch.cat([h, skips.pop()], dim=1), condition)
            h = stage['attention'](h)
            if index < 3:
                if 2 - index < self.spatial_reductions:
                    h = F.interpolate(h, scale_factor=2, mode='nearest')
                h = self.upsample[index](h)
        output = self.output(F.silu(self.output_norm(h)))
        if self.patch_size > 1:
            output = F.pixel_shuffle(output, self.patch_size)
        if self.detail_input is not None:
            detail = self.detail_input(torch.cat((x, output), dim=1))
            for block in self.detail:
                detail = block(detail, condition)
            output = output + self.detail_output(F.silu(detail))
        return output
