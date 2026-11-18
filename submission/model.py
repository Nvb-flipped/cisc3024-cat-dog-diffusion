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

    def forward(self, x, *, identity_attention=False):
        b, c, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(
            b, 3, self.heads, c // self.heads, height * width).unbind(1)
        if identity_attention:
            return x + self.proj(v.reshape(b, c, height, width))
        weights = torch.softmax(torch.matmul(q.transpose(-1, -2), k) / math.sqrt(c // self.heads), dim=-1)
        h = torch.matmul(v, weights.transpose(-1, -2)).reshape(b, c, height, width)
        return x + self.proj(h)


def mix_haar_subbands(packed):
    """Apply the symmetric, orthonormal 2x2 Haar transform to packed pixels."""
    batch, channels, height, width = packed.shape
    a, b, c, d = packed.reshape(batch, channels // 4, 4, height, width).unbind(dim=2)
    bands = torch.stack((a + b + c + d, a - b + c - d,
                         a + b - c - d, a - b - c + d), dim=2)
    return (bands * .5).reshape_as(packed)


class ConditionalUNet(nn.Module):
    def __init__(self, channels=(48, 96, 144, 192), blocks_per_level=2,
                 conditioning_dim=256, attention_heads=4, dropout=0.10, patch_size=1,
                 patch_detail=False, self_condition=False, haar_packing=False, freeu=False,
                 freeu_strength=1.0, encoder_attention16=False):
        super().__init__()
        if len(channels) != 4 or any(c % 8 for c in channels):
            raise ValueError('Four channel levels divisible by 8 required')
        if conditioning_dim % 2:
            raise ValueError('conditioning_dim must be even')
        if type(patch_size) is not int or patch_size not in (1, 2, 4, 8):
            raise ValueError('patch_size must be 1, 2, 4, or 8')
        if type(haar_packing) is not bool or (haar_packing and patch_size != 2):
            raise ValueError('Haar packing requires a 2 by 2 patch')
        if patch_detail and patch_size == 1:
            raise ValueError('patch_detail requires a patch grid')
        if type(freeu) is not bool or (freeu and patch_size != 1):
            raise ValueError('FreeU requires the pixel-space decoder')
        if not 1 <= freeu_strength <= 2.2 or (freeu_strength != 1 and not freeu):
            raise ValueError('FreeU strength must be between 1 and 2.2 and requires FreeU')
        if type(encoder_attention16) is not bool or (encoder_attention16 and patch_size != 1):
            raise ValueError('16x16 encoder attention requires the pixel-space model')
        self.config = dict(channels=list(channels), blocks_per_level=blocks_per_level,
                           conditioning_dim=conditioning_dim, attention_heads=attention_heads,
                           dropout=dropout, patch_size=patch_size, patch_detail=patch_detail,
                           self_condition=self_condition)
        self.patch_size = patch_size
        self.haar_packing = haar_packing
        if haar_packing:
            self.config['haar_packing'] = True
        self.self_condition = self_condition
        self.freeu = freeu
        if freeu:
            self.config['freeu'] = True
        if freeu_strength != 1:
            self.config['freeu_strength'] = freeu_strength
        self.freeu_decoder_scales = ((1.08, .90), (1.12, .80))
        if freeu_strength != 1:
            self.freeu_decoder_scales = tuple(
                (1 + (backbone - 1) * freeu_strength, 1 + (skip - 1) * freeu_strength)
                for backbone, skip in self.freeu_decoder_scales)
        # Use the same two spatial reductions for both patch grids. The deepest
        # transition still has a learned convolution, so 4×4 tokens can reach 1×1.
        self.spatial_reductions = 3 if patch_size <= 2 else 2
        patch_channels = 3 * patch_size ** 2
        self.embedding_dim = conditioning_dim
        self.time_mlp = nn.Sequential(nn.Linear(conditioning_dim, conditioning_dim), nn.SiLU(),
                                      nn.Linear(conditioning_dim, conditioning_dim))
        self.class_embedding = nn.Embedding(3, conditioning_dim)  # 2 is null conditioning.
        self.input = nn.Conv2d(patch_channels * (2 if self_condition else 1), channels[0], 3, padding=1)
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
        self.encoder_attention16 = SpatialAttention(channels[1], attention_heads) if encoder_attention16 else None
        if encoder_attention16:
            # Initialize after the shared layers to preserve their seeded initialization.
            self.config['encoder_attention16'] = True

    def forward(self, x, timesteps, labels, clean_hint=None, *, perturb_encoder_attention=False,
                class_noise=None, class_noise_fraction=0.0):
        embedding = self.class_embedding(labels)
        if class_noise is not None:
            spread = embedding.std(dim=-1, keepdim=True, unbiased=False)
            noisy = (math.sqrt(1 - class_noise_fraction) * embedding
                     + .5 * math.sqrt(class_noise_fraction) * spread * class_noise)
            noisy_spread = noisy.std(dim=-1, keepdim=True, unbiased=False)
            corrected = (noisy - noisy.mean(dim=-1, keepdim=True)) * (
                spread / noisy_spread.clamp_min(1e-6)) + embedding.mean(dim=-1, keepdim=True)
            embedding = torch.where(labels[:, None] == 2, embedding, corrected)
        condition = self.time_mlp(timestep_embedding(timesteps, self.embedding_dim)) + embedding
        model_input = x
        if self.self_condition:
            hint = torch.zeros_like(x) if clean_hint is None else clean_hint
            model_input = torch.cat((x, hint), dim=1)
        packed = F.pixel_unshuffle(model_input, self.patch_size) if self.patch_size > 1 else model_input
        if self.haar_packing:
            packed = mix_haar_subbands(packed)
        h = self.input(packed)
        skips = []
        for level, stage in enumerate(self.encoder):
            for block in stage['blocks']:
                h = block(h, condition)
                skips.append(h)
            if perturb_encoder_attention and level == 2:
                h = stage['attention'](h, identity_attention=True)
            else:
                h = stage['attention'](h)
            if level == 1 and self.encoder_attention16 is not None:
                h = self.encoder_attention16(h)
            if level < 3:
                h = self.downsample[level](h)
        h = self.middle2(self.middle_attention(self.middle1(h, condition)), condition)
        for index, stage in enumerate(self.decoder):
            rebalance = self.freeu and index < 2
            if rebalance:
                backbone_scale, skip_scale = self.freeu_decoder_scales[index]
                first_half, second_half = h.chunk(2, dim=1)
                h = torch.cat((first_half * backbone_scale, second_half), dim=1)
                # Attenuate the DC and adjacent Fourier modes in each skip independently.
                height, width = h.shape[-2:]
                fy = torch.fft.fftfreq(height, device=h.device).abs()
                fx = torch.fft.rfftfreq(width, device=h.device).abs()
                low = (fy[:, None] <= 1 / height) & (fx[None, :] <= 1 / width)
                skip_filter = torch.where(low, skip_scale, 1.0)
            for block in stage['blocks']:
                skip = skips.pop()
                if rebalance:
                    spectrum = torch.fft.rfft2(skip.float())
                    skip = torch.fft.irfft2(spectrum * skip_filter, s=(height, width)).to(skip.dtype)
                h = block(torch.cat([h, skip], dim=1), condition)
            h = stage['attention'](h)
            if index < 3:
                if 2 - index < self.spatial_reductions:
                    h = F.interpolate(h, scale_factor=2, mode='nearest')
                h = self.upsample[index](h)
        output = self.output(F.silu(self.output_norm(h)))
        if self.haar_packing:
            if output.dtype in (torch.float16, torch.bfloat16):
                output = output.float()
            output = mix_haar_subbands(output)
        if self.patch_size > 1:
            output = F.pixel_shuffle(output, self.patch_size)
        if self.detail_input is not None:
            detail = self.detail_input(torch.cat((x, output), dim=1))
            for block in self.detail:
                detail = block(detail, condition)
            output = output + self.detail_output(F.silu(detail))
        return output
