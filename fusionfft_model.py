import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


# -------------------------
# Core building blocks
# -------------------------

class AdaptiveFrequencyDecoupling(nn.Module):
    """
    Frequency split: learn a soft low-pass via channel-wise gating + depthwise conv.
    Returns (low_freq, high_freq) so detail can be preserved downstream.
    """
    def __init__(self, channels, groups=2):
        super().__init__()
        assert channels % groups == 0, "channels must be divisible by groups in low_freq_conv"
        self.channels = channels
        self.groups = groups

        self.freq_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # depthwise-ish smoothing for structure
        self.low_freq_conv = nn.Conv2d(channels, channels, 3, 1, 1, groups=groups)

    def forward(self, x):
        w = self.freq_weight(x)
        low = self.low_freq_conv(x) * w
        high = x - low
        return low, high


class LearnedUpsampling(nn.Module):
    """
    Learned upsampling with pixel shuffle; robust fallback for arbitrary target sizes.
    Ensures output has out_channels in both paths.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.refine = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        # Fallback projection when target scale does not match expected factor
        self.fallback_proj = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, target_size=None):
        if target_size is not None:
            scale_h = target_size[0] / x.shape[2]
            scale_w = target_size[1] / x.shape[3]
            close_to_factor = (
                abs(scale_h - self.scale_factor) < 0.1 and
                abs(scale_w - self.scale_factor) < 0.1
            )
            if close_to_factor:
                x = self.conv(x)
                x = self.pixel_shuffle(x)
                x = self.refine(x)
            else:
                # map channels first, then resize, then refine
                x = self.fallback_proj(x)
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                x = self.refine(x)
        else:
            x = self.conv(x)
            x = self.pixel_shuffle(x)
            x = self.refine(x)
        return x


class DetailPreservingAttention(nn.Module):
    """
    Dual-branch attention that weights low- and high-frequency cues.
    """
    def __init__(self, channels):
        super().__init__()
        self.low_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1)
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1)
        )
        self.mix_weight = nn.Parameter(torch.tensor([0.7, 0.3]))

    def forward(self, x, low_freq, high_freq):
        low_attn = torch.sigmoid(self.low_branch(low_freq))
        high_attn = torch.sigmoid(self.high_branch(high_freq))
        w = F.softmax(self.mix_weight, dim=0)
        attn = w[0] * low_attn + w[1] * high_attn
        return x * attn


class ConvBlock(nn.Module):
    """
    Conv + optional GN/IN/BN + activation. GroupNorm groups made safe for small channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 norm='none', activation='relu', bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.norm = None
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'gn':
            num_groups = max(1, min(32, out_channels // 4))
            while out_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, out_channels)

        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class DetailAwareResBlock(nn.Module):
    """
    Residual block with optional frequency-aware attention.
    """
    def __init__(self, channels, norm='gn', preserve_details=True):
        super().__init__()
        self.preserve_details = preserve_details
        self.conv1 = ConvBlock(channels, channels, norm=norm, activation='leaky')
        self.conv2 = ConvBlock(channels, channels, norm=norm, activation='none')
        if preserve_details:
            self.freq = AdaptiveFrequencyDecoupling(channels)
            self.dpa = DetailPreservingAttention(channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.residual_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        r = x
        y = self.conv1(x)
        y = self.conv2(y)
        if self.preserve_details:
            low, high = self.freq(y)
            y = self.dpa(y, low, high)
        y = y + r * self.residual_weight
        return self.activation(y)


# -------------------------
# Cross-anchor fusion
# -------------------------

class PyramidCrossAttention(nn.Module):
    """
    Hierarchical cross-anchor attention with frequency-aware queries
    and a learnable post-upsample refinement.
    """
    def __init__(self, channels, num_heads=4, max_attention_size=64*64):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // max(1, num_heads)
        self.max_attention_size = max_attention_size

        self.freq_decomp = AdaptiveFrequencyDecoupling(channels)

        # per-scale qkv projections
        possible_scales = [64, 32, 16, 8, 4, 2]
        self.attn_modules = nn.ModuleDict()
        for s in possible_scales:
            self.attn_modules[str(s)] = nn.ModuleDict({
                'q': nn.Conv2d(channels, channels, 1),
                'k': nn.Conv2d(channels, channels, 1),
                'v': nn.Conv2d(channels, channels, 1),
            })

        # learned fusers
        self.fw2 = nn.Parameter(torch.ones(2) / 2)
        self.fw3 = nn.Parameter(torch.ones(3) / 3)
        self.fw4 = nn.Parameter(torch.ones(4) / 4)
        self.fuse2 = nn.Conv2d(channels * 2, channels, 1)
        self.fuse3 = nn.Conv2d(channels * 3, channels, 1)
        self.fuse4 = nn.Conv2d(channels * 4, channels, 1)

        # light refinement after upsampling pooled attention outputs
        self.post_upsample_refine = nn.Conv2d(channels, channels, 3, 1, 1)

        # fallback scale when none selected externally
        self.hf_scale = nn.Parameter(torch.tensor(0.3))

    def _scales_for(self, H, W):
        area = H * W
        min_scale = max(2, int(np.sqrt(area / self.max_attention_size)))
        scales = []
        if min_scale <= 64:
            scales.append(min_scale)
        if min_scale * 2 <= 64:
            scales.append(min_scale * 2)
        if min_scale * 4 <= 64 and len(scales) < 3:
            scales.append(min_scale * 4)
        if len(scales) < 2:
            scales = [16, 8] if min_scale < 64 else [64, 32]
        scales = [s for s in scales[:4] if str(s) in self.attn_modules]
        if not scales:
            scales = [16, 8]
        return sorted(scales, reverse=True)

    def forward(self, query, keys, values, hf_res_scale: torch.Tensor = None):
        """
        query: [B,C,H,W]
        keys, values: [B,N,C,H,W]
        """
        B, C, H, W = query.shape
        N = keys.shape[1]

        q_low, q_high = self.freq_decomp(query)
        scales = self._scales_for(H, W)
        collected = []

        for s in scales:
            am = self.attn_modules[str(s)]
            if s > 1:
                Hs = max(1, H // s)
                Ws = max(1, W // s)

                q_s = F.adaptive_avg_pool2d(q_low, (Hs, Ws))

                k_in = keys.view(B * N, C, H, W)
                v_in = values.view(B * N, C, H, W)
                k_s = F.adaptive_avg_pool2d(k_in, (Hs, Ws)).view(B, N, C, Hs, Ws)
                v_s = F.adaptive_avg_pool2d(v_in, (Hs, Ws)).view(B, N, C, Hs, Ws)
            else:
                q_s, k_s, v_s = query, keys, values
                Hs, Ws = H, W

            q = am['q'](q_s)
            heads = self.num_heads if C % self.num_heads == 0 else 1
            d = C // heads
            q = q.view(B, heads, d, Hs * Ws).transpose(-2, -1)  # [B,heads,HW,d]^T

            attn_out = []
            for i in range(N):
                k = am['k'](k_s[:, i]).view(B, heads, d, Hs * Ws)
                v = am['v'](v_s[:, i]).view(B, heads, d, Hs * Ws).transpose(-2, -1)

                scores = torch.matmul(q, k) * (d ** -0.5)
                scores = scores - scores.max(dim=-1, keepdim=True)[0]
                a = F.softmax(scores, dim=-1)
                o = torch.matmul(a, v)
                attn_out.append(o)

            if len(attn_out) > 0:
                combined = torch.stack(attn_out, dim=1).mean(dim=1)  # [B,heads,HW,d]
                combined = combined.transpose(-2, -1).contiguous().view(B, C, Hs, Ws)
            else:
                combined = q_s

            if s > 1:
                combined = F.interpolate(combined, size=(H, W), mode='nearest')
                combined = self.post_upsample_refine(combined)

            collected.append(combined)

        k = len(collected)
        if k == 1:
            out = collected[0]
        elif k == 2:
            w = F.softmax(self.fw2, dim=0)
            out = self.fuse2(torch.cat([collected[0] * w[0], collected[1] * w[1]], dim=1))
        elif k == 3:
            w = F.softmax(self.fw3, dim=0)
            out = self.fuse3(torch.cat([collected[i] * w[i] for i in range(3)], dim=1))
        else:
            w = F.softmax(self.fw4, dim=0)
            out = self.fuse4(torch.cat([collected[i] * w[i] for i in range(4)], dim=1))

        scale = hf_res_scale if hf_res_scale is not None else self.hf_scale
        out = out + q_high * scale
        return out


# -------------------------
# Utility attention blocks
# -------------------------

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        a = self.conv(x)
        return x * a


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# -------------------------
# Warping and weighting
# -------------------------

class FlowWarping(nn.Module):
    """Backward warping: sample img at grid + flow."""
    def forward(self, img, flow):
        B, C, H, W = img.size()
        xx = torch.arange(0, W, device=img.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
        yy = torch.arange(0, H, device=img.device).view(1, 1, H, 1).repeat(B, 1, 1, W)
        grid = torch.cat((xx, yy), 1).float()  # [B,2,H,W]

        vgrid = grid + flow
        vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
        vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        return F.grid_sample(img, vgrid, align_corners=True)


class TemporalWeightingMLP(nn.Module):
    """Turns per-anchor time scalars into a softmax over anchors."""
    def __init__(self, num_anchors=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_anchors, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_anchors),
            nn.Softmax(dim=1)
        )
    def forward(self, timesteps):
        return self.net(timesteps)


# -------------------------
# Noise-preserving utilities
# -------------------------

def _gaussian_kernel2d(channels: int, sigma: float):
    k = int(2 * math.ceil(3 * sigma) + 1)
    xs = torch.arange(k, dtype=torch.float32) - k // 2
    g1d = torch.exp(-0.5 * (xs / sigma) ** 2)
    g1d = g1d / g1d.sum().clamp_min(1e-8)
    g2d = torch.outer(g1d, g1d)
    weight = g2d.view(1, 1, k, k).repeat(channels, 1, 1, 1)
    return weight, k // 2


class NoiseInject(nn.Module):
    """
    Extract a band-pass residual from the warped prior and inject it with a learned gate.
    """
    def __init__(self, in_rgb=3, feat_channels=64, sigma_lo=0.6, sigma_hi=1.6):
        super().__init__()
        w_lo, pad_lo = _gaussian_kernel2d(in_rgb, sigma_lo)
        w_hi, pad_hi = _gaussian_kernel2d(in_rgb, sigma_hi)
        self.register_buffer("w_lo", w_lo)
        self.register_buffer("w_hi", w_hi)
        self.pad_lo = pad_lo
        self.pad_hi = pad_hi

        self.shaper = nn.Conv2d(in_rgb, in_rgb, kernel_size=1, bias=True)

        self.gate = nn.Sequential(
            nn.Conv2d(feat_channels + in_rgb, feat_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        self.scale = nn.Parameter(torch.tensor(0.5))

    def extract(self, x_rgb):
        lo = F.conv2d(x_rgb, self.w_hi, padding=self.pad_hi, groups=x_rgb.shape[1])
        hi = F.conv2d(x_rgb, self.w_lo, padding=self.pad_lo, groups=x_rgb.shape[1])
        residual = hi - lo
        return self.shaper(residual)

    def forward(self, feat, prior_rgb):
        bp = self.extract(prior_rgb)
        g = self.gate(torch.cat([feat, bp], dim=1))
        injected = g * self.scale * bp
        return injected, g


class ContentSkip(nn.Module):
    """
    Gated identity skip from prior RGB. Preserves noise where gate is high.
    """
    def __init__(self, feat_channels, in_rgb=3):
        super().__init__()
        self.gate = nn.Sequential(
            ConvBlock(feat_channels, feat_channels // 2, norm='gn', activation='leaky'),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat, prior_rgb):
        a = self.gate(feat)
        mixed = a * prior_rgb
        return mixed, a


# -------------------------
# Decoder helper wrapper
# -------------------------

class DetailFuse(nn.Module):
    """
    Wrapper to make frequency-aware attention plug into nn.Sequential cleanly.
    """
    def __init__(self, channels):
        super().__init__
        self.pre = DetailAwareResBlock(channels, norm='gn', preserve_details=True)
        self.freq = AdaptiveFrequencyDecoupling(channels)
        self.dpa = DetailPreservingAttention(channels)

    def forward(self, x):
        y = self.pre(x)
        low, high = self.freq(y)
        return self.dpa(y, low, high)


# -------------------------
# Main model
# -------------------------

class AnchorFusionNet(nn.Module):
    """
    Multi-anchor fusion with frequency-aware encoding, pyramid cross-attention,
    learned upsampling, and noise-preserving output paths.
    """
    def __init__(self, num_anchors=3, base_channels=64, max_attention_size=96*96):
        super().__init__()
        self.num_anchors = num_anchors
        self.base_channels = base_channels

        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.detail_weight = nn.Parameter(torch.tensor(0.3))

        # spectral swap controls
        self.spectral_alpha = nn.Parameter(torch.tensor(0.3))  # 0..1 blend amount
        self.spectral_lo = 0.32  # start of high-band in Nyquist units (0..0.5)
        self.spectral_hi = 0.50  # end of band
        self.spectral_soft = True

        self.flow_warp = FlowWarping()
        self.temporal_weighter = TemporalWeightingMLP(num_anchors)
        self.temporal_temperature = nn.Parameter(torch.tensor(1.0))

        # Encoder
        self.encoder = nn.ModuleDict({
            'low': nn.Sequential(
                ConvBlock(11, base_channels, 7, 1, 3, norm='none', activation='leaky'),
                ConvBlock(base_channels, base_channels * 2, 3, 2, 1, norm='gn', activation='leaky'),
                DetailAwareResBlock(base_channels * 2, norm='gn', preserve_details=True)
            ),
            'mid': nn.Sequential(
                ConvBlock(base_channels * 2, base_channels * 4, 3, 2, 1, norm='gn', activation='leaky'),
                DetailAwareResBlock(base_channels * 4, norm='gn', preserve_details=True),
                DetailAwareResBlock(base_channels * 4, norm='gn', preserve_details=True)
            ),
            'high': nn.Sequential(
                ConvBlock(base_channels * 4, base_channels * 8, 3, 2, 1, norm='gn', activation='leaky'),
                DetailAwareResBlock(base_channels * 8, norm='gn', preserve_details=True),
                DetailAwareResBlock(base_channels * 8, norm='gn', preserve_details=True),
                DetailAwareResBlock(base_channels * 8, norm='gn', preserve_details=False)
            )
        })

        # Per-anchor frequency adapters at low stage
        self.anchor_adapters = nn.ModuleList([
            nn.Sequential(
                AdaptiveFrequencyDecoupling(base_channels * 2),
                nn.Conv2d(base_channels * 4, base_channels * 2, 1)
            ) for _ in range(num_anchors)
        ])

        # Small refiners for flow and mask
        self.flow_refiners = nn.ModuleList([
            nn.Sequential(
                ConvBlock(4, base_channels, 5, 1, 2, norm='none', activation='leaky'),
                DetailAwareResBlock(base_channels, norm='gn', preserve_details=False),
                ConvBlock(base_channels, base_channels // 2, norm='gn', activation='leaky'),
                ConvBlock(base_channels // 2, 4, activation='none')
            ) for _ in range(num_anchors)
        ])

        self.mask_refiners = nn.ModuleList([
            nn.Sequential(
                ConvBlock(1, base_channels // 2, 5, 1, 2, norm='none', activation='leaky'),
                DetailAwareResBlock(base_channels // 2, norm='gn', preserve_details=False),
                ConvBlock(base_channels // 2, 1, activation='sigmoid')
            ) for _ in range(num_anchors)
        ])

        # Cross-attention at three scales
        self.cross_low = PyramidCrossAttention(base_channels * 2, num_heads=4, max_attention_size=max_attention_size)
        self.cross_mid = PyramidCrossAttention(base_channels * 4, num_heads=4, max_attention_size=max_attention_size)
        self.cross_high = PyramidCrossAttention(base_channels * 8, num_heads=4, max_attention_size=max_attention_size)

        # Decoder
        self.decoder = nn.ModuleDict({
            'up_high_to_mid': nn.Sequential(
                LearnedUpsampling(base_channels * 8, base_channels * 4, scale_factor=2),
                ConvBlock(base_channels * 4, base_channels * 4, norm='gn', activation='leaky')
            ),
            'fuse_mid': nn.Sequential(
                ConvBlock(base_channels * 8, base_channels * 4, norm='gn', activation='leaky'),
                DetailAwareResBlock(base_channels * 4, norm='gn', preserve_details=True),
            ),
            'up_mid_to_low': nn.Sequential(
                LearnedUpsampling(base_channels * 4, base_channels * 2, scale_factor=2),
                ConvBlock(base_channels * 2, base_channels * 2, norm='gn', activation='leaky')
            ),
            'fuse_low': nn.Sequential(
                ConvBlock(base_channels * 4, base_channels * 2, norm='gn', activation='leaky'),
                DetailAwareResBlock(base_channels * 2, norm='gn', preserve_details=True),
            ),
            'up_to_original': nn.Sequential(
                LearnedUpsampling(base_channels * 2, base_channels, scale_factor=2),
                ConvBlock(base_channels, base_channels, norm='gn', activation='leaky'),
                ChannelAttention(base_channels)
            )
        })

        # Context aggregation
        self.context_refine = nn.Conv2d(base_channels * 2 * num_anchors,
                                        base_channels * 2 * num_anchors, 3, 1, 1)
        self.context_aggregator = nn.Sequential(
            ConvBlock(base_channels * 2 * num_anchors, base_channels * 2, 1, norm='gn', activation='leaky'),
            ConvBlock(base_channels * 2, base_channels, 3, 1, 1, norm='gn', activation='leaky')
        )

        # Image heads
        self.synthesis = nn.Sequential(
            ConvBlock(base_channels + 3, base_channels, norm='gn', activation='leaky'),
            DetailAwareResBlock(base_channels, norm='gn', preserve_details=True),
            DetailAwareResBlock(base_channels, norm='gn', preserve_details=True),
            ConvBlock(base_channels, base_channels // 2, norm='gn', activation='leaky'),
            ConvBlock(base_channels // 2, 3, activation='sigmoid')
        )
        self.residual_head = nn.Sequential(
            ConvBlock(base_channels, base_channels // 2, norm='gn', activation='leaky'),
            ConvBlock(base_channels // 2, 3, activation='tanh')
        )

        # New noise-preserving heads
        self.noise_inject = NoiseInject(in_rgb=3, feat_channels=base_channels,
                                        sigma_lo=0.6, sigma_hi=1.6)
        self.content_skip = ContentSkip(feat_channels=base_channels, in_rgb=3)

        self._init_weights()

    def _init_weights(self):
        # Standard inits
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # ICNR for PixelShuffle convs
        def icnr_(w, scale=2, initializer=init.kaiming_normal_):
            oc, ic, k1, k2 = w.shape
            sub = oc // (scale ** 2)
            k = torch.zeros([sub, ic, k1, k2], device=w.device)
            initializer(k)
            k = k.repeat_interleave(scale ** 2, dim=0)
            with torch.no_grad():
                w.copy_(k)
        for m in self.modules():
            if isinstance(m, LearnedUpsampling):
                icnr_(m.conv.weight, scale=m.pixel_shuffle.upscale_factor)

    # ------------- spectral swap utility -------------
    def _spectral_swap(self, base, prior, lo=0.32, hi=0.50, alpha=0.3, soft=True):
        """
        Replace high-band magnitude of base with that of prior, keep base phase.
        lo, hi in [0,0.5]; alpha in [0,1].
        """
        B, C, H, W = base.shape
        X = torch.fft.rfft2(base, dim=(-2, -1), norm="ortho")
        P = torch.fft.rfft2(prior, dim=(-2, -1), norm="ortho")

        mag_x = torch.abs(X)
        mag_p = torch.abs(P)
        phase = torch.angle(X)

        fy = torch.fft.fftfreq(H, d=1.0).to(base.device).abs()
        fx = torch.fft.rfftfreq(W, d=1.0).to(base.device).abs()
        wy, wx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(wx**2 + wy**2)  # 0..~0.707; but we only use up to 0.5 effectively
        if soft:
            # smooth ramp from lo to hi, then 1 beyond hi
            t = ((r - lo) / max(hi - lo, 1e-6)).clamp(0, 1)
            mask = 0.5 - 0.5 * torch.cos(math.pi * t)
        else:
            mask = (r >= lo).float()
        mask = mask.view(1, 1, H, W // 2 + 1)

        # blend magnitudes in high band
        new_mag = mag_x + mask * alpha * (mag_p - mag_x)
        Y = new_mag * torch.exp(1j * phase)
        y = torch.fft.irfft2(Y, s=(H, W), dim=(-2, -1), norm="ortho")
        return y

    def forward(self, I0_all, I1_all, flows_all, masks_all, timesteps, skip_noise_inject=False, skip_content_skip=False):
        """
        I0_all, I1_all: [B,N,3,H,W] in [0,1]
        flows_all: [B,N,4,H,W] with [t->0_x, t->0_y, t->1_x, t->1_y]
        masks_all: [B,N,H,W] or [B,N,1,H,W] in [0,1]
        timesteps: [B,N] real-valued features
        """
        B, N, _, H, W = I0_all.shape

        # temporal weights
        t_weights = self.temporal_weighter(timesteps * self.temporal_temperature)

        warped_imgs = []
        refined_masks = []
        feats_low, feats_mid, feats_high = [], [], []
        context_low = []

        for i in range(N):
            I0 = I0_all[:, i]
            I1 = I1_all[:, i]
            flow = flows_all[:, i]
            mask = masks_all[:, i].unsqueeze(1) if masks_all[:, i].dim() == 3 else masks_all[:, i]

            flow = flow + self.flow_refiners[i](flow)
            f01 = flow[:, :2]
            f10 = flow[:, 2:]

            wI0 = self.flow_warp(I0, f01)
            wI1 = self.flow_warp(I1, f10)

            m = self.mask_refiners[i](mask)
            warped = wI0 * m + wI1 * (1 - m)

            x = torch.cat([I0, I1, warped, f01], dim=1)

            low = self.encoder['low'](x)
            low_l, low_h = self.anchor_adapters[i][0](low)
            low = self.anchor_adapters[i][1](torch.cat([low_l, low_h], dim=1))

            mid = self.encoder['mid'](low)
            high = self.encoder['high'](mid)

            w = t_weights[:, i:i+1, None, None]
            feats_low.append(low * w)
            feats_mid.append(mid * w)
            feats_high.append(high * w)

            warped_imgs.append(warped)
            refined_masks.append(m)
            context_low.append(low)

        low_features = torch.stack(feats_low, dim=1)   # [B,N,C,H/2,W/2]
        mid_features = torch.stack(feats_mid, dim=1)   # [B,N,2C,H/4,W/4]
        high_features = torch.stack(feats_high, dim=1) # [B,N,4C,H/8,W/8]

        # cross-anchor attention
        query_high = high_features.mean(dim=1)
        fused_high = self.cross_high(query_high, high_features, high_features, hf_res_scale=self.detail_weight)

        up_high = self.decoder['up_high_to_mid'][0](fused_high, target_size=mid_features.shape[-2:])
        up_high = self.decoder['up_high_to_mid'][1](up_high)

        query_mid = mid_features.mean(dim=1)
        fused_mid = self.cross_mid(query_mid, mid_features, mid_features, hf_res_scale=self.detail_weight)
        fused_mid = nn.Sequential(*self.decoder['fuse_mid'])(torch.cat([up_high, fused_mid], dim=1))

        up_mid = self.decoder['up_mid_to_low'][0](fused_mid, target_size=low_features.shape[-2:])
        up_mid = self.decoder['up_mid_to_low'][1](up_mid)

        query_low = low_features.mean(dim=1)
        fused_low = self.cross_low(query_low, low_features, low_features, hf_res_scale=self.detail_weight)
        fused_low = nn.Sequential(*self.decoder['fuse_low'])(torch.cat([up_mid, fused_low], dim=1))

        # to full resolution
        if fused_low.shape[-2:] != (H, W):
            decoded = self.decoder['up_to_original'][0](fused_low, target_size=(H, W))
            decoded = self.decoder['up_to_original'][1](decoded)
            decoded = self.decoder['up_to_original'][2](decoded)
        else:
            decoded = nn.Sequential(*self.decoder['up_to_original'])(fused_low)

        # context aggregation from all anchors (low stage)
        context_concat = torch.cat(context_low, dim=1)  # [B, N*2C, H/2, W/2]
        context_up = F.interpolate(context_concat, size=(H, W), mode='nearest')
        context_up = self.context_refine(context_up)
        context_agg = self.context_aggregator(context_up)

        if decoded.shape[-2:] != context_agg.shape[-2:]:
            context_agg = F.interpolate(context_agg, size=decoded.shape[-2:], mode='nearest')

        decoded = decoded + context_agg

        # RGB prior from warped images weighted by time
        warped_stack = torch.stack(warped_imgs, dim=1)  # [B,N,3,H,W]
        weights_exp = t_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,N,1,1,1]
        warped_avg = (warped_stack * weights_exp).sum(dim=1)

        # base reconstruction
        synth_in = torch.cat([decoded, warped_avg], dim=1)
        synthesized = self.synthesis(synth_in)
        residual = self.residual_head(decoded) * self.residual_scale

        # Optional noise injection
        if not skip_noise_inject:
            noise_add, noise_gate = self.noise_inject(decoded, warped_avg)
        else:
            noise_add, noise_gate = torch.zeros_like(decoded), torch.zeros_like(decoded)

        # Optional content skip
        if not skip_content_skip:
            prior_mix, content_gate = self.content_skip(decoded, warped_avg)
        else:
            prior_mix, content_gate = torch.zeros_like(decoded), torch.zeros_like(decoded)

        # Final output
        out = synthesized + residual + noise_add
        out = (1.0 - content_gate) * out + prior_mix

        # spectral swap of the very high band from prior
        alpha = self.spectral_alpha.clamp(0.0, 1.0)
        if alpha.item() > 0:
            out = self._spectral_swap(out, warped_avg,
                                      lo=self.spectral_lo, hi=self.spectral_hi,
                                      alpha=float(alpha.item()),
                                      soft=self.spectral_soft)

        output = torch.clamp(out, 0, 1)

        aux = {
            'warped_imgs': warped_imgs,           # list of tensors
            'refined_masks': refined_masks,       # list of tensors
            'temporal_weights': t_weights.detach(),
            'warped_avg': warped_avg.detach(),
            'residual': residual.detach(),
            'synthesized': synthesized.detach(),
            'noise_add': noise_add.detach(),
            'noise_gate': noise_gate.detach(),
            'content_gate': content_gate.detach(),
            'residual_scale': float(self.residual_scale.item()),
            'detail_weight': float(self.detail_weight.item()),
            'spectral_alpha': float(alpha.item()),
            'spectral_band': (self.spectral_lo, self.spectral_hi)
        }
        return output, aux


# -------------------------
# Factory
# -------------------------

def build_fusion_net(num_anchors=3, base_channels=64, max_attention_size=96*96):
    return AnchorFusionNet(
        num_anchors=num_anchors,
        base_channels=base_channels,
        max_attention_size=max_attention_size
    )


# -------------------------
# Backward-compat aliases
# -------------------------

ImprovedHierarchicalCrossAttentionFusion = PyramidCrossAttention
EnhancedMultiAnchorFusionModel = AnchorFusionNet
TemporalWeightingModule = TemporalWeightingMLP
EnhancedResidualBlock = DetailAwareResBlock
create_enhanced_fusion_model = build_fusion_net


# -------------------------
# Smoke test
# -------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("Testing AnchorFusionNet")
    print("-" * 60)
    print("device:", device)

    model = build_fusion_net(num_anchors=3, base_channels=32).to(device)

    B, N, H, W = 1, 3, 2048, 2048
    I0_all = torch.randn(B, N, 3, H, W, device=device)
    I1_all = torch.randn(B, N, 3, H, W, device=device)
    flows_all = torch.randn(B, N, 4, H, W, device=device)
    masks_all = torch.rand(B, N, H, W, device=device)
    timesteps = torch.rand(B, N, device=device)

    with torch.no_grad():
        out, aux = model(I0_all, I1_all, flows_all, masks_all, timesteps)

    print(f"Output shape: {tuple(out.shape)}")
    print(f"Residual scale: {aux['residual_scale']:.3f}")
    print(f"Detail weight: {aux['detail_weight']:.3f}")
    print(f"Spectral alpha: {aux['spectral_alpha']:.3f}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
