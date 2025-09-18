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
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)

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
# Temporal Guidance Modules
# -------------------------

class TemporalPositionEncoding(nn.Module):
    """
    Encodes temporal position (timestep) information using sinusoidal embeddings
    and learnable projections to guide interpolation.
    """
    def __init__(self, channels, max_freq=10):
        super().__init__()
        self.channels = channels
        self.max_freq = max_freq
        
        # Create sinusoidal frequency bases
        freq_bands = 2.0 ** torch.linspace(0, max_freq - 1, max_freq)
        self.register_buffer('freq_bands', freq_bands)
        
        # Learnable projection to map encoded timesteps to channel dimension
        self.time_proj = nn.Sequential(
            nn.Linear(max_freq * 2, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, timesteps):
        """
        timesteps: [B, N] values in [0, 1] indicating position between frames
        Returns: [B, N, C] temporal encodings
        """
        B, N = timesteps.shape
        
        # Expand timesteps for frequency encoding
        t = timesteps.unsqueeze(-1)  # [B, N, 1]
        
        # Apply sinusoidal encoding
        freq_features = []
        for freq in self.freq_bands:
            freq_features.append(torch.sin(2 * math.pi * freq * t))
            freq_features.append(torch.cos(2 * math.pi * freq * t))
        
        encoded = torch.cat(freq_features, dim=-1)  # [B, N, max_freq*2]
        
        # Project to channel dimension
        return self.time_proj(encoded)  # [B, N, C]


class TemporalGuidanceModule(nn.Module):
    """
    Injects temporal guidance into features based on timestep position.
    Uses both additive and multiplicative modulation.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Temporal encoding
        self.time_encoder = TemporalPositionEncoding(channels)
        
        # Generate scale and shift parameters from temporal encoding
        self.scale_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()  # Scale between 0 and 1
        )
        
        self.shift_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Tanh()  # Shift between -1 and 1
        )
        
        # Learnable balance between scale and shift
        self.scale_weight = nn.Parameter(torch.tensor(1.0))
        self.shift_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, features, timesteps):
        """
        features: [B, C, H, W] or [B, N, C, H, W]
        timesteps: [B, N] temporal positions
        Returns: temporally modulated features
        """
        # Encode temporal positions
        time_emb = self.time_encoder(timesteps)  # [B, N, C]
        
        if features.dim() == 4:  # Single feature [B, C, H, W]
            # Average over anchors for single feature case
            time_emb = time_emb.mean(dim=1)  # [B, C]
            scale = self.scale_net(time_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            shift = self.shift_net(time_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            
            return features * (1 + scale * self.scale_weight) + shift * self.shift_weight
        
        else:  # Multiple features [B, N, C, H, W]
            B, N, C, H, W = features.shape
            modulated = []
            
            for i in range(N):
                feat = features[:, i]  # [B, C, H, W]
                t_emb = time_emb[:, i]  # [B, C]
                
                scale = self.scale_net(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                shift = self.shift_net(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
                
                mod_feat = feat * (1 + scale * self.scale_weight) + shift * self.shift_weight
                modulated.append(mod_feat)
            
            return torch.stack(modulated, dim=1)


class TemporalWeightingMLP(nn.Module):
    """
    Enhanced temporal weighting that considers timestep positions to predict
    interpolation weights rather than just processing them directly.
    """
    def __init__(self, num_anchors=3, hidden_dim=128, use_position_guidance=True):
        super().__init__()
        self.use_position_guidance = use_position_guidance
        
        if use_position_guidance:
            # Encode temporal positions first
            self.time_encoder = TemporalPositionEncoding(hidden_dim // 2)
            
            # Process encoded positions to predict weights
            self.net = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_anchors),
                nn.Softmax(dim=1)
            )
        else:
            # Original implementation
            self.net = nn.Sequential(
                nn.Linear(num_anchors, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_anchors),
                nn.Softmax(dim=1)
            )
    
    def forward(self, timesteps):
        if self.use_position_guidance:
            # Encode timesteps and process to get weights
            encoded = self.time_encoder(timesteps)  # [B, N, hidden_dim//2]
            encoded = encoded.mean(dim=1)  # [B, hidden_dim//2]
            return self.net(encoded)
        else:
            return self.net(timesteps)


class TemporalFlowModulation(nn.Module):
    """
    Modulates optical flow based on temporal position.
    Adjusts flow magnitude and direction based on where we are in time.
    """
    def __init__(self, channels=64):
        super().__init__()
        
        self.time_encoder = TemporalPositionEncoding(channels)
        
        # Predict flow adjustments
        self.flow_adjust = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 32),
            nn.ReLU(inplace=True)
        )
        
        # Conv for spatial flow adjustment
        self.flow_conv = nn.Sequential(
            nn.Conv2d(32, 4, 1),
            nn.Tanh()
        )
        
        # Learnable scale for adjustments
        self.adjust_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, flow, timesteps):
        """
        flow: [B, N, 4, H, W] optical flow
        timesteps: [B, N] temporal positions
        Returns: adjusted flow
        """
        B, N, _, H, W = flow.shape
        adjusted_flows = []
        
        for i in range(N):
            t = timesteps[:, i:i+1]  # [B, 1]
            f = flow[:, i]  # [B, 4, H, W]
            
            # Encode temporal position
            t_emb = self.time_encoder(t)  # [B, 1, C]
            t_emb = t_emb.squeeze(1)  # [B, C]
            
            # Predict flow adjustment
            adjustment = self.flow_adjust(t_emb)  # [B, 32]
            adjustment = adjustment.unsqueeze(-1).unsqueeze(-1).expand(B, 32, H, W)
            adjustment = self.flow_conv(adjustment)  # [B, 4, H, W]
            
            # Apply temporal-aware adjustment
            # Scale adjustment based on distance from endpoints (0 or 1)
            t_weight = 4 * t[:, 0:1, None, None] * (1 - t[:, 0:1, None, None])  # Peak at t=0.5
            adjusted_f = f + adjustment * self.adjust_scale * t_weight
            
            adjusted_flows.append(adjusted_f)
        
        return torch.stack(adjusted_flows, dim=1)


# -------------------------
# Cross-anchor fusion
# -------------------------

class PyramidCrossAttention(nn.Module):
    """
    Deformable Pyramid Cross Attention - drop-in replacement for the original.
    Uses deformable sampling instead of full attention for massive memory savings.
    """
    def __init__(self, channels, num_heads=4, max_attention_size=64*64,
                 num_points=4, num_levels=3, init_spatial_range=0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.max_attention_size = max_attention_size  # Kept for compatibility
        
        assert channels % num_heads == 0
        self.head_dim = channels // num_heads
        
        # Deformable attention components
        self.sampling_offsets = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GroupNorm(min(8, channels // 4), channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_heads * num_levels * num_points * 2, 1)
        )
        
        self.attention_weights = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.GroupNorm(min(8, channels // 4), channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, num_heads * num_levels * num_points, 1)
        )
        
        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.output_proj = nn.Conv2d(channels, channels, 1)
        
        # Frequency decomposition (matching original)
        self.freq_decomp = AdaptiveFrequencyDecoupling(channels)
        self.hf_scale = nn.Parameter(torch.tensor(0.3))
        
        # Level embeddings
        self.level_embed = nn.Parameter(torch.zeros(num_levels, channels))
        
        self._reset_parameters(init_spatial_range)
    
    def _reset_parameters(self, init_range):
        nn.init.constant_(self.sampling_offsets[-1].weight.data, 0.)
        
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]) * init_range
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1) / self.num_points
        
        for lvl in range(self.num_levels):
            grid_init[:, lvl, :, :] *= (2 ** lvl) * 0.1
            
        with torch.no_grad():
            self.sampling_offsets[-1].bias.copy_(grid_init.view(-1))
        
        nn.init.constant_(self.attention_weights[-1].weight.data, 0.)
        nn.init.constant_(self.attention_weights[-1].bias.data, 0.)
        nn.init.normal_(self.level_embed, 0.0, 0.02)
    
    def forward(self, query, keys, values, hf_res_scale=None):
        """
        Exact same interface as original PyramidCrossAttention.
        query: [B, C, H, W]
        keys: [B, N, C, H, W] 
        values: [B, N, C, H, W]
        hf_res_scale: optional high-freq scale
        """
        B, C, H, W = query.shape
        N = min(keys.shape[1], self.num_levels)  # Use available levels
        
        # Frequency decomposition
        q_low, q_high = self.freq_decomp(query)
        
        # Generate reference points
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=query.device),
            torch.linspace(0, 1, W, device=query.device),
            indexing='ij'
        )
        ref_points = torch.stack([ref_x, ref_y], dim=-1).view(1, H*W, 2)
        
        # Predict sampling offsets
        offsets = self.sampling_offsets(q_low)
        offsets = offsets.view(B, self.num_heads, self.num_levels, self.num_points, 2, H, W)
        offsets = offsets.permute(0, 5, 6, 1, 2, 3, 4).reshape(B, H*W, self.num_heads, self.num_levels, self.num_points, 2)
        
        # Normalize offsets
        offsets = offsets / torch.tensor([W, H], device=query.device).view(1, 1, 1, 1, 1, 2)
        
        # Sampling locations
        sampling_locs = ref_points.unsqueeze(2).unsqueeze(3).unsqueeze(4) + offsets
        sampling_locs = sampling_locs.clamp(0, 1)
        
        # Attention weights
        attn_weights = self.attention_weights(q_low)
        attn_weights = attn_weights.view(B, self.num_heads, self.num_levels, self.num_points, H, W)
        attn_weights = attn_weights.permute(0, 4, 5, 1, 2, 3).reshape(B, H*W, self.num_heads, self.num_levels * self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(B, H*W, self.num_heads, self.num_levels, self.num_points)
        
        # Sample and aggregate
        output = torch.zeros(B, C, H, W, device=query.device)
        
        for lvl in range(N):
            v_lvl = self.value_proj(values[:, lvl] + self.level_embed[lvl].view(1, -1, 1, 1))
            
            for head in range(self.num_heads):
                head_dim_start = head * self.head_dim
                head_dim_end = (head + 1) * self.head_dim
                v_head = v_lvl[:, head_dim_start:head_dim_end]
                
                for pt in range(self.num_points):
                    # Get sampling grid for this point
                    locs = sampling_locs[:, :, head, lvl, pt, :] * 2.0 - 1.0
                    locs = locs.view(B, H, W, 2)
                    
                    # Sample values
                    sampled = F.grid_sample(v_head, locs, mode='bilinear', align_corners=False)
                    
                    # Apply attention weight
                    weight = attn_weights[:, :, head, lvl, pt].view(B, 1, H, W)
                    output[:, head_dim_start:head_dim_end] += sampled * weight
        
        # Output projection
        output = self.output_proj(output)
        
        # Add high-frequency
        scale = hf_res_scale if hf_res_scale is not None else self.hf_scale
        output = output + q_high * scale
        
        return output


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
    """Backward warping with grid caching and improved normalization."""
    
    def __init__(self):
        super().__init__()
        self.backwarp_tenGrid = {}
    
    def forward(self, img, flow):
        """
        Warp img using flow.
        img: [B, C, H, W]
        flow: [B, 2, H, W] with flow[:, 0] = x-displacement, flow[:, 1] = y-displacement
        """
        device = flow.device
        B, _, H, W = flow.shape
        
        # Cache key based on device and size
        k = (str(device), str(flow.size()))
        
        # Create or retrieve cached grid
        if k not in self.backwarp_tenGrid:
            tenHorizontal = torch.linspace(-1.0, 1.0, W, device=device).view(
                1, 1, 1, W).expand(B, -1, H, -1)
            tenVertical = torch.linspace(-1.0, 1.0, H, device=device).view(
                1, 1, H, 1).expand(B, -1, -1, W)
            self.backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1)
        
        # Normalize flow based on image dimensions
        tenFlow = torch.cat([
            flow[:, 0:1, :, :] / ((img.shape[3] - 1.0) / 2.0),
            flow[:, 1:2, :, :] / ((img.shape[2] - 1.0) / 2.0)
        ], 1)
        
        # Add normalized flow to grid
        g = (self.backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
        
        # Use appropriate padding mode based on device
        padding_mode = 'zeros' if device.type == 'mps' else 'border'
        
        return F.grid_sample(
            input=img, 
            grid=g, 
            mode='bilinear', 
            padding_mode=padding_mode, 
            align_corners=False
        )


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
        super().__init__()
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
    Enhanced with temporal guidance for proper interpolation.
    """
    def __init__(self, num_anchors=3, base_channels=64, max_attention_size=96*96, use_temporal_guidance=True):
        super().__init__()
        self.num_anchors = num_anchors
        self.base_channels = base_channels
        self.use_temporal_guidance = use_temporal_guidance

        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.detail_weight = nn.Parameter(torch.tensor(0.3))

        # spectral swap controls
        self.spectral_alpha = nn.Parameter(torch.tensor(0.3))  # 0..1 blend amount
        self.spectral_lo = 0.32  # start of high-band in Nyquist units (0..0.5)
        self.spectral_hi = 0.50  # end of band
        self.spectral_soft = True

        self.flow_warp = FlowWarping()
        
        # Temporal modules - enhanced version if use_temporal_guidance is True
        self.temporal_weighter = TemporalWeightingMLP(num_anchors, use_position_guidance=use_temporal_guidance)
        self.temporal_temperature = nn.Parameter(torch.tensor(1.0))
        
        # Additional temporal guidance modules when enabled
        if use_temporal_guidance:
            self.temporal_guidance = TemporalGuidanceModule(base_channels * 2)
            self.temporal_flow_mod = TemporalFlowModulation(base_channels)
        else:
            self.temporal_guidance = None
            self.temporal_flow_mod = None

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
        timesteps: [B,N] temporal positions between 0 (at I0) and 1 (at I1)
        """
        B, N, _, H, W = I0_all.shape
        
        # Apply temporal flow modulation if enabled
        if self.use_temporal_guidance and self.temporal_flow_mod is not None:
            flows_all = self.temporal_flow_mod(flows_all, timesteps)

        # Get temporal weights (enhanced if using temporal guidance)
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

            feats_low.append(low)
            feats_mid.append(mid)
            feats_high.append(high)

            warped_imgs.append(warped)
            refined_masks.append(m)
            context_low.append(low)

        low_features = torch.stack(feats_low, dim=1)   # [B,N,C,H/2,W/2]
        mid_features = torch.stack(feats_mid, dim=1)   # [B,N,2C,H/4,W/4]
        high_features = torch.stack(feats_high, dim=1) # [B,N,4C,H/8,W/8]
        
        # Apply temporal guidance to features if enabled
        if self.use_temporal_guidance and self.temporal_guidance is not None:
            low_features = self.temporal_guidance(low_features, timesteps)
        
        # Apply temporal weights to features
        w_exp = t_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        low_features_weighted = low_features * w_exp
        mid_features_weighted = mid_features * w_exp
        high_features_weighted = high_features * w_exp

        # cross-anchor attention (using weighted features)
        query_high = high_features_weighted.sum(dim=1) if self.use_temporal_guidance else high_features_weighted.mean(dim=1)
        fused_high = self.cross_high(query_high, high_features_weighted, high_features_weighted, hf_res_scale=self.detail_weight)

        up_high = self.decoder['up_high_to_mid'][0](fused_high, target_size=mid_features.shape[-2:])
        up_high = self.decoder['up_high_to_mid'][1](up_high)

        query_mid = mid_features_weighted.sum(dim=1) if self.use_temporal_guidance else mid_features_weighted.mean(dim=1)
        fused_mid = self.cross_mid(query_mid, mid_features_weighted, mid_features_weighted, hf_res_scale=self.detail_weight)
        fused_mid = nn.Sequential(*self.decoder['fuse_mid'])(torch.cat([up_high, fused_mid], dim=1))

        up_mid = self.decoder['up_mid_to_low'][0](fused_mid, target_size=low_features.shape[-2:])
        up_mid = self.decoder['up_mid_to_low'][1](up_mid)

        query_low = low_features_weighted.sum(dim=1) if self.use_temporal_guidance else low_features_weighted.mean(dim=1)
        fused_low = self.cross_low(query_low, low_features_weighted, low_features_weighted, hf_res_scale=self.detail_weight)
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
            noise_add = torch.zeros_like(synthesized)
            noise_gate = torch.zeros_like(synthesized[:, :1])

        # Optional content skip
        if not skip_content_skip:
            prior_mix, content_gate = self.content_skip(decoded, warped_avg)
        else:
            prior_mix = torch.zeros_like(synthesized)
            content_gate = torch.zeros_like(synthesized[:, :1])

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

def build_fusion_net(num_anchors=3, base_channels=64, max_attention_size=96*96, use_temporal_guidance=True):
    return AnchorFusionNet(
        num_anchors=num_anchors,
        base_channels=base_channels,
        max_attention_size=max_attention_size,
        use_temporal_guidance=use_temporal_guidance
    )


# -------------------------
# Backward-compat aliases
# -------------------------

ImprovedHierarchicalCrossAttentionFusion = PyramidCrossAttention
EnhancedMultiAnchorFusionModel = AnchorFusionNet
TemporalWeightingModule = TemporalWeightingMLP
EnhancedResidualBlock = DetailAwareResBlock
create_fusion_model = build_fusion_net


# -------------------------
# Smoke test and gradient check
# -------------------------

def check_gradients(model, verbose=True):
    """
    Check gradient flow through the model.
    Returns dict with gradient statistics for each parameter.
    """
    grad_stats = {}
    zero_grad_params = []
    nan_grad_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_mean = param.grad.data.mean().item()
            grad_std = param.grad.data.std().item()
            
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'shape': tuple(param.grad.shape)
            }
            
            if grad_norm == 0:
                zero_grad_params.append(name)
            if torch.isnan(param.grad).any():
                nan_grad_params.append(name)
    
    if verbose:
        print("\n" + "="*60)
        print("GRADIENT CHECK RESULTS")
        print("="*60)
        
        if zero_grad_params:
            print(f"\n⚠️  WARNING: {len(zero_grad_params)} parameters have zero gradients:")
            for p in zero_grad_params[:5]:  # Show first 5
                print(f"  - {p}")
            if len(zero_grad_params) > 5:
                print(f"  ... and {len(zero_grad_params) - 5} more")
        
        if nan_grad_params:
            print(f"\n❌ ERROR: {len(nan_grad_params)} parameters have NaN gradients:")
            for p in nan_grad_params[:5]:
                print(f"  - {p}")
        
        if not zero_grad_params and not nan_grad_params:
            print("\n✅ All gradients are healthy!")
        
        # Show gradient statistics for key layers
        print("\nGradient Statistics (key layers):")
        key_patterns = ['encoder', 'decoder', 'cross_', 'temporal_', 'synthesis', 'residual_head']
        for pattern in key_patterns:
            matching = [k for k in grad_stats.keys() if pattern in k]
            if matching:
                # Show first match for each pattern
                k = matching[0]
                s = grad_stats[k]
                print(f"  {k[:50]:50s} -> norm: {s['norm']:.6f}, mean: {s['mean']:.2e}")
    
    return grad_stats, zero_grad_params, nan_grad_params


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print("Testing AnchorFusionNet")
    print("-" * 60)
    print("device:", device)

    # Test with temporal guidance enabled
    model = build_fusion_net(num_anchors=3, base_channels=64, use_temporal_guidance=True).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Prepare test data
    B, N, H, W = 1, 3, 512, 512
    I0_all = torch.randn(B, N, 3, H, W, device=device, requires_grad=True)
    I1_all = torch.randn(B, N, 3, H, W, device=device, requires_grad=True)
    flows_all = torch.randn(B, N, 4, H, W, device=device, requires_grad=True)
    masks_all = torch.rand(B, N, H, W, device=device, requires_grad=True)
    timesteps = torch.rand(B, N, device=device, requires_grad=True)
    
    # Create target for loss computation
    target = torch.rand(B, 3, H, W, device=device)

    print("\n" + "="*60)
    print("FORWARD PASS TEST")
    print("="*60)
    
    # Test forward pass
    try:
        out, aux = model(I0_all, I1_all, flows_all, masks_all, timesteps)
        print(f"Forward pass successful!")
        print(f"Output shape: {tuple(out.shape)}")
        print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
        print(f"Temporal weights: {aux['temporal_weights'][0].detach().cpu().tolist()}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise

    print("\n" + "="*60)
    print("BACKWARD PASS TEST")
    print("="*60)
    
    # Test backward pass with different loss functions
    loss_functions = [
        ("L1", nn.L1Loss()),
        ("L2", nn.MSELoss()),
        ("Perceptual (simulated)", lambda x, y: ((x - y) ** 2).mean() + 0.1 * torch.abs(x - y).mean())
    ]
    
    for loss_name, loss_fn in loss_functions:
        print(f"\nTesting with {loss_name} loss:")
        
        # Reset gradients
        model.zero_grad()
        for param in [I0_all, I1_all, flows_all, masks_all, timesteps]:
            if param.grad is not None:
                param.grad.zero_()
        
        # Forward pass
        out, aux = model(I0_all, I1_all, flows_all, masks_all, timesteps)
        
        # Compute loss
        if callable(loss_fn):
            loss = loss_fn(out, target)
        else:
            loss = loss_fn(out, target)
        
        print(f"  Loss value: {loss.item():.6f}")
        
        # Backward pass
        try:
            loss.backward()
            print(f"Backward pass successful!")
            
            # Check gradients
            grad_stats, zero_grads, nan_grads = check_gradients(model, verbose=(loss_name == "L1"))
            
            # Check input gradients
            print(f"\n  Input gradients:")
            for name, tensor in [("I0", I0_all), ("I1", I1_all), ("flows", flows_all), 
                                ("masks", masks_all), ("timesteps", timesteps)]:
                if tensor.grad is not None:
                    grad_norm = tensor.grad.norm().item()
                    print(f"    {name:10s} grad norm: {grad_norm:.6f}")
                else:
                    print(f"    {name:10s} grad: None")
            
        except Exception as e:
            print(f"Backward pass failed: {e}")
            raise
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)