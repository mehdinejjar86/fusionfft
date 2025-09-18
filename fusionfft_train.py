import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime
import warnings
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from torch.amp import autocast, GradScaler

# Weights & Biases support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Dataset
from fusion_dataset import create_multi_dataloader

# New model file
# If you named the new model file differently, update the import below.
from fusionfft_model import build_fusion_net as create_fusion_model


# ----------------------------
# Losses
# ----------------------------

class FusionLoss(nn.Module):
    """
    Composite loss:
      - L1 on RGB
      - Frequency loss on log-magnitude spectra with high-frequency weighting
      - Edge loss via Sobel gradients
      - Perceptual loss on VGG features
      - Consistency to the temporally weighted warped prior (aux['warped_avg'])
    """
    def __init__(self,
                 lambda_l1=0.8,
                 lambda_freq=0.3,
                 lambda_edge=0.2,
                 lambda_perceptual=0.15,
                 lambda_consistency=0.1,
                 device=None):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = float(lambda_l1)
        self.lambda_freq = float(lambda_freq)
        self.lambda_edge = float(lambda_edge)
        self.lambda_perceptual = float(lambda_perceptual)
        self.lambda_consistency = float(lambda_consistency)

        self.device = device if device is not None else torch.device("cpu")

        # VGG for perceptual
        self.vgg = None
        self.register_buffer("imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        try:
            from torchvision.models import vgg16, VGG16_Weights
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features[:16].to(self.device).eval()
            for p in self.vgg.parameters():
                p.requires_grad = False
        except Exception:
            self.vgg = None  # Perceptual term will be 0 if unavailable

        # Sobel kernels for edge loss
        sx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sy.view(1, 1, 3, 3))

    def forward(self, pred, target, aux=None):
        losses = {}

        # L1
        l1_val = self.l1(pred, target)
        losses["l1"] = l1_val

        # Frequency loss
        freq_val = self._frequency_loss(pred, target)
        losses["frequency"] = freq_val

        # Edge loss
        edge_val = self._edge_loss(pred, target)
        losses["edge"] = edge_val

        # Perceptual loss
        perc_val = self._perceptual_loss(pred, target)
        losses["perceptual"] = perc_val

        # Consistency to warped prior if available
        cons_val = torch.tensor(0.0, device=pred.device)
        if aux is not None and isinstance(aux, dict) and ("warped_avg" in aux):
            warped_avg = aux["warped_avg"]
            if warped_avg is not None and isinstance(warped_avg, torch.Tensor):
                if warped_avg.shape != pred.shape:
                    warped_avg = torch.nn.functional.interpolate(
                        warped_avg, size=pred.shape[-2:], mode="bilinear", align_corners=False
                    )
                cons_val = self.l1(pred, warped_avg)
        losses["consistency"] = cons_val

        total = (self.lambda_l1 * l1_val
                 + self.lambda_freq * freq_val
                 + self.lambda_edge * edge_val
                 + self.lambda_perceptual * perc_val
                 + self.lambda_consistency * cons_val)
        losses["total"] = total
        return losses

    def _frequency_loss(self, x, y):
        """
        Compare log-magnitude spectra with radial high-frequency weighting.
        x, y: [B,3,H,W] in [0,1]
        """
        B, C, H, W = x.shape
        # rFFT along H,W
        X = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        Y = torch.fft.rfft2(y, dim=(-2, -1), norm="ortho")

        # Radial weights on the rFFT grid
        fy = torch.fft.fftfreq(H, d=1.0).to(x.device)  # [-0.5..0.5)
        fx = torch.fft.rfftfreq(W, d=1.0).to(x.device)  # [0..0.5]
        wy, wx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(wx**2 + wy**2)
        r_max = r.max().clamp(min=1e-6)
        w = (r / r_max) ** 1.5  # emphasize high frequencies, smooth

        # Log magnitude
        eps = 1e-6
        Xmag = torch.log1p(torch.abs(X) + eps)
        Ymag = torch.log1p(torch.abs(Y) + eps)

        diff = torch.abs(Xmag - Ymag)  # [B,C,H,W_rfft]
        # Broadcast weights to [B,C,H,W_rfft]
        w = w.view(1, 1, H, W // 2 + 1)
        return (diff * w).mean()

    def _edge_loss(self, x, y):
        """
        Sobel gradient magnitude L1.
        """
        B, C, H, W = x.shape
        # group-wise conv with shared kernels per channel
        weight_x = self.sobel_x.repeat(C, 1, 1, 1)
        weight_y = self.sobel_y.repeat(C, 1, 1, 1)

        gx_x = torch.nn.functional.conv2d(x, weight_x, padding=1, groups=C)
        gy_x = torch.nn.functional.conv2d(x, weight_y, padding=1, groups=C)
        gm_x = torch.sqrt(gx_x**2 + gy_x**2 + 1e-6)

        gx_y = torch.nn.functional.conv2d(y, weight_x, padding=1, groups=C)
        gy_y = torch.nn.functional.conv2d(y, weight_y, padding=1, groups=C)
        gm_y = torch.sqrt(gx_y**2 + gy_y**2 + 1e-6)

        return torch.mean(torch.abs(gm_x - gm_y))

    def _perceptual_loss(self, x, y):
        if self.vgg is None:
            return torch.tensor(0.0, device=x.device)
        # ImageNet normalization
        x_n = (x - self.imnet_mean) / self.imnet_std
        y_n = (y - self.imnet_mean) / self.imnet_std
        fx = self.vgg(x_n)
        fy = self.vgg(y_n)
        return torch.mean(torch.abs(fx - fy))


# ----------------------------
# Parallel wrapper
# ----------------------------

class ModelParallelWrapper(nn.Module):
    """
    Spatial and batch parallel for AnchorFusionNet.
    Anchor parallel is intentionally disabled since the model has per-anchor modules.
    """
    def __init__(self, base_model, max_attention_size, num_gpus=None, split_strategy='spatial'):
        super().__init__()
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.split_strategy = split_strategy
        self.max_attention_size = max_attention_size
        self.num_anchors = base_model.num_anchors
        self.base_channels = base_model.base_channels

        if self.num_gpus <= 1:
            self.model = base_model
            self.parallel_mode = False
        else:
            self.models = nn.ModuleList()
            for i in range(self.num_gpus):
                model_copy = create_fusion_model(
                    num_anchors=self.num_anchors,
                    base_channels=self.base_channels,
                    max_attention_size=max_attention_size
                )
                self.models.append(model_copy.cuda(i))
            self.parallel_mode = True
            # sync initial params
            source_state = self.models[0].state_dict()
            for i in range(1, self.num_gpus):
                self.models[i].load_state_dict(source_state)

    def sync_gradients(self):
        if not self.parallel_mode or self.num_gpus <= 1:
            return
        # average grads onto model[0]
        params_ref = list(self.models[0].parameters())
        for p_idx, p_main in enumerate(params_ref):
            if p_main.grad is None:
                continue
            grads = [p_main.grad.data]
            for i in range(1, self.num_gpus):
                p_copy = list(self.models[i].parameters())[p_idx]
                if p_copy.grad is not None:
                    grads.append(p_copy.grad.data.cuda(0))
            if len(grads) > 1:
                p_main.grad.data = torch.stack(grads, dim=0).mean(dim=0)

    def sync_parameters(self):
        if not self.parallel_mode or self.num_gpus <= 1:
            return
        source_state = self.models[0].state_dict()
        for i in range(1, self.num_gpus):
            self.models[i].load_state_dict(source_state)

    def forward(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        if not self.parallel_mode:
            return self.model(I0_all, I1_all, flows_all, masks_all, timesteps)

        if self.split_strategy == 'spatial':
            return self._forward_spatial_parallel(I0_all, I1_all, flows_all, masks_all, timesteps)
        elif self.split_strategy == 'batch':
            return self._forward_batch_parallel(I0_all, I1_all, flows_all, masks_all, timesteps)
        else:
            print("Anchor splitting is not supported. Falling back to spatial parallel.")
            return self._forward_spatial_parallel(I0_all, I1_all, flows_all, masks_all, timesteps)

    def _forward_spatial_parallel(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        B, A, C, H, W = I0_all.shape
        overlap = 32
        base_strip_height = H // self.num_gpus

        outputs = []
        aux_list = []
        strip_infos = []

        for gpu_id in range(self.num_gpus):
            start_h = max(0, gpu_id * base_strip_height - overlap // 2)
            end_h = min(H, (gpu_id + 1) * base_strip_height + overlap // 2)
            strip_infos.append({'start_h': start_h, 'end_h': end_h, 'gpu_id': gpu_id})

            with torch.cuda.device(gpu_id):
                I0_strip = I0_all[:, :, :, start_h:end_h, :].cuda(gpu_id, non_blocking=True)
                I1_strip = I1_all[:, :, :, start_h:end_h, :].cuda(gpu_id, non_blocking=True)
                flows_strip = flows_all[:, :, :, start_h:end_h, :].cuda(gpu_id, non_blocking=True)
                masks_strip = masks_all[:, :, start_h:end_h, :].cuda(gpu_id, non_blocking=True)
                t_gpu = timesteps.cuda(gpu_id, non_blocking=True)

                out_s, aux_s = self.models[gpu_id](I0_strip, I1_strip, flows_strip, masks_strip, t_gpu)

            outputs.append(out_s)
            aux_list.append(aux_s)

        merged_output = self._merge_spatial_outputs(outputs, H, overlap, base_strip_height)
        merged_aux = self._merge_aux_outputs_spatial(aux_list, strip_infos, H, W, overlap, base_strip_height)

        return merged_output, merged_aux

    def _merge_spatial_outputs(self, outputs, full_height, overlap, base_strip_height):
        device = outputs[0].device
        B, C, _, W = outputs[0].shape
        merged = torch.zeros(B, C, full_height, W, device=device)
        weights = torch.zeros(B, 1, full_height, W, device=device)

        for gpu_id, output in enumerate(outputs):
            start_h = max(0, gpu_id * base_strip_height - overlap // 2)
            end_h = min(full_height, (gpu_id + 1) * base_strip_height + overlap // 2)
            strip_h = end_h - start_h

            weight = torch.ones(B, 1, strip_h, W, device=output.device)
            if overlap > 0:
                if gpu_id > 0 and overlap // 2 < strip_h:
                    fade_size = min(overlap // 2, strip_h)
                    fade = torch.linspace(0, 1, fade_size, device=output.device)
                    weight[:, :, :fade_size, :] *= fade.view(1, 1, -1, 1)
                if gpu_id < self.num_gpus - 1 and overlap // 2 < strip_h:
                    fade_size = min(overlap // 2, strip_h)
                    fade = torch.linspace(1, 0, fade_size, device=output.device)
                    weight[:, :, -fade_size:, :] *= fade.view(1, 1, -1, 1)

            merged[:, :, start_h:end_h, :] = merged[:, :, start_h:end_h, :] + output.to(merged.device) * weight.to(merged.device)
            weights[:, :, start_h:end_h, :] = weights[:, :, start_h:end_h, :] + weight.to(weights.device)

        merged = merged / (weights + 1e-8)
        return merged

    def _merge_aux_outputs_spatial(self, aux_list, strip_infos, H, W, overlap, base_strip_height):
        # Only merge what is needed for the loss: warped_avg
        device = torch.device("cuda:0")
        B = aux_list[0]['warped_avg'].shape[0]
        merged = torch.zeros(B, 3, H, W, device=device)
        weights = torch.zeros(B, 1, H, W, device=device)

        for gpu_id, aux in enumerate(aux_list):
            wa = aux.get('warped_avg', None)
            if wa is None:
                continue
            wa = wa.to(device)

            start_h = max(0, gpu_id * base_strip_height - overlap // 2)
            end_h = min(H, (gpu_id + 1) * base_strip_height + overlap // 2)
            strip_h = end_h - start_h

            weight = torch.ones(B, 1, strip_h, W, device=device)
            if overlap > 0:
                if gpu_id > 0 and overlap // 2 < strip_h:
                    fade_size = min(overlap // 2, strip_h)
                    fade = torch.linspace(0, 1, fade_size, device=device)
                    weight[:, :, :fade_size, :] *= fade.view(1, 1, -1, 1)
                if gpu_id < len(aux_list) - 1 and overlap // 2 < strip_h:
                    fade_size = min(overlap // 2, strip_h)
                    fade = torch.linspace(1, 0, fade_size, device=device)
                    weight[:, :, -fade_size:, :] *= fade.view(1, 1, -1, 1)

            merged[:, :, start_h:end_h, :] += wa * weight
            weights[:, :, start_h:end_h, :] += weight

        warped_avg_full = merged / (weights + 1e-8)
        # carry over a few scalars from GPU 0
        base_aux = aux_list[0]
        return {
            'warped_avg': warped_avg_full,
            'residual_scale': base_aux.get('residual_scale', 0.0),
            'detail_weight': base_aux.get('detail_weight', 0.0),
            'temporal_weights': base_aux.get('temporal_weights', None)
        }

    def _forward_batch_parallel(self, I0_all, I1_all, flows_all, masks_all, timesteps):
        B = I0_all.shape[0]
        splits = torch.tensor_split(torch.arange(B), self.num_gpus)
        outputs = []
        aux_parts = []

        for gpu_id, idx in enumerate(splits):
            if idx.numel() == 0:
                continue
            with torch.cuda.device(gpu_id):
                sel = idx.to(torch.long)
                out, aux = self.models[gpu_id](
                    I0_all[sel].cuda(gpu_id, non_blocking=True),
                    I1_all[sel].cuda(gpu_id, non_blocking=True),
                    flows_all[sel].cuda(gpu_id, non_blocking=True),
                    masks_all[sel].cuda(gpu_id, non_blocking=True),
                    timesteps[sel].cuda(gpu_id, non_blocking=True),
                )
                outputs.append((sel, out))
                aux_parts.append((sel, aux))

        # stitch by original order
        device0 = torch.device("cuda:0")
        merged_out = torch.zeros_like(outputs[0][1]).repeat(B, 1, 1, 1)  # placeholder resized soon
        merged_out = None
        out_list = [None] * B
        warped_list = [None] * B

        for sel, out in outputs:
            out = out.to(device0)
            for j, i_orig in enumerate(sel.tolist()):
                out_list[i_orig] = out[j:j+1]

        merged_out = torch.cat(out_list, dim=0)

        # Merge warped_avg for loss
        for sel, aux in aux_parts:
            if 'warped_avg' in aux and isinstance(aux['warped_avg'], torch.Tensor):
                wa = aux['warped_avg'].to(device0)
                for j, i_orig in enumerate(sel.tolist()):
                    warped_list[i_orig] = wa[j:j+1]
        warped_avg = torch.cat(warped_list, dim=0) if all(t is not None for t in warped_list) else None

        base_aux = aux_parts[0][1]
        merged_aux = {
            'warped_avg': warped_avg,
            'residual_scale': base_aux.get('residual_scale', 0.0),
            'detail_weight': base_aux.get('detail_weight', 0.0),
            'temporal_weights': base_aux.get('temporal_weights', None)
        }
        return merged_out, merged_aux


# ----------------------------
# Metrics
# ----------------------------

class MetricsCalculator:
    def __init__(self, eval_size=(256, 256)):
        self.eval_size = eval_size

    def calculate_psnr(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        if pred.ndim == 4:
            pred = pred.transpose(0, 2, 3, 1)
            target = target.transpose(0, 2, 3, 1)
        elif pred.ndim == 3 and pred.shape[0] == 3:
            pred = pred.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)

        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)

        if pred.ndim == 4:
            vals = []
            for i in range(pred.shape[0]):
                vals.append(psnr_func(target[i], pred[i], data_range=1.0))
            return np.mean(vals)
        else:
            return psnr_func(target, pred, data_range=1.0)

    def calculate_ssim(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        if pred.ndim == 4:
            pred = pred.transpose(0, 2, 3, 1)
            target = target.transpose(0, 2, 3, 1)
        elif pred.ndim == 3 and pred.shape[0] == 3:
            pred = pred.transpose(1, 2, 0)
            target = target.transpose(1, 2, 0)

        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)

        if pred.ndim == 4:
            vals = []
            for i in range(pred.shape[0]):
                vals.append(ssim_func(target[i], pred[i], data_range=1.0, channel_axis=2 if pred.shape[-1] == 3 else None))
            return np.mean(vals)
        else:
            return ssim_func(target, pred, data_range=1.0, channel_axis=2 if pred.shape[-1] == 3 else None)

    def resize_for_metrics(self, tensor, size=None):
        size = size or self.eval_size
        if tensor.shape[-2:] == size:
            return tensor
        out = torch.nn.functional.interpolate(
            tensor if tensor.dim() == 4 else tensor.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False
        )
        return out if tensor.dim() == 4 else out.squeeze(0)


# ----------------------------
# Trainer
# ----------------------------

class AnchorFusionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self._setup_mixed_precision()

        # Run dirs
        self.run_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.run_name = f"{config.run_name}_{self.run_timestamp}" if getattr(config, 'run_name', None) else self.run_timestamp

        self.config.checkpoint_dir = Path("runs") / self.run_name / Path(self.config.checkpoint_dir)
        self.config.log_dir = Path("runs") / self.run_name / Path(self.config.log_dir)
        self.config.sample_dir = Path("runs") / self.run_name / Path(self.config.sample_dir)
        self.setup_directories()

        # W&B
        self.use_wandb = WANDB_AVAILABLE and getattr(config, 'use_wandb', False)
        if self.use_wandb:
            self._init_wandb()

        # Data
        self.train_loader, self.train_dataset = self._create_dataloader()
        self.val_loader, self.val_dataset = self._create_validation_dataloader()

        # Model
        self.model = self._create_parallel_model()

        # Loss
        self.loss_fn = FusionLoss(
            lambda_l1=config.lambda_l1,
            lambda_freq=config.lambda_freq,
            lambda_edge=config.lambda_edge,
            lambda_perceptual=config.lambda_perceptual,
            lambda_consistency=config.lambda_consistency,
            device=self.device
        ).to(self.device)

        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Metrics
        metric_size = tuple(getattr(config, 'metric_size', (256, 256)))
        self.metrics_calc = MetricsCalculator(eval_size=metric_size)

        # Logging
        self.writer = SummaryWriter(self.config.log_dir)
        self.start_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        self.best_val_psnr = 0
        self.best_val_ssim = 0
        self.epoch_metrics = []

        # Resume
        if config.resume:
            self.load_checkpoint(config.resume)

        print("\n" + "="*60)
        print("Anchor Fusion Training Configuration")
        print("="*60 + "\n")

    def _init_wandb(self):
        wandb_config = vars(self.config).copy()
        wandb_config['run_timestamp'] = self.run_timestamp
        wandb_config['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        wandb_config['model_type'] = 'AnchorFusionNet'
        wandb.init(
            project=getattr(self.config, 'wandb_project', 'anchor-fusion-model'),
            entity=getattr(self.config, 'wandb_entity', None),
            name=self.run_name,
            config=wandb_config,
            resume=self.config.resume is not None
        )
        if getattr(self.config, 'wandb_watch_model', False):
            if hasattr(self.model, 'models'):
                wandb.watch(self.model.models[0], log='all', log_freq=100)
            else:
                wandb.watch(self.model, log='all', log_freq=100)
        print(f"W&B initialized: {wandb.run.url}")

    def _setup_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            ng = torch.cuda.device_count()
            print(f"Using {ng} CUDA GPU(s)")
            for i in range(ng):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training based on config."""
        self.precision = getattr(self.config, 'precision', 'fp32')
        self.use_amp = False
        self.scaler = None
        self.amp_dtype = None
        
        # Always set autocast_device based on device type (needed for autocast API)
        if self.device.type in ['cuda', 'cpu', 'mps']:
            self.autocast_device = self.device.type
        else:
            self.autocast_device = 'cpu'  # fallback
        
        if self.precision != 'fp32':
            if self.device.type == 'cuda':
                # CUDA supports both fp16 and bf16
                if self.precision == 'bf16':
                    if torch.cuda.is_bf16_supported():
                        self.use_amp = True
                        self.amp_dtype = torch.bfloat16
                    else:
                        print(f"Warning: BF16 not supported on this GPU. Falling back to FP32.")
                        self.precision = 'fp32'
                elif self.precision == 'fp16':
                    self.use_amp = True
                    self.amp_dtype = torch.float16
                    # Only fp16 needs GradScaler
                    self.scaler = GradScaler('cuda')
                        
            elif self.device.type == 'cpu':
                # CPU only supports bf16
                if self.precision == 'bf16':
                    self.use_amp = True
                    self.amp_dtype = torch.bfloat16
                else:
                    print(f"Warning: CPU only supports BF16 mixed precision. Falling back to FP32.")
                    self.precision = 'fp32'
                        
            elif self.device.type == 'mps':
                # MPS doesn't support autocast yet
                print(f"Warning: MPS doesn't support mixed precision training yet. Using FP32.")
                self.precision = 'fp32'
                # Note: You could still manually cast model to fp16 on MPS, but without autocast
                # it's more complex and may not provide benefits
            else:
                print(f"Warning: Device {self.device.type} doesn't support mixed precision. Using FP32.")
                self.precision = 'fp32'
        
        # Set default dtype for fp32 mode
        if not self.use_amp:
            self.amp_dtype = torch.float32
        
        print(f"Training precision: {self.precision.upper()}")
        if self.use_amp:
            print(f"  Device: {self.autocast_device}")
            print(f"  AMP dtype: {self.amp_dtype}")
            print(f"  GradScaler: {'enabled' if self.scaler else 'not needed'}")
        else:
            print(f"  Device: {self.autocast_device} (autocast disabled)")

    def setup_directories(self):
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.log_dir = Path(self.config.log_dir)
        self.sample_dir = Path(self.config.sample_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRun directory: {self.run_name}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Samples: {self.sample_dir}")

        config_dict = vars(self.config)
        config_dict['run_timestamp'] = self.run_timestamp
        config_dict['run_name'] = self.run_name
        config_dict['num_gpus'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        config_dict['model_type'] = 'AnchorFusionNet'

        with open(self.checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)

    def _create_dataloader(self):
        gt_paths = self.config.gt_paths
        steps = self.config.steps
        if isinstance(steps, int):
            steps = [steps] * len(gt_paths)
        elif len(steps) == 1 and len(gt_paths) > 1:
            steps = steps * len(gt_paths)

        mix_strategy = getattr(self.config, 'mix_strategy', 'uniform')
        path_weights = getattr(self.config, 'path_weights', None)
        cache_flows = getattr(self.config, 'cache_flows', False)

        num_workers = self.config.num_workers
        if self.device.type == 'mps' and num_workers > 0:
            print("Warning: MPS device detected. Setting num_workers=0")
            num_workers = 0

        train_dataloader, train_dataset = create_multi_dataloader(
            gt_paths=gt_paths,
            steps=steps,
            anchor=self.config.num_anchors,
            scale=self.config.scale,
            UHD=self.config.UHD,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            model_dir=self.config.rife_model_dir,
            mix_strategy=mix_strategy,
            path_weights=path_weights,
            cache_flows=cache_flows
        )

        print("\nTraining Dataset Configuration:")
        print(f"Total samples: {len(train_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Batches per epoch: {len(train_dataloader)}")
        return train_dataloader, train_dataset

    def _create_validation_dataloader(self):
        val_paths = getattr(self.config, 'val_paths', None)
        val_steps = getattr(self.config, 'val_steps', None)
        val_split = getattr(self.config, 'val_split', 0.1)

        if val_paths is not None:
            if isinstance(val_steps, int):
                val_steps = [val_steps] * len(val_paths)
            elif val_steps is None:
                val_steps = self.config.steps
                if isinstance(val_steps, int):
                    val_steps = [val_steps] * len(val_paths)
            elif len(val_steps) == 1 and len(val_paths) > 1:
                val_steps = val_steps * len(val_paths)

            print("\nValidation Dataset (Separate Paths):")
            print(f"Validation paths: {val_paths}")
            print(f"Validation steps: {val_steps}")

            num_workers = self.config.num_workers
            if self.device.type == 'mps' and num_workers > 0:
                num_workers = 0

            val_dataloader, val_dataset = create_multi_dataloader(
                gt_paths=val_paths,
                steps=val_steps,
                anchor=self.config.num_anchors,
                scale=self.config.scale,
                UHD=self.config.UHD,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                model_dir=self.config.rife_model_dir,
                mix_strategy='uniform',
                path_weights=None,
                cache_flows=getattr(self.config, 'cache_flows', False)
            )

            print(f"Validation samples: {len(val_dataset)}")
            print(f"Validation batches: {len(val_dataloader)}")
        elif val_split > 0:
            from torch.utils.data import random_split
            total = len(self.train_dataset)
            val_size = int(total * val_split)
            train_size = total - val_size
            train_subset, val_subset = random_split(
                self.train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

            num_workers = self.config.num_workers
            if self.device.type == 'mps' and num_workers > 0:
                num_workers = 0

            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            val_dataloader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            val_dataset = val_subset

            print(f"Training samples after split: {len(train_subset)}")
            print(f"Validation samples: {len(val_subset)}")
            print(f"Validation batches: {len(val_dataloader)}")
        else:
            print("\nNo validation dataset configured (val_split=0)")
            return None, None

        return val_dataloader, val_dataset

    def _create_parallel_model(self):
        max_attention_size = getattr(self.config, 'max_attention_size', 96*96)
        base_model = create_fusion_model(
            num_anchors=self.config.num_anchors,
            base_channels=self.config.base_channels,
            max_attention_size=max_attention_size
        )

        size = int(np.sqrt(max_attention_size))
        print(f"\nModel Configuration:")
        print(f"  Pyramid Attention window: {size}x{size} = {max_attention_size:,} elements")
        print(f"  Detail preservation: enabled")

        use_parallel = getattr(self.config, 'model_parallel', False)
        parallel_strategy = getattr(self.config, 'parallel_strategy', 'spatial')

        if use_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if parallel_strategy not in ('spatial', 'batch'):
                print("Parallel strategy set to unsupported value. Using spatial.")
                parallel_strategy = 'spatial'
            print(f"Model parallel: {parallel_strategy}")
            model = ModelParallelWrapper(
                base_model,
                max_attention_size=max_attention_size,
                num_gpus=torch.cuda.device_count(),
                split_strategy=parallel_strategy
            )
        else:
            model = base_model.to(self.device)
            print(f"Using model on {self.device}")

        total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        return model

    def _create_optimizer(self):
        if hasattr(self.model, 'models'):
            params = self.model.models[0].parameters()
        else:
            params = self.model.parameters()

        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(params, lr=self.config.learning_rate, betas=(0.9, 0.999),
                                   weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(params, lr=self.config.learning_rate, betas=(0.9, 0.999),
                                    weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        return optimizer

    def _create_scheduler(self):
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=self.config.min_lr)
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        else:
            return None

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = {'total': 0, 'l1': 0, 'frequency': 0, 'edge': 0, 'perceptual': 0, 'consistency': 0}
        epoch_psnr = 0
        epoch_ssim = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{self.config.epochs}")
        for batch_idx, batch in enumerate(pbar):
            I0_all = batch['I0'].to(self.device, non_blocking=True)
            I1_all = batch['I1'].to(self.device, non_blocking=True)
            flows_all = batch['flows'].to(self.device, non_blocking=True)
            masks_all = batch['masks'].to(self.device, non_blocking=True)
            timesteps = batch['timesteps'].to(self.device, non_blocking=True)
            I_gt = batch['I_gt'].to(self.device, non_blocking=True)

            # Forward pass with optional mixed precision
            with autocast(device_type=self.autocast_device, dtype=self.amp_dtype, enabled=self.use_amp):
                output, aux = self.model(I0_all, I1_all, flows_all, masks_all, timesteps)
                if output.device != I_gt.device:
                    output = output.to(I_gt.device)
                
                losses = self.loss_fn(output, I_gt, aux)
                total_loss = losses['total']

            self.optimizer.zero_grad()
            
            # Backward pass - handle gradient scaling for fp16
            if self.scaler is not None:
                # FP16 path with gradient scaling
                self.scaler.scale(total_loss).backward()
                
                if hasattr(self.model, 'sync_gradients'):
                    self.model.sync_gradients()
                
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if hasattr(self.model, 'models'):
                        nn.utils.clip_grad_norm_(self.model.models[0].parameters(), self.config.grad_clip)
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # FP32 or BF16 path (no gradient scaling needed)
                total_loss.backward()
                
                if hasattr(self.model, 'sync_gradients'):
                    self.model.sync_gradients()
                
                if self.config.grad_clip > 0:
                    if hasattr(self.model, 'models'):
                        nn.utils.clip_grad_norm_(self.model.models[0].parameters(), self.config.grad_clip)
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()

            if hasattr(self.model, 'sync_parameters'):
                self.model.sync_parameters()

            # Metrics calculation (always in fp32 for accuracy)
            with torch.no_grad():
                output_fp32 = output.float() if output.dtype != torch.float32 else output
                I_gt_fp32 = I_gt.float() if I_gt.dtype != torch.float32 else I_gt
                
                output_resized = self.metrics_calc.resize_for_metrics(output_fp32)
                I_gt_resized = self.metrics_calc.resize_for_metrics(I_gt_fp32)
                batch_psnr = self.metrics_calc.calculate_psnr(output_resized, I_gt_resized)
                batch_ssim = self.metrics_calc.calculate_ssim(output_resized, I_gt_resized)
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim

            for k in epoch_losses:
                if k in losses:
                    epoch_losses[k] += float(losses[k].item())

            num_batches += 1

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'freq': f"{losses.get('frequency', 0):.4f}",
                'edge': f"{losses.get('edge', 0):.4f}",
                'psnr': f"{batch_psnr:.2f}",
                'ssim': f"{batch_ssim:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                'prec': self.precision
            })

            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': total_loss.item(),
                    'train/batch_frequency_loss': float(losses.get('frequency', 0)),
                    'train/batch_edge_loss': float(losses.get('edge', 0)),
                    'train/batch_psnr': batch_psnr,
                    'train/batch_ssim': batch_ssim,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'step': epoch * len(self.train_loader) + batch_idx
                })

        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)
        epoch_psnr /= max(1, num_batches)
        epoch_ssim /= max(1, num_batches)
        epoch_losses['psnr'] = epoch_psnr
        epoch_losses['ssim'] = epoch_ssim
        return epoch_losses

    def validate_epoch(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = {'total': 0, 'l1': 0, 'frequency': 0, 'edge': 0, 'perceptual': 0, 'consistency': 0}
        val_psnr = 0
        val_ssim = 0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}/{self.config.epochs}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                I0_all = batch['I0'].to(self.device, non_blocking=True)
                I1_all = batch['I1'].to(self.device, non_blocking=True)
                flows_all = batch['flows'].to(self.device, non_blocking=True)
                masks_all = batch['masks'].to(self.device, non_blocking=True)
                timesteps = batch['timesteps'].to(self.device, non_blocking=True)
                I_gt = batch['I_gt'].to(self.device, non_blocking=True)

                # Use mixed precision for validation too
                with autocast(device_type=self.autocast_device, dtype=self.amp_dtype, enabled=self.use_amp):
                    output, aux = self.model(I0_all, I1_all, flows_all, masks_all, timesteps)
                    if output.device != I_gt.device:
                        output = output.to(I_gt.device)
                    
                    losses = self.loss_fn(output, I_gt, aux)

                # Convert to fp32 for metrics
                output_fp32 = output.float() if output.dtype != torch.float32 else output
                I_gt_fp32 = I_gt.float() if I_gt.dtype != torch.float32 else I_gt
                
                output_resized = self.metrics_calc.resize_for_metrics(output_fp32)
                I_gt_resized = self.metrics_calc.resize_for_metrics(I_gt_fp32)
                batch_psnr = self.metrics_calc.calculate_psnr(output_resized, I_gt_resized)
                batch_ssim = self.metrics_calc.calculate_ssim(output_resized, I_gt_resized)
                val_psnr += batch_psnr
                val_ssim += batch_ssim

                for k in val_losses:
                    if k in losses:
                        val_losses[k] += float(losses[k].item())

                num_batches += 1
                pbar.set_postfix({'loss': f"{losses['total'].item():.4f}", 'psnr': f"{batch_psnr:.2f}", 'ssim': f"{batch_ssim:.4f}"})

        for k in val_losses:
            val_losses[k] /= max(1, num_batches)
        val_psnr /= max(1, num_batches)
        val_ssim /= max(1, num_batches)
        val_losses['psnr'] = val_psnr
        val_losses['ssim'] = val_ssim
        return val_losses


    def save_sample_images(self, epoch, num_samples=4):
        self.model.eval()
        with torch.no_grad():
            sample_batch = next(iter(self.val_loader if self.val_loader else self.train_loader))
            num_samples = min(num_samples, sample_batch['I0'].shape[0])

            I0_all = sample_batch['I0'][:num_samples].to(self.device)
            I1_all = sample_batch['I1'][:num_samples].to(self.device)
            flows_all = sample_batch['flows'][:num_samples].to(self.device)
            masks_all = sample_batch['masks'][:num_samples].to(self.device)
            timesteps = sample_batch['timesteps'][:num_samples].to(self.device)
            I_gt = sample_batch['I_gt'][:num_samples].to(self.device)

            output, aux = self.model(I0_all, I1_all, flows_all, masks_all, timesteps)
            output = output.clamp(0, 1).cpu()
            I_gt = I_gt.clamp(0, 1).cpu()
            I0 = I0_all[:, 0].clamp(0, 1).cpu()
            I1 = I1_all[:, 0].clamp(0, 1).cpu()

            for idx in range(num_samples):
                images = []
                images.append(I0[idx])
                images.append(I1[idx])
                images.append(I_gt[idx])
                images.append(output[idx])

                error = torch.abs(I_gt[idx] - output[idx]) * 5.0
                error = error.clamp(0, 1)
                images.append(error)

                grid = torch.cat(images, dim=2)
                save_path = self.sample_dir / f'epoch_{epoch:04d}_sample_{idx:02d}.png'
                torchvision.utils.save_image(grid, save_path)

            batch_grid = torchvision.utils.make_grid(torch.cat([I0, I1, I_gt, output], dim=0),
                                                     nrow=num_samples, normalize=True, scale_each=True)
            batch_path = self.sample_dir / f'epoch_{epoch:04d}_batch.png'
            torchvision.utils.save_image(batch_grid, batch_path)

            if self.use_wandb and getattr(self.config, 'save_samples_to_wandb', True):
                wandb.log({'samples/batch_grid': wandb.Image(batch_grid.permute(1, 2, 0).numpy()), 'epoch': epoch})

            self.writer.add_image('samples/inputs_0', torchvision.utils.make_grid(I0, nrow=2, normalize=True), epoch)
            self.writer.add_image('samples/inputs_1', torchvision.utils.make_grid(I1, nrow=2, normalize=True), epoch)
            self.writer.add_image('samples/ground_truth', torchvision.utils.make_grid(I_gt, nrow=2, normalize=True), epoch)
            self.writer.add_image('samples/predictions', torchvision.utils.make_grid(output, nrow=2, normalize=True), epoch)

            output_resized = self.metrics_calc.resize_for_metrics(output)
            gt_resized = self.metrics_calc.resize_for_metrics(I_gt)
            sample_psnr = self.metrics_calc.calculate_psnr(output_resized, gt_resized)
            sample_ssim = self.metrics_calc.calculate_ssim(output_resized, gt_resized)
            print(f"  Sample metrics - PSNR: {sample_psnr:.2f} dB, SSIM: {sample_ssim:.4f}")
            print(f"  Samples saved to: {self.sample_dir}")

        self.model.train()
        return sample_psnr, sample_ssim

    def save_checkpoint(self, epoch, is_best=False, is_best_val=False):
        if hasattr(self.model, 'models'):
            model_state = self.model.models[0].state_dict()
            cpu_state = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_state.items()}
            model_state = cpu_state
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,  # Add scaler state
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'best_val_psnr': self.best_val_psnr,
            'best_val_ssim': self.best_val_ssim,
            'epoch_metrics': self.epoch_metrics,
            'config': vars(self.config),
            'precision': self.precision,  # Save precision setting
            'wandb_run_id': wandb.run.id if self.use_wandb else None
        }

        if epoch % self.config.save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            if self.use_wandb:
                wandb.save(str(checkpoint_path))

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print("  Best training model saved!")
            if self.use_wandb:
                wandb.save(str(best_path))
                wandb.run.summary["best_psnr"] = self.best_psnr
                wandb.run.summary["best_ssim"] = self.best_ssim

        if is_best_val:
            best_val_path = self.checkpoint_dir / 'best_val_model.pth'
            torch.save(checkpoint, best_val_path)
            print("  Best validation model saved!")
            if self.use_wandb:
                wandb.save(str(best_val_path))
                wandb.run.summary["best_val_psnr"] = self.best_val_psnr
                wandb.run.summary["best_val_ssim"] = self.best_val_ssim

        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if hasattr(self.model, 'models'):
            state_dict = checkpoint['model_state_dict']
            for i, model in enumerate(self.model.models):
                device_state = {k: (v.cuda(i) if torch.is_tensor(v) else v) for k, v in state_dict.items()}
                model.load_state_dict(device_state)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore scaler state if using fp16
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint.get('best_psnr', 0)
        self.best_ssim = checkpoint.get('best_ssim', 0)
        self.best_val_psnr = checkpoint.get('best_val_psnr', 0)
        self.best_val_ssim = checkpoint.get('best_val_ssim', 0)
        self.epoch_metrics = checkpoint.get('epoch_metrics', [])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}")
        print(f"Best train PSNR: {self.best_psnr:.2f} dB, Best val PSNR: {self.best_val_psnr:.2f} dB")
        if 'precision' in checkpoint:
            print(f"Model was trained with precision: {checkpoint['precision']}")


    def train(self):
        print(f"\nStarting training for {self.config.epochs} epochs")
        print("-" * 60)
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = datetime.now()

            train_losses = self.train_epoch(epoch)
            val_losses = None
            if self.val_loader is not None:
                val_losses = self.validate_epoch(epoch)

            sample_interval = getattr(self.config, 'sample_interval', 5)
            if epoch % sample_interval == 0 or epoch == 0:
                print("\nSaving visual samples...")
                self.save_sample_images(epoch, num_samples=getattr(self.config, 'num_samples', 4))

            is_best = train_losses['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = train_losses['psnr']
                self.best_ssim = train_losses['ssim']
                print(f"  New best training PSNR: {self.best_psnr:.2f} dB")

            is_best_val = False
            if val_losses is not None:
                is_best_val = val_losses['psnr'] > self.best_val_psnr
                if is_best_val:
                    self.best_val_psnr = val_losses['psnr']
                    self.best_val_ssim = val_losses['ssim']
                    print(f"  New best validation PSNR: {self.best_val_psnr:.2f} dB")

            if self.scheduler:
                self.scheduler.step()

            for name, value in train_losses.items():
                self.writer.add_scalar(f'train/{name}', value, epoch)
            if val_losses is not None:
                for name, value in val_losses.items():
                    self.writer.add_scalar(f'val/{name}', value, epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            if self.use_wandb:
                wb = {f'train/{k}': v for k, v in train_losses.items()}
                if val_losses is not None:
                    wb.update({f'val/{k}': v for k, v in val_losses.items()})
                wb['epoch'] = epoch
                wb['learning_rate'] = self.optimizer.param_groups[0]['lr']
                wandb.log(wb)

            epoch_time = (datetime.now() - epoch_start).total_seconds()
            print(f"\nEpoch {epoch}/{self.config.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_losses['total']:.4f}, Freq: {train_losses.get('frequency', 0):.4f}, Edge: {train_losses.get('edge', 0):.4f}, PSNR: {train_losses['psnr']:.2f} dB, SSIM: {train_losses['ssim']:.4f}")
            if val_losses is not None:
                print(f"  Val   - Loss: {val_losses['total']:.4f}, PSNR: {val_losses['psnr']:.2f} dB, SSIM: {val_losses['ssim']:.4f}")

            self.save_checkpoint(epoch, is_best, is_best_val)

            record = {'epoch': epoch, 'train': train_losses, 'lr': self.optimizer.param_groups[0]['lr']}
            if val_losses is not None:
                record['val'] = val_losses
            self.epoch_metrics.append(record)

            with open(self.checkpoint_dir / 'metrics.json', 'w') as f:
                json.dump(self.epoch_metrics, f, indent=2, default=str)

        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        print("\nTraining completed!")
        print(f"Best training PSNR: {self.best_psnr:.2f} dB, SSIM: {self.best_ssim:.4f}")
        if self.val_loader is not None:
            print(f"Best validation PSNR: {self.best_val_psnr:.2f} dB, SSIM: {self.best_val_ssim:.4f}")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Train AnchorFusionNet')

    # Dataset
    parser.add_argument('--gt_paths', '--gt_path', type=str, nargs='+', required=True, dest='gt_paths')
    parser.add_argument('--steps', '--step', type=int, nargs='+', required=True, dest='steps')

    # Validation
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--val_paths', '--val_path', type=str, nargs='+', default=None, dest='val_paths')
    parser.add_argument('--val_steps', '--val_step', type=int, nargs='+', default=None, dest='val_steps')

    # Model
    parser.add_argument('--num_anchors', type=int, default=3)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--max_attention_size', type=int, default=96*96)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--UHD', action='store_true')
    parser.add_argument('--rife_model_dir', type=str, default='ckpt/rifev4_25')

    # Parallelism
    parser.add_argument('--model_parallel', action='store_true')
    parser.add_argument('--parallel_strategy', type=str, default='spatial', choices=['spatial', 'batch'])

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=0.8)
    parser.add_argument('--lambda_freq', type=float, default=0.3)
    parser.add_argument('--lambda_edge', type=float, default=0.2)
    parser.add_argument('--lambda_perceptual', type=float, default=0.15)
    parser.add_argument('--lambda_consistency', type=float, default=0.1)

    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)

    # Logging and saving
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_interval', type=int, default=10)

    # Samples
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--save_flow_viz', action='store_true')
    parser.add_argument('--sample_on_best', action='store_true')

    # W&B
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='anchor-fusion-model')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_watch_model', action='store_true')

    # Metrics
    parser.add_argument('--metric_size', type=int, nargs=2, default=[256, 256])

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mix_strategy', type=str, default='uniform')
    parser.add_argument('--cache_flows', action='store_true')

    parser.add_argument('--precision', type=str, default='fp32', 
                    choices=['fp32', 'fp16', 'bf16'],
                    help='Training precision: fp32 (default), fp16, or bf16')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = AnchorFusionTrainer(args)
    trainer.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()