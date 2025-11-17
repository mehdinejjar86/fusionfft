# FusionFFT: Multi-Anchor Video Frame Interpolation

A state-of-the-art deep learning framework for high-quality video frame interpolation using multi-anchor fusion with frequency-aware processing and temporal guidance.

## Overview

FusionFFT is an advanced neural network architecture designed for video frame interpolation that combines multiple anchor frames with sophisticated frequency-domain processing. The model leverages temporal guidance, deformable attention mechanisms, and noise-preserving techniques to generate high-quality intermediate frames.

### Key Features

- **Multi-Anchor Fusion**: Leverages multiple reference frames (default: 3) for robust interpolation
- **Frequency-Aware Processing**: Adaptive frequency decoupling for detail preservation
- **Temporal Guidance**: Enhanced temporal position encoding for accurate interpolation
- **Deformable Pyramid Attention**: Memory-efficient cross-attention mechanism
- **Noise-Preserving Output**: Maintains film grain and texture through spectral swap techniques
- **UHD Support**: Optimized for 4K resolution processing
- **RIFE Integration**: Uses RIFE for optical flow extraction

## Architecture Highlights

### Core Components

1. **AdaptiveFrequencyDecoupling**: Separates low and high-frequency components for targeted processing
2. **TemporalGuidanceModule**: Encodes temporal position information using sinusoidal embeddings
3. **PyramidCrossAttention**: Deformable attention mechanism for efficient multi-scale fusion
4. **NoiseInject & ContentSkip**: Preserves film grain and texture details

### Model Pipeline

```
Input Frames (I0, I1) + Optical Flows + Masks + Timesteps
    ↓
Multi-Anchor Encoder (Low/Mid/High scales)
    ↓
Temporal Weighting & Guidance
    ↓
Pyramid Cross-Attention Fusion
    ↓
Hierarchical Decoder with Learned Upsampling
    ↓
Noise Injection & Content Skip
    ↓
Spectral Swap for High-Frequency Detail
    ↓
Output Frame
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install opencv-python-headless
pip install tifffile
pip install scikit-image
pip install tensorboard
pip install tqdm
pip install wandb  # Optional, for experiment tracking
```

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd fusionfft

# Download RIFE model weights
# Place RIFE weights in ckpt/rifev4_25/
```

## Usage

### Training

```python
python fusionfft_train.py \
    --gt_path /path/to/training/frames \
    --steps 8 \
    --num_anchors 3 \
    --base_channels 64 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --use_temporal_guidance \
    --wandb_project "fusionfft-training"
```

#### Training Arguments

- `--gt_path`: Path to ground truth frames
- `--steps`: Frame step for interpolation (e.g., 8 = interpolate 7 frames between every 8th frame)
- `--num_anchors`: Number of anchor frames (default: 3)
- `--base_channels`: Base channel count (default: 64)
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--use_temporal_guidance`: Enable temporal guidance modules (recommended)
- `--UHD`: Enable 4K processing mode
- `--mixed_precision`: Use automatic mixed precision training

### Inference

```python
from fusion_inference import FusionInference

# Initialize inference engine
inferencer = FusionInference(
    model_path='path/to/checkpoint.pth',
    rife_model_dir='ckpt/rifev4_25',
    num_anchors=3,
    base_channels=64,
    UHD=True,  # For 4K processing
    scale=0.5  # Processing scale
)

# Interpolate between frames
output = inferencer.interpolate_frame(
    frame0,
    frame1,
    timestep=0.5
)
```



## Model Architecture

### Specifications

- **Input**: RGB frames + optical flows + masks + timesteps
- **Output**: Interpolated RGB frame
- **Parameters**: ~8-15M (depending on base_channels)
- **Supported Resolutions**: 480p - 4K (with UHD mode)

### Loss Functions

The training uses a composite loss function:

1. **L1 Loss**: Pixel-wise reconstruction
2. **Frequency Loss**: Log-magnitude spectrum matching with high-frequency emphasis
3. **Edge Loss**: Sobel gradient matching
4. **Perceptual Loss**: VGG feature matching
5. **Consistency Loss**: Temporal consistency with warped priors

## Performance

### Memory Requirements

- **HD (1920x1080)**: ~8GB VRAM
- **4K (3840x2160)**: ~16GB VRAM (with UHD mode and scale=0.5)

### Speed

- **HD frames**: ~0.1-0.2s per frame (RTX 3090)
- **4K frames**: ~0.3-0.5s per frame (RTX 3090, UHD mode)

## File Structure

```
fusionfft/
├── fusionfft_model.py      # Core model architecture
├── fusionfft_train.py      # Training script
├── fusion_inference.py     # Inference engine
├── fusion_dataset.py       # Dataset loader with RIFE integration
├── model/
│   ├── loss.py           # Loss functions
│   ├── warplayer.py      # Optical flow warping
│   └── pytorch_msssim/   # SSIM implementation
├── utility/
│   └── imaging.py        # Image I/O utilities
└── ckpt/
    ├── fusion/           # Fusion model checkpoints
    └── rifev4_25/        # RIFE model files
```

## Advanced Features

### Temporal Guidance

Enable enhanced temporal guidance for improved interpolation accuracy:

```python
model = build_fusion_net(
    num_anchors=3,
    base_channels=64,
    use_temporal_guidance=True  # Enable temporal guidance
)
```

### Spectral Swap

Control high-frequency detail preservation:

```python
# During training or inference
model.spectral_alpha = 0.3  # Blend amount (0-1)
model.spectral_lo = 0.32    # Start of high-band
model.spectral_hi = 0.50    # End of band
```

### Flow Caching

Speed up training with flow caching:

```python
dataset = RIFEDatasetMulti(
    gt_paths='path/to/frames',
    steps=8,
    cache_flows=True,        # Cache extracted flows
    precompute_flows=True    # Precompute before training
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `base_channels` (try 48 or 32)
   - Enable `UHD=True` and reduce `scale`
   - Reduce batch size
   - Use mixed precision training

2. **Gradient Issues**
   - Check gradient flow with `check_gradients()` function
   - Adjust learning rate
   - Enable gradient clipping

3. **Quality Issues**
   - Increase `num_anchors` (3-5 recommended)
   - Adjust `spectral_alpha` for detail preservation
   - Fine-tune loss weights

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{fusionfft2024,
  title={FusionFFT: Multi-Anchor Video Frame Interpolation with Frequency-Aware Processing},
  author={[Your Name]},
  year={2024}
}
```

## License

[Specify your license here]

## Acknowledgments

- RIFE team for optical flow extraction
- PyTorch team for the deep learning framework
- Contributors and researchers in the video interpolation community

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact [your contact information].

---

**Note**: This is an active research project. Performance and features may vary based on configuration and dataset.
