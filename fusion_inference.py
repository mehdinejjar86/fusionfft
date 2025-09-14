import os
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import warnings
import pickle
from torch.nn import functional as F
import tifffile

# Import your model and utilities
from fusionfft_model import create_fusion_model


def read_image(image_path, extension):
    """Read image using the original method that handles BGR to RGB conversion"""
    flag_cv2 = False
    
    if extension == '.tif':
        image = tifffile.imread(image_path)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        flag_cv2 = True
    
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if flag_cv2:
        image = image[:, :, ::-1].copy()  # Convert BGR to RGB
    
    return image


def save_image_from_numpy(img_array, filepath, dtype, max_val):
    """
    Save numpy array image to file, matching original save_image behavior
    
    Args:
        img_array: Numpy array of shape (H, W, 3) in RGB format
        filepath: Path to save the image
        dtype: Original data type (np.uint8, np.uint16, etc.)
        max_val: Maximum value for the data type
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct dtype and range
    img_array = img_array.astype(dtype)
    
    # Convert RGB to BGR for OpenCV saving (matching original behavior)
    img_bgr = img_array[:, :, ::-1]
    
    # Save based on extension
    if filepath.suffix == '.tif':
        tifffile.imwrite(str(filepath), img_bgr)
    else:
        cv2.imwrite(str(filepath), img_bgr)


def pad_image(img, padding):
    """Pad image tensor"""
    return F.pad(img, padding)


class FusionInference:
    """
    Inference engine for Multi-Anchor Fusion Model
    Handles frame interpolation with multiple anchor points
    """
    
    def __init__(self, model_path, rife_model_dir='ckpt/rifev4_25', 
                 num_anchors=3, base_channels=64, max_attention_size=96*96,
                 device=None, scale=1.0, UHD=False, cache_flows=True,
                 skip_noise_inject=False, skip_content_skip=False):
        """
        Initialize the inference engine
        
        Args:
            model_path: Path to trained fusion model checkpoint
            rife_model_dir: Path to RIFE model for flow extraction
            num_anchors: Number of anchor frames to use
            base_channels: Base channels in fusion model
            max_attention_size: Maximum attention size for memory efficiency
            device: Computing device (auto-detect if None)
            scale: Processing scale factor
            UHD: Enable UHD (4K) mode
        """
        self.num_anchors = num_anchors
        self.scale = scale
        self.UHD = UHD
        self.cache_flows = cache_flows
        self.flow_cache = {} if cache_flows else None

        # Set skip flags
        self.skip_noise_inject = skip_noise_inject
        self.skip_content_skip = skip_content_skip
        
        # Auto-adjust scale for UHD
        if self.UHD and self.scale == 1.0:
            self.scale = 0.5
            print(f"UHD mode: adjusting scale to {self.scale}")
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple MPS")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        else:
            self.device = device
        
        # Load fusion model
        print(f"\nLoading fusion model from: {model_path}")
        self.fusion_model = create_fusion_model(
            num_anchors=num_anchors,
            base_channels=base_channels,
            max_attention_size=max_attention_size
        ).to(self.device)
        
        # Load checkpoint with compatibility for PyTorch 2.6+
        try:
            # First try with weights_only=True (safer)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as e:
            print(f"Note: Loading with weights_only=False for compatibility...")
            # Fallback to weights_only=False for older checkpoints
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.fusion_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.fusion_model.load_state_dict(checkpoint)
        self.fusion_model.eval()
        
        total_params = sum(p.numel() for p in self.fusion_model.parameters())
        print(f"Fusion model loaded: {total_params:,} parameters")
        
        # Load RIFE model for flow extraction (will be loaded on demand if caching)
        self.rife_model_dir = rife_model_dir
        self.rife_model = None
        
        if not self.cache_flows:
            # Load RIFE immediately if not caching
            self._load_rife_model()
        
        print(f"Flow caching: {'Enabled' if self.cache_flows else 'Disabled'}")
        
        # Image properties (will be set on first frame)
        self.img_dtype = None
        self.max_val = None
        self.padding = None
        self.h = None
        self.w = None
    
    def _load_rife_model(self):
        """Load RIFE model when needed"""
        if self.rife_model is None:
            print(f"\nLoading RIFE model from: {self.rife_model_dir}")
            from ckpt.rifev4_25.RIFE_HDv3 import Model as RIFEModel
            
            self.rife_model = RIFEModel()
            if not hasattr(self.rife_model, 'version'):
                self.rife_model.version = 0
            self.rife_model.load_model(self.rife_model_dir, -1)
            self.rife_model.eval()
            self.rife_model.device()
            
            print("RIFE model loaded for flow extraction")
    
    def _unload_rife_model(self):
        """Unload RIFE model to free GPU memory"""
        if self.rife_model is not None:
            del self.rife_model
            self.rife_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("RIFE model unloaded from memory")
    
    def _get_flow_cache_key(self, img0_path, img1_path, timestep):
        """Generate cache key for flow storage"""
        return f"{img0_path}_{img1_path}_{timestep:.3f}"
    
    def setup_image_properties(self, img_path):
        """Setup image properties based on first frame"""
        # Read image using the original method
        frame = read_image(str(img_path), img_path.suffix)
        self.img_dtype = frame.dtype
        
        if self.img_dtype == np.uint8:
            self.max_val = 255.0
        elif self.img_dtype == np.uint16:
            self.max_val = 65535.0
        else:
            self.max_val = 1.0
        
        h, w, _ = frame.shape
        self.h, self.w = h, w
        
        # Calculate padding for proper alignment
        tmp = max(128, int(128 / self.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        self.padding = (0, pw - w, 0, ph - h)
        
        print(f"Image properties: {h}×{w}, dtype: {self.img_dtype}, max_val: {self.max_val}, padding: {self.padding}")
    
    def load_and_process_image(self, img_path):
        """Load and preprocess image for model input"""
        img = read_image(str(img_path), img_path.suffix)
        
        # Convert to tensor and normalize (keeping RGB format)
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img_tensor = img_tensor.unsqueeze(0) / self.max_val
        
        # Pad if necessary
        img_tensor = pad_image(img_tensor, padding=self.padding)
        
        return img_tensor.to(self.device)
    
    def extract_flow_and_mask(self, img0, img1, timestep, cache_key=None):
        """Extract optical flow and mask using RIFE model with optional caching"""
        # Check cache first
        if self.cache_flows and cache_key and cache_key in self.flow_cache:
            cached_data = self.flow_cache[cache_key]
            return cached_data['flow'].to(self.device), cached_data['mask'].to(self.device)
        
        # Load RIFE model if needed
        self._load_rife_model()
        
        with torch.no_grad():
            flow, mask = self.rife_model.flow_extractor(img0, img1, timestep, self.scale)
        
        # Cache the results on CPU to save GPU memory
        if self.cache_flows and cache_key:
            self.flow_cache[cache_key] = {
                'flow': flow.cpu(),
                'mask': mask.cpu()
            }
        
        return flow, mask
    
    def precompute_all_flows(self, frame_paths, input_frames, frames_to_interpolate, step_size):
        """Precompute all flows before running fusion model"""
        if not self.cache_flows:
            return
        
        print("\nPrecomputing optical flows...")
        total_flows = len(frames_to_interpolate) * self.num_anchors
        pbar = tqdm(total=total_flows, desc="Extracting flows")
        
        # Load RIFE model for flow extraction
        self._load_rife_model()
        
        for target_idx in frames_to_interpolate:
            # Find anchor pairs (same logic as interpolate_frame)
            anchor_pairs = []
            for anchor_distance in range(1, self.num_anchors + 1):
                left_idx = None
                right_idx = None
                
                for f in input_frames:
                    if f < target_idx:
                        if left_idx is None or (target_idx - f) <= anchor_distance * step_size:
                            left_idx = f
                    elif f > target_idx:
                        if right_idx is None:
                            right_idx = f
                            break
                
                if left_idx is not None and right_idx is not None:
                    expected_distance = anchor_distance * step_size
                    actual_distance = right_idx - left_idx
                    if actual_distance == expected_distance:
                        anchor_pairs.append((left_idx, right_idx))
            
            # Pad with closest pair if needed
            while len(anchor_pairs) < self.num_anchors:
                if anchor_pairs:
                    anchor_pairs.append(anchor_pairs[-1])
                else:
                    break
            
            anchor_pairs = anchor_pairs[:self.num_anchors]
            
            # Extract flows for each anchor pair
            for left_idx, right_idx in anchor_pairs:
                timestep = (target_idx - left_idx) / (right_idx - left_idx)
                cache_key = self._get_flow_cache_key(
                    frame_paths[left_idx], 
                    frame_paths[right_idx], 
                    timestep
                )
                
                if cache_key not in self.flow_cache:
                    # Load images
                    I0 = self.load_and_process_image(frame_paths[left_idx])
                    I1 = self.load_and_process_image(frame_paths[right_idx])
                    
                    # Extract and cache flow
                    with torch.no_grad():
                        flow, mask = self.rife_model.flow_extractor(I0, I1, timestep, self.scale)
                        self.flow_cache[cache_key] = {
                            'flow': flow.cpu(),
                            'mask': mask.cpu()
                        }
                
                pbar.update(1)
        
        pbar.close()
        
        # Unload RIFE model to free GPU memory
        self._unload_rife_model()
        print(f"Precomputed {len(self.flow_cache)} unique flows")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def interpolate_frame(self, frame_paths, target_idx, step_size):
        """
        Interpolate a frame using multi-anchor fusion
        
        Args:
            frame_paths: Dictionary mapping frame numbers to paths
            target_idx: Target frame index to interpolate
            step_size: Step size between input frames
        
        Returns:
            interpolated_frame: The interpolated frame as numpy array
        """
        available_frames = sorted(frame_paths.keys())
        
        # Find the input frames around the target
        input_frames = []
        for f in available_frames:
            if f % step_size == 0:
                input_frames.append(f)
        
        # Find anchor pairs
        anchor_pairs = []
        for anchor_distance in range(1, self.num_anchors + 1):
            # Find left and right anchors at this distance
            left_idx = None
            right_idx = None
            
            for i, f in enumerate(input_frames):
                if f < target_idx:
                    if left_idx is None or (target_idx - f) <= anchor_distance * step_size:
                        left_idx = f
                elif f > target_idx:
                    if right_idx is None:
                        right_idx = f
                        break
            
            if left_idx is not None and right_idx is not None:
                # Check if this is a valid anchor pair at the expected distance
                expected_distance = anchor_distance * step_size
                actual_distance = right_idx - left_idx
                
                if actual_distance == expected_distance:
                    anchor_pairs.append((left_idx, right_idx))
        
        # We need at least one anchor pair
        if not anchor_pairs:
            print(f"  Warning: No valid anchor pairs for frame {target_idx}, skipping")
            return None
        
        # Pad with closest pair if we don't have enough anchors
        while len(anchor_pairs) < self.num_anchors:
            anchor_pairs.append(anchor_pairs[-1])
        
        # Limit to num_anchors
        anchor_pairs = anchor_pairs[:self.num_anchors]
        
        # Prepare inputs for the model
        I0_list = []
        I1_list = []
        flows_list = []
        masks_list = []
        timesteps_list = []
        
        for left_idx, right_idx in anchor_pairs:
            # Load images
            I0 = self.load_and_process_image(frame_paths[left_idx])
            I1 = self.load_and_process_image(frame_paths[right_idx])
            
            # Calculate timestep
            timestep = (target_idx - left_idx) / (right_idx - left_idx)
            
            # Generate cache key for flow
            cache_key = self._get_flow_cache_key(
                frame_paths[left_idx],
                frame_paths[right_idx],
                timestep
            ) if self.cache_flows else None
            
            # Extract flow and mask (with caching)
            flow, mask = self.extract_flow_and_mask(I0, I1, timestep, cache_key)
            
            # Store
            I0_list.append(I0)
            I1_list.append(I1)
            flows_list.append(flow)
            masks_list.append(mask.squeeze(1))  # Remove channel dim from mask
            timesteps_list.append(timestep)
        
        # Stack inputs
        I0_all = torch.cat(I0_list, dim=0).unsqueeze(0)  # [1, num_anchors, 3, H, W]
        I1_all = torch.cat(I1_list, dim=0).unsqueeze(0)
        flows_all = torch.cat(flows_list, dim=0).unsqueeze(0)  # [1, num_anchors, 4, H, W]
        masks_all = torch.cat([m.unsqueeze(0) for m in masks_list], dim=0).unsqueeze(0)  # [1, num_anchors, H, W]
        timesteps = torch.tensor(timesteps_list, device=self.device).unsqueeze(0)  # [1, num_anchors]
        
        # Run fusion model
        with torch.no_grad():
            output, _ = self.fusion_model(I0_all, I1_all, flows_all, masks_all, timesteps,
                                          skip_noise_inject=self.skip_noise_inject,
                                          skip_content_skip=self.skip_content_skip)
        
        # Remove padding if necessary
        if self.padding[1] + self.padding[3] > 0:
            output = output[:, :, :self.h, :self.w]
        
        # Convert to numpy - output is in RGB format from the model
        # Shape: [1, 3, H, W] -> [H, W, 3]
        output = output[0].cpu().numpy()  # [3, H, W]
        output = np.transpose(output, (1, 2, 0))  # [H, W, 3]
        
        # Denormalize to original range
        output = (output * self.max_val).clip(0, self.max_val)
        
        # Convert to appropriate dtype
        if self.img_dtype == np.uint8:
            output = output.astype(np.uint8)
        elif self.img_dtype == np.uint16:
            output = output.astype(np.uint16)
        else:
            output = output.astype(self.img_dtype)
        
        return output
    
    def process_directory(self, input_dir, output_dir, step_size, start_frame=None, end_frame=None):
        """
        Process a directory of frames
        
        Args:
            input_dir: Directory containing input frames
            output_dir: Directory to save interpolated frames
            step_size: Step size between input frames
            start_frame: Optional start frame number
            end_frame: Optional end frame number
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all frames
        frame_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
        
        # Build frame dictionary
        frame_paths = {}
        for f in frame_files:
            try:
                frame_num = int(f.stem)
                frame_paths[frame_num] = f
            except ValueError:
                continue
        
        if not frame_paths:
            raise ValueError(f"No valid frame files found in {input_dir}")
        
        available_frames = sorted(frame_paths.keys())
        print(f"\nFound {len(available_frames)} frames: {available_frames[0]} to {available_frames[-1]}")
        
        # Setup image properties from first frame
        first_frame_path = frame_paths[available_frames[0]]
        self.setup_image_properties(first_frame_path)
        
        # Determine processing range
        if start_frame is None:
            start_frame = available_frames[0]
        if end_frame is None:
            end_frame = available_frames[-1]
        
        # Find input frames (at step intervals)
        input_frames = []
        for f in available_frames:
            if f % step_size == 0 and start_frame <= f <= end_frame:
                input_frames.append(f)
        
        print(f"Input frames (step={step_size}): {input_frames}")
        
        # Copy input frames to output
        print("\nCopying input frames...")
        for frame_num in tqdm(input_frames):
            if frame_num in frame_paths:
                src_path = frame_paths[frame_num]
                dst_path = output_path / f"{frame_num:04d}.png"
                
                # Read and save to ensure consistent format
                img = read_image(str(src_path), src_path.suffix)
                save_image_from_numpy(img, dst_path, self.img_dtype, self.max_val)
        
        # Determine which frames need interpolation
        frames_to_interpolate = []
        for i in range(len(input_frames) - 1):
            start = input_frames[i]
            end = input_frames[i + 1]
            
            # Find intermediate frames that exist in ground truth
            for frame_num in range(start + 1, end):
                if frame_num in frame_paths:
                    # Check if we have enough anchors
                    min_required_frame = frame_num - (self.num_anchors - 1) * step_size
                    max_required_frame = frame_num + (self.num_anchors - 1) * step_size
                    
                    if min_required_frame >= input_frames[0] and max_required_frame <= input_frames[-1]:
                        frames_to_interpolate.append(frame_num)
                    else:
                        print(f"  Skipping frame {frame_num}: insufficient anchor frames")
        
        print(f"\nFrames to interpolate: {frames_to_interpolate}")
        
        # Precompute all flows if caching is enabled
        if self.cache_flows and frames_to_interpolate:
            self.precompute_all_flows(frame_paths, input_frames, frames_to_interpolate, step_size)
        
        # Interpolate frames
        if frames_to_interpolate:
            print(f"\nInterpolating {len(frames_to_interpolate)} frames...")
            
            pbar = tqdm(frames_to_interpolate, desc="Interpolating frames")
            for frame_num in pbar:
                # Update progress bar with current frame info
                pbar.set_description(f"Interpolating frame {frame_num:04d}")
                
                interpolated = self.interpolate_frame(frame_paths, frame_num, step_size)
                
                if interpolated is not None:
                    output_path_frame = output_path / f"{frame_num:04d}.png"
                    save_image_from_numpy(interpolated, output_path_frame, self.img_dtype, self.max_val)
        
        
        # Clean up if RIFE model is still loaded
        if self.cache_flows:
            self._unload_rife_model()
        
        print(f"\nProcessing complete! Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-Anchor Fusion Model Inference')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained fusion model checkpoint')
    parser.add_argument('--rife_model_dir', type=str, default='ckpt/rifev4_25',
                        help='Path to RIFE model directory for flow extraction')
    parser.add_argument('--num_anchors', type=int, default=3,
                        help='Number of anchor frames (must match training)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channels in model (must match training)')
    parser.add_argument('--max_attention_size', type=int, default=96*96,
                        help='Maximum attention size (must match training)')
    
    # Input/Output arguments
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input frames')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save interpolated frames')
    parser.add_argument('--step', type=int, required=True,
                        help='Step size between input frames (e.g., 3 for frames 0,3,6,9,...)')
    
    # Processing arguments
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Processing scale factor')
    parser.add_argument('--UHD', action='store_true',
                        help='Enable UHD (4K) mode')
    parser.add_argument('--start_frame', type=int, default=None,
                        help='Start frame number (optional)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='End frame number (optional)')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu, auto-detect if not specified)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable flow caching (uses less RAM but keeps both models in GPU)')
    
    # Add debug option
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to check image values')
    
        # Flags to skip NoiseInject or ContentSkip
    parser.add_argument('--skip_noise_inject', action='store_true', 
                        help='Skip NoiseInject module during inference')
    parser.add_argument('--skip_content_skip', action='store_true', 
                        help='Skip ContentSkip module during inference')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None
    
    # Create inference engine
    inference = FusionInference(
        model_path=args.model_path,
        rife_model_dir=args.rife_model_dir,
        num_anchors=args.num_anchors,
        base_channels=args.base_channels,
        max_attention_size=args.max_attention_size,
        device=device,
        scale=args.scale,
        UHD=args.UHD,
        cache_flows=not args.no_cache,
        skip_noise_inject=args.skip_noise_inject,
        skip_content_skip=args.skip_content_skip
    )
    
    # Process directory
    inference.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        step_size=args.step,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()