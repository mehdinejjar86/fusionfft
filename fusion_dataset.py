import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm, trange
from torch.nn import functional as F
import warnings
from model.pytorch_msssim import ssim_matlab
from utility.imaging import read_image, save_image, pad_image
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
import random
from typing import List, Union, Tuple


class RIFEDatasetMulti(Dataset):
    """
    Enhanced RIFE Dataset supporting multiple paths and steps for diverse training scenarios
    Fixed for multiprocessing with CUDA
    """
    def __init__(self, gt_paths, steps, anchor=3, scale=1.0, UHD=False, 
                 model_dir='ckpt/rifev4_25', mix_strategy='uniform', 
                 path_weights=None, cache_flows=False, precompute_flows=True):
        """
        Args:
            gt_paths (str or list): Single path or list of paths to ground truth images
            steps (int or list): Single step or list of steps for each path
            anchor (int): Number of anchor frames for flow computation
            scale (float): Scale factor for processing
            UHD (bool): Support for 4K images
            model_dir (str): Path to RIFE model directory
            mix_strategy (str): How to mix different paths ('uniform', 'weighted', 'sequential')
            path_weights (list): Weights for each path (used with 'weighted' strategy)
            cache_flows (bool): Cache extracted flows to speed up training
            precompute_flows (bool): Precompute all flows before training (recommended)
        """
        # Handle single or multiple paths
        if isinstance(gt_paths, (str, Path)):
            self.gt_paths = [Path(gt_paths)]
        else:
            self.gt_paths = [Path(p) for p in gt_paths]
        
        # Handle single or multiple steps
        if isinstance(steps, int):
            self.steps = [steps] * len(self.gt_paths)
        else:
            if len(steps) != len(self.gt_paths):
                raise ValueError(f"Number of steps ({len(steps)}) must match number of paths ({len(self.gt_paths)})")
            self.steps = steps
        
        self.anchor = anchor
        self.scale = scale
        self.UHD = UHD
        self.model_dir = model_dir
        self.mix_strategy = mix_strategy
        self.cache_flows = cache_flows
        self.precompute_flows = precompute_flows
        self.flow_cache = {} if cache_flows else None
        
        # Setup path weights for weighted sampling
        if path_weights is not None:
            if len(path_weights) != len(self.gt_paths):
                raise ValueError("path_weights must match number of paths")
            self.path_weights = np.array(path_weights) / np.sum(path_weights)
        else:
            self.path_weights = np.ones(len(self.gt_paths)) / len(self.gt_paths)
        
        # Auto-adjust scale for UHD
        if self.UHD and self.scale == 1.0:
            self.scale = 0.5
        assert self.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
        
        # Model will be loaded lazily in worker processes
        self.model = None
        self.device = None
        
        # Generate sequences for all paths
        self.all_samples = []
        self.path_sample_indices = []  # Track which samples belong to which path
        
        print(f"\nProcessing {len(self.gt_paths)} dataset paths:")
        print("-" * 50)
        
        for path_idx, (gt_path, step) in enumerate(zip(self.gt_paths, self.steps)):
            print(f"\nPath {path_idx + 1}: {gt_path}")
            print(f"  Step size: {step}")
            
            samples = self._process_single_path(gt_path, step, path_idx)
            start_idx = len(self.all_samples)
            self.all_samples.extend(samples)
            end_idx = len(self.all_samples)
            
            self.path_sample_indices.append((start_idx, end_idx))
            print(f"  Generated {len(samples)} samples")
        
        print("-" * 50)
        print(f"Total samples across all paths: {len(self.all_samples)}")
        
        # Create sample order based on mix strategy
        self._create_sample_order()
        
        # Precompute flows if requested (do this in main process)
        if self.precompute_flows and self.cache_flows:
            self._precompute_all_flows()
    
    def _setup_device(self):
        """Setup computing device - called lazily in worker processes"""
        if torch.cuda.is_available():
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # We're in a worker process
                # Use round-robin GPU assignment for workers
                gpu_id = worker_info.id % torch.cuda.device_count()
                self.device = torch.device(f"cuda:{gpu_id}")
                # Set the current CUDA device for this worker
                torch.cuda.set_device(gpu_id)
            else:
                # Main process
                self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def _load_model(self):
        """Load RIFE model - called lazily when needed"""
        if self.model is None:
            from ckpt.rifev4_25.RIFE_HDv3 import Model
            
            # Setup device first
            if self.device is None:
                self._setup_device()
            
            # Create model
            self.model = Model()
            if not hasattr(self.model, 'version'):
                self.model.version = 0
            
            # Load model weights first (before moving to device)
            self.model.load_model(self.model_dir, -1)
            self.model.eval()
            
            # Move entire model to the correct device
            # Use device() method if available, otherwise manually move
            if hasattr(self.model, 'device'):
                # This should handle the model's internal device management
                if torch.cuda.is_available():
                    # Set the device for the model's internal components
                    worker_info = torch.utils.data.get_worker_info()
                    if worker_info is not None:
                        gpu_id = worker_info.id % torch.cuda.device_count()
                        # Override the model's device method to use specific GPU
                        original_device = self.model.device
                        def custom_device():
                            # First call original device method
                            original_device()
                            # Then ensure everything is on the correct GPU
                            self.model.flownet = self.model.flownet.to(self.device)
                            if hasattr(self.model, 'contextnet'):
                                self.model.contextnet = self.model.contextnet.to(self.device)
                            if hasattr(self.model, 'unet'):
                                self.model.unet = self.model.unet.to(self.device)
                        custom_device()
                    else:
                        self.model.device()
                else:
                    self.model.device()
            else:
                # Manually move model to device
                self.model = self.model.to(self.device)
                
    
    def _precompute_all_flows(self):
        """Precompute all flows in the main process before training starts"""
        if not self.cache_flows:
            return
        
        print("\nPrecomputing flows for all samples...")
        print("This may take a while but will speed up training significantly.")
        
        # Load model in main process
        self._setup_device()
        self._load_model()
        
        # Process a subset of samples to estimate time
        total_computations = 0
        for sample_info in self.all_samples:
            total_computations += len(sample_info['anchor_info'])
        
        pbar = tqdm(total=total_computations, desc="Precomputing flows")
        
        with torch.no_grad():
            for sample_info in self.all_samples:
                gt_path = sample_info['gt_path']
                
                # Setup image properties if not done
                if not hasattr(self, 'frame_dtype'):
                    self._setup_image_properties(gt_path, sample_info['anchor_info'][0]['I0_frame'])
                
                for anchor_data in sample_info['anchor_info']:
                    I0_frame = anchor_data['I0_frame']
                    I1_frame = anchor_data['I1_frame']
                    timestep = anchor_data['timestep']
                    
                    cache_key = self._get_flow_cache_key(gt_path, I0_frame, I1_frame, timestep)
                    
                    if cache_key not in self.flow_cache:
                        # Load and process images (returns CPU tensors)
                        I0 = self._load_and_process_image(gt_path, I0_frame)
                        I1 = self._load_and_process_image(gt_path, I1_frame)
                        
                        # CRITICAL FIX: Move images to same device as model before flow extraction
                        I0_gpu = I0.to(self.device, non_blocking=True)
                        I1_gpu = I1.to(self.device, non_blocking=True)
                        
                        # Extract flow and mask on GPU
                        flow, mask = self.model.flow_extractor(I0_gpu, I1_gpu, timestep, self.scale)
                        
                        # Store in cache (move to CPU to save GPU memory)
                        self.flow_cache[cache_key] = {
                            'flow': flow.cpu(),
                            'mask': mask.cpu()
                        }
                    
                    pbar.update(1)
        
        pbar.close()
        print(f"Precomputed {len(self.flow_cache)} unique flows")
        
        # Clear model from GPU to save memory
        del self.model
        self.model = None
        torch.cuda.empty_cache()
    
    def _setup_image_properties(self, gt_path, frame_num):
        """Setup image properties based on first frame"""
        img_path = gt_path / f"{frame_num}.png"
        if not img_path.exists():
            img_path = gt_path / f"{frame_num}.jpg"
        
        frame = read_image(str(img_path), img_path.suffix)
        self.frame_dtype = frame.dtype
        
        if self.frame_dtype == np.uint8:
            self.max_val = 255.
        elif self.frame_dtype == np.uint16:
            self.max_val = 65535.
        else:
            self.max_val = 1.
        
        h, w, _ = frame.shape
        self.h, self.w = h, w
        
        # Calculate padding
        tmp = max(128, int(128 / self.scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        self.padding = (0, pw - w, 0, ph - h)
    
    def _process_single_path(self, gt_path: Path, step: int, path_idx: int) -> List[dict]:
        """Process a single dataset path and generate samples"""
        # Find all PNG files
        gt_files = list(gt_path.glob("*.png"))
        if not gt_files:
            gt_files = list(gt_path.glob("*.jpg"))
        if not gt_files:
            raise ValueError(f"No image files found in {gt_path}")
        
        # Extract frame numbers
        gt_frames = []
        for file in gt_files:
            try:
                frame_num = int(file.stem)
                gt_frames.append(frame_num)
            except ValueError:
                continue
        
        gt_frames.sort()
        print(f"  Found {len(gt_frames)} frames: {min(gt_frames)} to {max(gt_frames)}")
        
        # Generate input sequence
        input_frames = []
        current_frame = gt_frames[0]
        max_frame = gt_frames[-1]
        
        while current_frame <= max_frame:
            if current_frame in gt_frames:
                input_frames.append(current_frame)
            current_frame += step
        
        # Setup image properties if first path
        if path_idx == 0 and not hasattr(self, 'frame_dtype'):
            self._setup_image_properties(gt_path, input_frames[0])
            print(f"  Image properties: {self.h}x{self.w}, dtype: {self.frame_dtype}")
        
        # Generate valid samples for this path
        samples = []
        
        for i in range(len(input_frames) - 1):
            # Skip if we don't have enough anchor frames
            if i < self.anchor - 1 or i >= len(input_frames) - self.anchor:
                continue
            
            # Check intermediate frames
            for frame_num in range(input_frames[i] + 1, input_frames[i + 1]):
                if frame_num not in gt_frames:
                    continue
                
                anchor_info = []
                for anchor_idx in reversed(range(self.anchor)):
                    I0_index = i - anchor_idx
                    I1_index = i + anchor_idx + 1
                    
                    timestep = (frame_num - input_frames[I0_index]) / \
                              (input_frames[I1_index] - input_frames[I0_index])
                    
                    anchor_info.append({
                        'I0_frame': input_frames[I0_index],
                        'I1_frame': input_frames[I1_index],
                        'timestep': timestep
                    })
                
                samples.append({
                    'target_frame': frame_num,
                    'input_index': i,
                    'anchor_info': anchor_info,
                    'path_idx': path_idx,
                    'gt_path': gt_path,
                    'step': step
                })
        
        return samples
    
    def _create_sample_order(self):
        """Create sample ordering based on mix strategy"""
        n_samples = len(self.all_samples)
        
        if self.mix_strategy == 'sequential':
            # Use samples in order (path1, path2, path3, ...)
            self.sample_order = list(range(n_samples))
        
        elif self.mix_strategy == 'uniform':
            # Randomly shuffle all samples
            self.sample_order = list(range(n_samples))
            random.shuffle(self.sample_order)
        
        elif self.mix_strategy == 'weighted':
            # Sample based on path weights
            self.sample_order = []
            for _ in range(n_samples):
                # Choose a path based on weights
                path_idx = np.random.choice(len(self.gt_paths), p=self.path_weights)
                start_idx, end_idx = self.path_sample_indices[path_idx]
                
                if start_idx < end_idx:
                    sample_idx = np.random.randint(start_idx, end_idx)
                    self.sample_order.append(sample_idx)
        
        elif self.mix_strategy == 'balanced':
            # Ensure equal representation from each path
            self.sample_order = []
            path_iterators = []
            
            for start_idx, end_idx in self.path_sample_indices:
                indices = list(range(start_idx, end_idx))
                random.shuffle(indices)
                path_iterators.append(iter(indices))
            
            # Round-robin sampling from each path
            path_idx = 0
            exhausted_paths = set()
            
            while len(exhausted_paths) < len(self.gt_paths):
                if path_idx not in exhausted_paths:
                    try:
                        sample_idx = next(path_iterators[path_idx])
                        self.sample_order.append(sample_idx)
                    except StopIteration:
                        exhausted_paths.add(path_idx)
                
                path_idx = (path_idx + 1) % len(self.gt_paths)
    
    def shuffle(self):
        """Reshuffle the dataset (useful between epochs)"""
        if self.mix_strategy in ['uniform', 'weighted', 'balanced']:
            self._create_sample_order()
    
    def _load_and_process_image(self, gt_path: Path, frame_num: int):
        """Load and process a single image"""
        img_path = gt_path / f"{frame_num}.png"
        if not img_path.exists():
            img_path = gt_path / f"{frame_num}.jpg"
        
        img = read_image(str(img_path), img_path.suffix)
        img_tensor = torch.from_numpy(np.transpose(img.astype(np.int64), (2,0,1)))
        
        # Don't move to device here - return CPU tensor
        img_tensor = img_tensor.unsqueeze(0).float() / self.max_val
        return pad_image(img_tensor, padding=self.padding)
    
    def _get_flow_cache_key(self, gt_path: Path, I0_frame: int, I1_frame: int, timestep: float):
        """Generate cache key for flow storage"""
        return f"{gt_path}_{I0_frame}_{I1_frame}_{timestep:.3f}"
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.all_samples)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        # Get actual sample index based on mix strategy
        if hasattr(self, 'sample_order'):
            actual_idx = self.sample_order[idx % len(self.sample_order)]
        else:
            actual_idx = idx
        
        sample_info = self.all_samples[actual_idx]
        target_frame = sample_info['target_frame']
        anchor_info = sample_info['anchor_info']
        gt_path = sample_info['gt_path']
        path_idx = sample_info['path_idx']
        
        # Collect flows, masks, and timesteps for all anchors
        I0_list = []
        I1_list = []
        flows = []
        masks = []
        timesteps = []
        
        for anchor_data in anchor_info:
            I0_frame = anchor_data['I0_frame']
            I1_frame = anchor_data['I1_frame']
            timestep = anchor_data['timestep']
            
            # Load images (always on CPU)
            I0 = self._load_and_process_image(gt_path, I0_frame)
            I1 = self._load_and_process_image(gt_path, I1_frame)
            
            # Check cache for flows
            if self.cache_flows:
                cache_key = self._get_flow_cache_key(gt_path, I0_frame, I1_frame, timestep)
                
                if cache_key in self.flow_cache:
                    # Use cached flows
                    cached_data = self.flow_cache[cache_key]
                    flow = cached_data['flow']
                    mask = cached_data['mask']
                else:
                    # Compute flows (load model if needed)
                    if self.model is None:
                        self._setup_device()
                        self._load_model()
                    
                    # Move to device for computation
                    I0_gpu = I0.to(self.device, non_blocking=True)
                    I1_gpu = I1.to(self.device, non_blocking=True)
                    
                    with torch.no_grad():
                        flow, mask = self.model.flow_extractor(I0_gpu, I1_gpu, timestep, self.scale)
                    
                    # Move back to CPU
                    flow = flow.cpu()
                    mask = mask.cpu()
                    
                    # Cache if not precomputed
                    if not self.precompute_flows:
                        self.flow_cache[cache_key] = {
                            'flow': flow,
                            'mask': mask
                        }
            else:
                # Compute flows without caching
                if self.model is None:
                    self._setup_device()
                    self._load_model()
                
                I0_gpu = I0.to(self.device, non_blocking=True)
                I1_gpu = I1.to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    flow, mask = self.model.flow_extractor(I0_gpu, I1_gpu, timestep, self.scale)
                
                flow = flow.cpu()
                mask = mask.cpu()
            
            I0_list.append(I0.squeeze(0))
            I1_list.append(I1.squeeze(0))
            flows.append(flow.squeeze(0))
            masks.append(mask.squeeze(0).squeeze(0))
            timesteps.append(timestep)
        
        # Load ground truth
        I_gt = self._load_and_process_image(gt_path, target_frame)
        
        # Stack tensors (all on CPU)
        I0_stacked = torch.stack(I0_list, dim=0)
        I1_stacked = torch.stack(I1_list, dim=0)
        flows_stacked = torch.stack(flows, dim=0)
        masks_stacked = torch.stack(masks, dim=0)
        timesteps_tensor = torch.tensor(timesteps, dtype=torch.float32)
        
        return {
            'I0': I0_stacked,
            'I1': I1_stacked,
            'flows': flows_stacked,
            'masks': masks_stacked,
            'timesteps': timesteps_tensor,
            'I_gt': I_gt.squeeze(0),
            'target_frame': target_frame,
            'path_idx': path_idx
        }
    
    def get_sample_info(self, idx):
        """Get human-readable information about a sample"""
        if hasattr(self, 'sample_order'):
            actual_idx = self.sample_order[idx % len(self.sample_order)]
        else:
            actual_idx = idx
        
        sample_info = self.all_samples[actual_idx]
        return {
            'target_frame': sample_info['target_frame'],
            'anchor_pairs': [(info['I0_frame'], info['I1_frame']) for info in sample_info['anchor_info']],
            'timesteps': [info['timestep'] for info in sample_info['anchor_info']],
            'path': str(sample_info['gt_path']),
            'step': sample_info['step'],
            'path_idx': sample_info['path_idx']
        }
    
    def get_path_statistics(self):
        """Get statistics about samples from each path"""
        stats = {}
        for idx, (path, step) in enumerate(zip(self.gt_paths, self.steps)):
            start_idx, end_idx = self.path_sample_indices[idx]
            stats[str(path)] = {
                'step': step,
                'num_samples': end_idx - start_idx,
                'percentage': (end_idx - start_idx) / len(self.all_samples) * 100
            }
        return stats


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    batched = {}
    for key in batch[0].keys():
        if key in ['target_frame', 'path_idx']:
            batched[key] = [sample[key] for sample in batch]
        else:
            batched[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return batched


def create_multi_dataloader(gt_paths, steps, anchor=3, scale=1.0, UHD=False,
                           batch_size=4, shuffle=True, num_workers=0, 
                           model_dir='ckpt/rifev4_25', mix_strategy='uniform',
                           path_weights=None, cache_flows=False, precompute_flows=True):
    """
    Create a DataLoader with multiple dataset paths
    """
    # Force single worker when not caching flows to avoid CUDA issues
    if not cache_flows and num_workers > 0 and torch.cuda.is_available():
        print("Warning: Disabling multiprocessing (num_workers=0) when cache_flows=False to avoid CUDA conflicts")
        num_workers = 0
    
    # Set multiprocessing start method for CUDA
    if num_workers > 0 and torch.cuda.is_available():
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    dataset = RIFEDatasetMulti(
        gt_paths=gt_paths,
        steps=steps,
        anchor=anchor,
        scale=scale,
        UHD=UHD,
        model_dir=model_dir,
        mix_strategy=mix_strategy,
        path_weights=path_weights,
        cache_flows=cache_flows,
        precompute_flows=precompute_flows and cache_flows
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available() and num_workers > 0,
        persistent_workers=num_workers > 0
    )
    
    return dataloader, dataset


def main():
    """Example usage and testing"""
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description='Multi-Path RIFE Dataset')
    parser.add_argument('--gt_paths', type=str, nargs='+', required=True,
                        help='Paths to ground truth images (can specify multiple)')
    parser.add_argument('--steps', type=int, nargs='+', required=True,
                        help='Step sizes for each path')
    parser.add_argument('--anchor', type=int, default=3,
                        help='Number of anchor frames')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor')
    parser.add_argument('--mix_strategy', type=str, default='uniform',
                        choices=['uniform', 'weighted', 'sequential', 'balanced'],
                        help='How to mix samples from different paths')
    parser.add_argument('--path_weights', type=float, nargs='+',
                        help='Weights for each path (for weighted strategy)')
    parser.add_argument('--cache_flows', action='store_true',
                        help='Cache extracted flows')
    parser.add_argument('--no_precompute', action='store_true',
                        help='Disable flow precomputation')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--test_samples', type=int, default=10,
                        help='Number of samples to test')
    
    args = parser.parse_args()
    
    # Create multi-path dataset
    print("Creating multi-path dataset...")
    dataloader, dataset = create_multi_dataloader(
        gt_paths=args.gt_paths,
        steps=args.steps,
        anchor=args.anchor,
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mix_strategy=args.mix_strategy,
        path_weights=args.path_weights,
        cache_flows=args.cache_flows,
        precompute_flows=not args.no_precompute
    )
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    stats = dataset.get_path_statistics()
    for path, info in stats.items():
        print(f"Path: {path}")
        print(f"  Step: {info['step']}")
        print(f"  Samples: {info['num_samples']} ({info['percentage']:.1f}%)")
    print("-" * 50)
    print(f"Total samples: {len(dataset)}")
    print(f"Mix strategy: {args.mix_strategy}")
    
    # Test loading
    print(f"\nTesting data loading with {args.num_workers} workers...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:
            break
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch shapes: I0={batch['I0'].shape}, I_gt={batch['I_gt'].shape}")
        print(f"  Flows shape: {batch['flows'].shape}")
        print(f"  Masks shape: {batch['masks'].shape}")
    
    print("\nDataset test completed successfully!")


if __name__ == "__main__":
    main()