import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import os
import glob
from typing import Dict, List, Tuple, Optional, Union
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import h5py
from pathlib import Path
import json
import random
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BraTSDataset(Dataset):
    """
    Optimized BraTS dataset with multicore loading and preprocessing
    """
    
    def __init__(
        self,
        data_dir: str,
        mode: str = 'train',
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_patches: int = 16,
        normalize: bool = True,
        augment: bool = True,
        cache_data: bool = True,
        num_workers: int = 4
    ):
        """
        Args:
            data_dir: Path to BraTS data directory
            mode: 'train', 'val', or 'test'
            patch_size: Size of 3D patches to extract
            num_patches: Number of patches per volume
            normalize: Whether to normalize intensities
            augment: Whether to apply data augmentation
            cache_data: Whether to cache processed data
            num_workers: Number of worker processes for loading
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.normalize = normalize
        self.augment = augment and mode == 'train'
        self.cache_data = cache_data
        self.num_workers = num_workers
        
        # Modality mappings
        self.modalities = ['flair', 't1ce', 't1', 't2']
        self.modality_suffixes = {
            'flair': '_flair.nii.gz',
            't1ce': '_t1ce.nii.gz', 
            't1': '_t1.nii.gz',
            't2': '_t2.nii.gz'
        }
        
        # Load data paths
        self.data_paths = self._load_data_paths()
        
        # Cache for processed data
        self.cache = {} if cache_data else None
        
        # Initialize thread pool for I/O
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        logger.info(f"Initialized BraTSDataset with {len(self.data_paths)} samples")
    
    def _load_data_paths(self) -> List[Dict[str, str]]:
        """Load all data paths for the dataset"""
        data_paths = []
        
        # Find all subject directories
        pattern = os.path.join(self.data_dir, "BraTS2020_training_data", "content", "BraTS20_Training_*")
        subject_dirs = glob.glob(pattern)
        
        for subject_dir in subject_dirs:
            subject_id = os.path.basename(subject_dir)
            
            # Build paths for all modalities
            paths = {'subject_id': subject_id}
            
            # Add modality paths
            for modality, suffix in self.modality_suffixes.items():
                file_path = os.path.join(subject_dir, f"{subject_id}{suffix}")
                if os.path.exists(file_path):
                    paths[modality] = file_path
            
            # Add segmentation path
            seg_path = os.path.join(subject_dir, f"{subject_id}_seg.nii.gz")
            if os.path.exists(seg_path):
                paths['segmentation'] = seg_path
            
            # Only add if all required files exist
            if len(paths) == len(self.modalities) + 2:  # +2 for subject_id and segmentation
                data_paths.append(paths)
        
        return data_paths
    
    def _load_nifti_volume(self, filepath: str) -> np.ndarray:
        """Load a single NIfTI volume"""
        try:
            nifti = nib.load(filepath)
            volume = nifti.get_fdata().astype(np.float32)
            return volume
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    def _load_subject_data(self, subject_paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """Load all modalities and segmentation for a subject"""
        subject_id = subject_paths['subject_id']
        
        # Check cache first
        if self.cache is not None and subject_id in self.cache:
            return self.cache[subject_id]
        
        # Load all modalities concurrently
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Load modalities
            modality_futures = {
                modality: executor.submit(self._load_nifti_volume, subject_paths[modality])
                for modality in self.modalities
            }
            
            # Load segmentation
            seg_future = executor.submit(self._load_nifti_volume, subject_paths['segmentation'])
            
            # Collect results
            modality_volumes = {}
            for modality, future in modality_futures.items():
                modality_volumes[modality] = future.result()
            
            segmentation = seg_future.result()
        
        # Stack modalities into 4D array (C, H, W, D)
        volume_stack = np.stack([modality_volumes[mod] for mod in self.modalities], axis=0)
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[subject_id] = (volume_stack, segmentation)
        
        return volume_stack, segmentation
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume intensities"""
        normalized = np.zeros_like(volume)
        
        for i in range(volume.shape[0]):  # For each modality
            modality = volume[i]
            mask = modality > 0  # Non-zero mask
            
            if np.any(mask):
                mean = np.mean(modality[mask])
                std = np.std(modality[mask])
                normalized[i] = np.where(mask, (modality - mean) / (std + 1e-8), 0)
            else:
                normalized[i] = modality
        
        return normalized
    
    def _extract_patches(self, volume: np.ndarray, segmentation: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract random patches from volume and segmentation"""
        patches = []
        c, h, w, d = volume.shape
        ph, pw, pd = self.patch_size
        
        # Find valid patch locations
        valid_h = max(1, h - ph + 1)
        valid_w = max(1, w - pw + 1)
        valid_d = max(1, d - pd + 1)
        
        # Extract patches
        for _ in range(self.num_patches):
            # Random patch location
            start_h = random.randint(0, valid_h - 1)
            start_w = random.randint(0, valid_w - 1)
            start_d = random.randint(0, valid_d - 1)
            
            # Extract patch
            volume_patch = volume[:, 
                                start_h:start_h + ph,
                                start_w:start_w + pw,
                                start_d:start_d + pd]
            
            seg_patch = segmentation[start_h:start_h + ph,
                                   start_w:start_w + pw,
                                   start_d:start_d + pd]
            
            patches.append((volume_patch, seg_patch))
        
        return patches
    
    def _augment_patch(self, volume: np.ndarray, segmentation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to a patch"""
        if not self.augment:
            return volume, segmentation
        
        # Random flip
        if random.random() > 0.5:
            volume = np.flip(volume, axis=1)
            segmentation = np.flip(segmentation, axis=0)
        
        if random.random() > 0.5:
            volume = np.flip(volume, axis=2)
            segmentation = np.flip(segmentation, axis=1)
        
        if random.random() > 0.5:
            volume = np.flip(volume, axis=3)
            segmentation = np.flip(segmentation, axis=2)
        
        # Random rotation (90 degrees)
        if random.random() > 0.5:
            axes = random.choice([(1, 2), (1, 3), (2, 3)])
            volume = np.rot90(volume, axes=axes)
            segmentation = np.rot90(segmentation, axes=(axes[0]-1, axes[1]-1))
        
        # Intensity augmentation
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            volume = np.power(np.abs(volume), gamma) * np.sign(volume)
        
        return volume, segmentation
    
    def __len__(self) -> int:
        return len(self.data_paths) * self.num_patches
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single patch"""
        # Determine which subject and patch
        subject_idx = idx // self.num_patches
        patch_idx = idx % self.num_patches
        
        # Load subject data
        subject_paths = self.data_paths[subject_idx]
        volume, segmentation = self._load_subject_data(subject_paths)
        
        # Normalize
        if self.normalize:
            volume = self._normalize_volume(volume)
        
        # Extract patches
        patches = self._extract_patches(volume, segmentation)
        volume_patch, seg_patch = patches[patch_idx]
        
        # Augment
        volume_patch, seg_patch = self._augment_patch(volume_patch, seg_patch)
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(volume_patch).float()
        seg_tensor = torch.from_numpy(seg_patch).long()
        
        return volume_tensor, seg_tensor
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class BraTSDataLoader:
    """
    Optimized data loader factory for BraTS dataset
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 2,
        num_workers: int = 4,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_patches: int = 16,
        pin_memory: bool = True,
        prefetch_factor: int = 2
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
    
    def get_dataloader(self, mode: str = 'train') -> DataLoader:
        """Create optimized DataLoader for specified mode"""
        
        dataset = BraTSDataset(
            data_dir=self.data_dir,
            mode=mode,
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            normalize=True,
            augment=(mode == 'train'),
            cache_data=True,
            num_workers=self.num_workers
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(mode == 'train'),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            drop_last=(mode == 'train')
        )
    
    def get_train_val_test_loaders(self, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """Create train/val/test dataloaders with specified split ratios"""
        
        # Create datasets
        train_loader = self.get_dataloader('train')
        val_loader = self.get_dataloader('val')
        test_loader = self.get_dataloader('test')
        
        return train_loader, val_loader, test_loader


# Utility functions
def collate_fn(batch):
    """Custom collate function for batching"""
    volumes, segmentations = zip(*batch)
    
    # Stack tensors
    volume_batch = torch.stack(volumes, dim=0)
    seg_batch = torch.stack(segmentations, dim=0)
    
    return volume_batch, seg_batch


def test_dataloader():
    """Test function for the dataloader"""
    data_dir = "data/braindata"
    
    # Create dataloader
    loader_factory = BraTSDataLoader(
        data_dir=data_dir,
        batch_size=2,
        num_workers=4,
        patch_size=(128, 128, 128),
        num_patches=4
    )
    
    # Test train loader
    train_loader = loader_factory.get_dataloader('train')
    
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Test batch loading
    for i, (volume, segmentation) in enumerate(train_loader):
        print(f"Batch {i}: Volume shape: {volume.shape}, Segmentation shape: {segmentation.shape}")
        if i >= 2:  # Test first 3 batches
            break
    
    print("Dataloader test completed successfully!")


if __name__ == "__main__":
    test_dataloader()