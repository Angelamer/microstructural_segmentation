import os
import re
import numpy as np
import pandas as pd
import kikuchipy as kp
import cv2
import torch
from torch.utils.data import Dataset
from skimage.exposure import rescale_intensity
# from data_processing import signal_process

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

# def img_normalization(pattern):
#     """
#     Uses the rescale_intensity function to transform grayscale values between 0-1
#     """

#     # temp = np.zeros((120,120)) # the size of the image.
#     pattern = rescale_intensity(
#         pattern, out_range="float"
#     )  # out_range can be adjusted.
#     # print('pattern_normalized')
#     return pattern

def coord_phase_dict_from_dataframe(df):
    """
    Build a dictionary mapping (x_index, y_index) → phase_id 
    from a DataFrame of EBSD/EDS scan data.

    Args:
        df (pd.DataFrame): Must contain columns 'x_indice', 'y_indice', 'phase_id'.
        step (float): Scan step size (not used in indexing, just for reference).

    Returns:
        dict: phase_dict where keys are (ix, iy) tuples and values are phase_id.
              Dictionary is filled in column-major order (x fixed, iterate over y).
    """
    
    # Extract required columns
    x_indices = df['x_indice'].values
    y_indices = df['y_indice'].values
    phase_ids = df['phase_id'].values
    
   
    # Determine maximum indices
    max_ix = np.max(x_indices)
    max_iy = np.max(y_indices)
    
    # Initialize array with -1 (meaning "no phase")
    phase_array = np.full((max_iy + 1, max_ix + 1), -1)
    
    # Fill phase array
    for ix, iy, pid in zip(x_indices, y_indices, phase_ids):
        phase_array[iy, ix] = pid
    
    # Build dictionary (column-major order: loop over y for each x)
    phase_dict = {}
    for ix in range(max_ix + 1):
        for iy in range(max_iy + 1):
            pid = phase_array[iy, ix]
            if pid != -1:
                phase_dict[(ix, iy)] = pid
    
    return phase_dict

def coord_xmap_dict(xmap, step=0.05):
    """
    construct the mapping dictionary for the xmap and kikuchi patterns' x/y index
    Args：
    - xmap: indexing information
    - step: scan step size (adjusted based on the experiment)

    Returns：
    - phase_dict: {(ix, iy): phase_id}
    """

    x_indices = np.round(xmap.x / step).astype(int)
    y_indices = np.round(xmap.y / step).astype(int)
    phase_ids = xmap.phase_id
    
    # Obtain the max value
    max_ix = np.max(x_indices)
    max_iy = np.max(y_indices)
    
    phase_dict = {}

    # change the x,y to y,x 
    phase_array = np.full((max_iy + 1, max_ix + 1), -1)
    for ix, iy, pid in zip(x_indices, y_indices, phase_ids):
        phase_array[iy, ix] = pid
    
    # reconstruct x,y, pid
    for ix in range(max_ix + 1):
        for iy in range(max_iy + 1):
            pid = phase_array[iy, ix]
            if pid != -1:
                phase_dict[(ix, iy)] = pid

    return phase_dict
    
# def custom_collate(batch):
    """
    Process the batch dataset of return format: ((ix,iy), image) 
    Returns：
        coords: tensor [batch_size, 2] (x_idx, y_idx)
        images: tensor [batch_size, C, H, W]
    """
    coords = torch.tensor([item[0] for item in batch])  # shape: (B,2)
    images = torch.stack([item[1] for item in batch])   # shape: (B,C,H,W)
    return coords, images

# define the selection of files within ROI
# def filter_files_by_coordinates(folder_path, x_range=(0, 100), y_range=(0, 100)):
    """
    Filter files based on coordinate patterns in filenames.
    
    Args:
        folder_path: path of file folder
        x_range: tuple (x_min, x_max)
        y_range: tuple (y_min, y_max)
    
    Returns:
        List: paths of files that match coordinate patterns and fall within specified ranges
    """
    valid_files = []
    
    # Define coordinate extraction patterns
    patterns = [
        # Pattern 1: Labeled format - x_<int> _ y_<int> .ext
        (re.compile(r"x_+\s*(\d+)[_\s]*y_+\s*(\d+)\.(?:jpg|jpeg|tif|tiff|png|bmp)$", re.IGNORECASE), "labeled"),
        
        # Pattern 2: Unlabeled format - <int> _ <int> .ext (interpret as y, x)
        (re.compile(r"(\d+)[_\s]+(\d+)\.(?:jpg|jpeg|tif|tiff|png|bmp)$", re.IGNORECASE), "unlabeled_yx"),
        
        # Pattern 3: scan3_x_<int>_y_<int> format (your original pattern)
        (re.compile(r"scan3_x_(\d+)_y_(\d+)", re.IGNORECASE), "scan3_format"),
        
        # Pattern 4: Alternative labeled format with different separators
        (re.compile(r"x[_\s-]*(\d+)[_\s-]*y[_\s-]*(\d+)\.(?:jpg|jpeg|tif|tiff|png|bmp)$", re.IGNORECASE), "labeled"),
    ]
    
    for filename in os.listdir(folder_path):
        name = filename.strip()
        
        # Try each pattern until we find a match
        for pat, kind in patterns:
            m = pat.search(name)
            if m:
                if kind == "labeled" or kind == "scan3_format":
                    x = int(m.group(1))
                    y = int(m.group(2))
                else:  # unlabeled_yx
                    y = int(m.group(1))
                    x = int(m.group(2))
                
                # Check if coordinates are within specified ranges
                if x_range[0] <= x < x_range[1] and y_range[0] <= y < y_range[1]:
                    valid_files.append(os.path.join(folder_path, filename))
                
                # Break after first successful match
                break
    
    return valid_files

# class KikuchiDataset(Dataset):
    def __init__(self, file_list, transform=None, step=0.05, slice_x=(86,586), slice_y=(56,456)):
        """
        Args:
            file_list: List of file paths
            transform: Optional transform to be applied on a sample
            step: Step size for processing (unused in this implementation)
            slice_x: Tuple (start, end) for x-axis slicing
            slice_y: Tuple (start, end) for y-axis slicing
        """
        self.file_list = file_list
        self.transform = transform
        self.step = step
        self.slice_x = slice_x
        self.slice_y = slice_y
        
        # Define coordinate extraction patterns
        patterns = [
            # Pattern 1: Labeled format - x_<int> _ y_<int> .ext
            (re.compile(r"x_+\s*(\d+)[_\s]*y_+\s*(\d+)\.(?:jpg|jpeg|tif|tiff|png|bmp)$", re.IGNORECASE), "labeled"),
            
            # Pattern 2: Unlabeled format - <int> _ <int> .ext (interpret as y, x)
            (re.compile(r"(\d+)[_\s]+(\d+)\.(?:jpg|jpeg|tif|tiff|png|bmp)$", re.IGNORECASE), "unlabeled_yx"),
            
            # Pattern 3: scan3_x_<int>_y_<int> format (your original pattern)
            (re.compile(r"scan3_x_(\d+)_y_(\d+)", re.IGNORECASE), "scan3_format"),
            
            # Pattern 4: Alternative labeled format with different separators
            (re.compile(r"x[_\s-]*(\d+)[_\s-]*y[_\s-]*(\d+)\.(?:jpg|jpeg|tif|tiff|png|bmp)$", re.IGNORECASE), "labeled"),
        ]
        
        self.indices = []
        for fname in file_list:
            filename = os.path.basename(fname)
            
            # Try each pattern until we find a match
            ix, iy = None, None
            for pat, kind in patterns:
                m = pat.search(filename)
                if m:
                    if kind == "labeled" or kind == "scan3_format":
                        ix = int(m.group(1))
                        iy = int(m.group(2))
                    else:  # unlabeled_yx
                        iy = int(m.group(1))
                        ix = int(m.group(2))
                    break
            
            if ix is not None and iy is not None:
                self.indices.append((ix, iy))
            else:
                # If no pattern matches, use a default value or skip
                print(f"Warning: Could not extract coordinates from filename: {filename}")
                self.indices.append((0, 0))  # Default value

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        
        # Read image
        temp = cv2.imread(img_path, 0)
        if temp is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        
        # Extract slice coordinates from tuples
        slice_x_start, slice_x_end = self.slice_x
        slice_y_start, slice_y_end = self.slice_y
        
        # Apply slicing
        temp2 = (kp.signals.EBSD(temp)).isig[slice_x_start:slice_x_end, slice_y_start:slice_y_end]
        
        # Process image
        temp3 = temp2.remove_dynamic_background(
            operation="subtract",
            filter_domain="frequency",
            std=8,
            truncate=4,
            inplace=False,
        )
        
        # Normalize to [0,1]
        image_nor = img_normalization(temp3.data).flatten()
        
        # Calculate new dimensions after slicing
        W = slice_x_end - slice_x_start
        H = slice_y_end - slice_y_start
        
        # Reshape with correct dimensions
        image = np.reshape(image_nor, (1, H, W))
        
        # Normalize to [-1, 1]
        image = (image * 2.0) - 1.0
        
        # Ensure the final output is float32
        image = image.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        ix, iy = self.indices[idx]
        return (ix, iy), image

# # class KikuchiParquetDataset(Dataset):
#     """
#     Dataset for Parquet produced by get_processed_signals_local():
#       columns: x (int), y (int), features (list<float>, length = H*W)

#     Parameters
#     ----------
#     parquet_path : str
#         Path to the .parquet file (e.g., "20min_processed_signals.parquet").
#     H, W : int
#         Pattern height and width used when saving (reshape target).
#     normalize : {'none','zero_one','minus_one_one'}, default 'minus_one_one'
#         Apply optional intensity scaling to features BEFORE reshaping:
#           - 'none'           : no change (assume already scaled)
#           - 'zero_one'       : min-max per sample -> [0,1]
#           - 'minus_one_one'  : min-max per sample -> [-1,1]
#         (Use 'none' if your features are already in [-1,1] to avoid double scaling.)
#     dtype : np.dtype
#         dtype for the tensor; float32 recommended.
#     """

#     def __init__(self, parquet_path, H, W,
#                  normalize='minus_one_one',
#                  dtype=np.float32):
#         super().__init__()
#         self.parquet_path = parquet_path
#         self.H, self.W = int(H), int(W)
#         self.normalize = normalize
#         self.dtype = dtype

#         # Read the entire parquet file into memory
#         self.df = pd.read_parquet(parquet_path)
#         self.N = len(self.df)
        
#         print(f"Loaded {self.N} samples from {parquet_path}")

#     def __len__(self):
#         return self.N

    
#     @staticmethod
#     def _minmax_scale(arr, mode):
#         if mode == 'none':
#             return arr
#         a = arr.astype(np.float32, copy=False)
#         amin = a.min()
#         amax = a.max()
#         if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
#             # degenerate; just return zeros to avoid NaNs
#             return np.zeros_like(a, dtype=np.float32)
#         if mode == 'zero_one':
#             return (a - amin) / (amax - amin)
#         elif mode == 'minus_one_one':
#             a01 = (a - amin) / (amax - amin)
#             return a01 * 2.0 - 1.0
#         else:
#             return a

#     def __getitem__(self, idx):
#         if idx < 0:
#             idx = self.N + idx
#         if idx < 0 or idx >= self.N:
#             raise IndexError(idx)

#         # Get the row directly from the pandas DataFrame
#         row = self.df.iloc[idx]
#         x = int(row['x'])
#         y = int(row['y'])
        
#         # Handle both list and numpy array formats for features
#         if hasattr(row['features'], '__array__'):
#             feats = np.asarray(row['features'], dtype=self.dtype)
#         else:
#             feats = np.array(row['features'], dtype=self.dtype)
            
#         # Optional scaling
#         feats = self._minmax_scale(feats, self.normalize)

#         # Reshape to (1, H, W)
#         if feats.size != self.H * self.W:
#             warnings.warn(f"Row {idx}: features length {feats.size} != H*W {self.H*self.W}. "
#                          f"Resizing with interpolation.")
#             # Resize using interpolation if the size doesn't match
#             feats = feats.reshape(-1, 1)  # Make it 2D for resize
#             feats = torch.from_numpy(feats).unsqueeze(0)  # Add channel dimension
#             feats = torch.nn.functional.interpolate(feats, size=(self.H, self.W), mode='bilinear')
#             feats = feats.squeeze(0).numpy()
#         else:
#             feats = feats.reshape(1, self.H, self.W)

#         # To tensor
#         img_t = torch.from_numpy(feats)
#         return (x, y), img_t
    
    
class KikuchiH5Dataset(Dataset):
    """
    Map-style dataset for an HDF5 written as:
      /images : float32, shape (N, 1, H, W)  (or (N, H, W) also supported)
      /coords : int32,   shape (N, 2)        -> [x, y] per sample

    Returns: ((x, y), torch.FloatTensor of shape (1, H, W))
    """
    def __init__(self, h5_path, normalize="none", dtype=np.float32, use_swmr=False):
        """
        Args:
          h5_path   : path to HDF5 file
          normalize : 'none' | 'zero_one' | 'minus_one_one' (per-sample min-max)
          dtype     : numpy dtype to read images as (usually np.float32)
          use_swmr  : open file in SWMR read mode for multi-worker dataloaders
        """
        self.h5_path = h5_path
        self.normalize = normalize
        self.dtype = dtype
        self.use_swmr = use_swmr

        # Light metadata read (do not keep file open)
        with h5py.File(self.h5_path, "r") as f:
            if "images" not in f or "coords" not in f:
                raise KeyError("HDF5 must contain datasets 'images' and 'coords'.")

            self.N = int(f["images"].shape[0])

            imshape = f["images"].shape  # (N,1,H,W) or (N,H,W)
            if len(imshape) == 4:
                _, self.C, self.H, self.W = imshape
            elif len(imshape) == 3:
                self.C, self.H, self.W = 1, imshape[1], imshape[2]
            else:
                raise ValueError(f"Unexpected 'images' shape: {imshape}")

            if f["coords"].shape != (self.N, 2):
                raise ValueError(f"'coords' must be (N,2); got {f['coords'].shape}")

        # Per-worker file handle cache
        self._fh = None

    def __len__(self):
        return self.N

    @staticmethod
    def _minmax_scale(a: np.ndarray, mode: str) -> np.ndarray:
        if mode == "none":
            return a
        a = a.astype(np.float32, copy=False)
        amin = np.nanmin(a)
        amax = np.nanmax(a)
        if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
            # degenerate; avoid NaNs/Infs
            return np.zeros_like(a, dtype=np.float32)
        if mode == "zero_one":
            return (a - amin) / (amax - amin)
        elif mode == "minus_one_one":
            a01 = (a - amin) / (amax - amin)
            return a01 * 2.0 - 1.0
        else:
            return a

    def _ensure_open(self):
        # One H5 handle per worker/process; SWMR=True for concurrent readers
        if self._fh is None:
            # NOTE: SWMR requires file created with libver='latest', but works fine to read otherwise
            self._fh = h5py.File(self.h5_path, "r", swmr=self.use_swmr)

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.N + idx
        if idx < 0 or idx >= self.N:
            raise IndexError(idx)

        self._ensure_open()
        imgs = self._fh["images"]
        coords = self._fh["coords"]

        # coords[i] -> [x, y]
        x, y = coords[idx]
        x = int(x); y = int(y)

        arr = imgs[idx]                        # (1,H,W) or (H,W)
        arr = np.asarray(arr, dtype=self.dtype)
        if arr.ndim == 2:                      # (H,W) -> (1,H,W)
            arr = arr[None, ...]

        # Per-sample normalization
        arr = self._minmax_scale(arr, self.normalize)

        # To tensor
        img_t = torch.from_numpy(arr).float()  # (1,H,W)
        return (x, y), img_t

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass

