import os
import re
import numpy as np
import kikuchipy as kp
import cv2
import torch
from torch.utils.data import Dataset
from skimage.exposure import rescale_intensity

def img_normalization(pattern):
    """
    Uses the rescale_intensity function to transform grayscale values between 0-1
    """

    # temp = np.zeros((120,120)) # the size of the image.
    pattern = rescale_intensity(
        pattern, out_range="float"
    )  # out_range can be adjusted.
    # print('pattern_normalized')
    return pattern

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

class KikuchiDataset(Dataset):
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

