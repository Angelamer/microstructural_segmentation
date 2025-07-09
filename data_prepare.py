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
def filter_files_by_coordinates(folder_path, x_range=(0, 100), y_range=(0, 100)):
    """
    Args:
        folder_path: path of file folder
        x_range:  (x_min, x_max)
        y_range:  (y_min, y_max)
    Returns:
        List: the lists of file path
    """
    valid_files = []
    pattern = re.compile(r'scan3_x_(\d+)_y_(\d+)')  # regular expressions for matching the file names
    
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                valid_files.append(os.path.join(folder_path, filename))
    
    return valid_files


class KikuchiDataset(Dataset):
    def __init__(self, file_list, transform=None, step=0.05):
        self.file_list = file_list
        self.transform = transform  # Augmentation here?
        self.step = step
        # get the x,y from the filename
        pattern = re.compile(r'scan3_x_(\d+)_y_(\d+)')
        self.indices = []
        for fname in file_list:
            m = pattern.search(os.path.basename(fname))
            ix, iy = int(m.group(1)), int(m.group(2))
            self.indices.append((ix, iy))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        
        img_path = self.file_list[idx]
        # image = Image.open(img_path).convert('L')  # Convert into grey figure
        
        temp = cv2.imread(img_path, 0)
        if temp is None:
                # Handle case where image reading fails
                raise FileNotFoundError(f"Could not read image: {img_path}")
                                        
        temp2 = (kp.signals.EBSD(temp)).isig[60:180, 60:180]
        
        temp3 = temp2.remove_dynamic_background(
            operation="subtract",  # Default
            filter_domain="frequency",  # Default
            std=8,  # Default is 1/8 of the pattern width
            truncate=4,  # Default
            inplace=False,
        )
        
        # normalize to [0,1]
        image_nor = img_normalization(temp3.data).flatten() 
        len = np.shape(image_nor)[0]
        # reshape from (120*120，) to (1,120,120)
        H = W = int(np.sqrt(len))
        image = np.reshape(image_nor, (1, H, W))
        
        # Normalize to [-1, 1]
        image = (image * 2.0) - 1.0
        print(f"Image data range: {image.min()} to {image.max()}") # Verify range
        
        # Ensure the final output is float32 (important for PyTorch default tensors)
        image = image.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        # image_tensor = torch.from_numpy(image)
        # dataset = TensorDataset(image_tensor)
        ix, iy = self.indices[idx]
        return (ix,iy),image  # Output size (x,y),[1, H, W]


