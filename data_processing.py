#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing function which uses Kikuchipy to process individual EBSPs

Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""
import numpy as np
import kikuchipy as kp
import cv2
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from typing import List, Sequence, Union, Optional

def img_normalization(pattern):
    """
    Uses the rescale_intensity function to transform grayscale values between 0-1
    """
    from skimage.exposure import rescale_intensity

    # temp = np.zeros((300,300)) # the size of the image.
    pattern = rescale_intensity(
        pattern, out_range="float"
    )  # out_range can be adjusted.
    # print('pattern_normalized')
    return pattern

# old
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

def get_processed_signals(ROI, h, w, slice_x, slice_y):
    """
    Process a list of ROI images and return flattened, normalized signal arrays.

    This function applies `signal_process` to each image in the ROI list, crops the images
    according to the specified slices, and reshapes the resulting signals into a 2D array
    suitable for further analysis (e.g., PCA or machine learning).

    Args:
        ROI (list of str): List of image file paths or ROI identifiers to process.
        h (int): Height of the input images.
        w (int): Width of the input images.
        slice_x (tuple of int): Start and end indices for the x-axis crop (slice_x_start, slice_x_end).
        slice_y (tuple of int): Start and end indices for the y-axis crop (slice_y_start, slice_y_end).

    Returns:
        signal_processed (np.ndarray): 2D array of shape (num_images, num_features), where each row
        corresponds to a flattened, processed signal from an ROI image.

    Notes:
        - Each image is first processed by `signal_process` with the provided crop and dimensions.
        - The processed signals are flattened before returning.
        - Example: For a 31x31 EBSP pattern with a feature dimension of 150x150, each signal will be
          flattened to shape (1, 90000), and the final array will be of shape (num_images, 90000).
    """
    file_list = ROI
    signal_processed = []
    for file in file_list: 
        signal_processed.append(signal_process(file, flag="ROI", h=h, w=w, slice_x=slice_x, slice_y=slice_y))
        signal_processed = np.array(signal_processed).reshape(len(file_list),-1) 
        
    return signal_processed
    
def signal_process(temp, flag, pattern_height=239, pattern_width=239, 
                  slice_x=(45, 195), slice_y=(45, 195)):
    """
    Process Electron Backscatter Diffraction (EBSD) pattern signals, including component extraction and ROI processing.
    
    Parameters:
    ----------
    temp : various types
        - When flag="component": A 1D array that needs to be reshaped to (scan_rows, scan_cols, pattern_height, pattern_width)
        - When flag="ROI": A file path string pointing to the EBSD image file to be read
    
    flag : str
        Processing mode identifier:
        - "component": Process component extraction, input is an array, output is processed component data
        - "ROI": Process region of interest, input is a file path, output is processed ROI data
    
    pattern_height : int, optional
        Height of the EBSD pattern (default: 239)
    
    pattern_width : int, optional
        Width of the EBSD pattern (default: 239)
    
    slice_x : tuple, optional
        Slice coordinates in x-direction as (start, end) (default: (45, 195))
    
    slice_y : tuple, optional
        Slice coordinates in y-direction as (start, end) (default: (45, 195))
    
    Returns:
    -------
    numpy.ndarray
        - When flag="component": Returns processed component data with shape (pattern_height * pattern_width,)
        - When flag="ROI": Returns processed ROI data with shape (1, pattern_height * pattern_width)
    """
    
    # Extract slice coordinates from tuples
    slice_x_start, slice_x_end = slice_x
    slice_y_start, slice_y_end = slice_y
    
    if flag == "component":
        # Reshape input array to EBSD object
        c1 = kp.signals.EBSD(np.reshape(temp, (3, 3, pattern_height, pattern_width)))
        
        # Slice to focus on specific parts of Kikuchi patterns, remove edges with artifacts
        # Corrected order: x first, then y
        c2 = c1.isig[slice_x_start:slice_x_end, slice_y_start:slice_y_end]
        
        # Average EBSP with its neighbors using Gaussian window
        c2.average_neighbour_patterns(window="gaussian", std=1)
        
        # Correct dynamic background variation in EBSD
        c2.remove_dynamic_background(
            operation="subtract",  # Default
            filter_domain="frequency",  # Default
            std=8,  # Default is 1/8 of pattern width
            truncate=4,  # Default
            inplace=True,
        )
        
        # Select first EBSP in the 3x3 window
        comp = c2.inav[0, 0].data
        
        # Return a copy of array collapsed into one dimension (pixel width * height)
        comp = img_normalization(comp).flatten()
        return comp

    elif flag == "ROI":
        # Return normalized EBSD object, every EBSD has feature dimensions of pattern_width * pattern_height
        file = temp
        temp_img = cv2.imread(file, 0)
        
        # Create EBSD signal and apply slicing with corrected order
        temp2 = (kp.signals.EBSD(temp_img)).isig[slice_x_start:slice_x_end, slice_y_start:slice_y_end]
        
        # Remove dynamic background
        temp3 = temp2.remove_dynamic_background(
            operation="subtract",  # Default
            filter_domain="frequency",  # Default
            std=8,  # Default is 1/8 of pattern width
            truncate=4,  # Default
            inplace=False,
        )

        dump = img_normalization(temp3.data).flatten()  # dump (pattern_width * pattern_height,)
        shape = np.shape(dump)[0] 
        input_X = np.reshape(dump, (1, shape))

        return input_X



# def get_components(Ca, Cb):
#     """
#     to get the normalized intensities of averaged EBSPs within selected points/ components
#     """
#     dim1 = int(np.sqrt(len(Ca)))
#     dim2 = int(np.sqrt(len(Cb)))
#     if dim1 != 3 or dim2 != 3:
#         print("The component grid dimensions is not 3x3. Must be 3x3 ")

#     # component 1
#     file_list_arr = np.reshape(Ca, (dim1, dim2))  # only Ca is read
#     print("The component C1 grid shape is", np.shape(file_list_arr))
#     temp = []
#     for i in range(0, 3):
#         for j in range(0, 3):
#             # print(file_list_arr[i][j])
#             temp_file = cv2.imread(file_list_arr[i][j], 0) #grey value
#             # print(temp_file)
#             temp.append(temp_file)

#     comp1 = signal_process(temp, flag="component")

#     # component 2
#     file_list_arr = np.reshape(Cb, (dim1, dim2))  # only Cb is read
#     print("The component C2 grid shape is", np.shape(file_list_arr))
#     temp = []
#     for i in range(0, 3):
#         for j in range(0, 3):
#             temp_file = cv2.imread(file_list_arr[i][j], 0)
#             temp.append(temp_file)
#     comp2 = signal_process(temp, flag="component")

#     components = np.stack((comp1, comp2)) #Join a sequence of arrays along a new axis

#     return components

#
def get_components(
    components_files: Sequence[Sequence[Union[str, np.ndarray]]],
    grid_dim: Optional[int] = None,
    normalize: Optional[str] = None,   # None | "l2" | "minmax"
    h=200,
    w=200,
    slice_x=(0,100),
    slice_y=(0,100)
) -> np.ndarray:
    """
    Build component signals from multiple 3x3 (or NxN) tiles and stack them.

    Each component is provided as a flat list of N*N tiles (file paths or
    already-loaded grayscale arrays). For every component we:
      1) reshape to (N, N),
      2) load tiles as grayscale (if paths),
      3) call `signal_process(tiles, flag="component")`,
      4) optionally normalize the resulting 1D signal.

    Args:
        components_files:
            A sequence where each item is a flat list/tuple of length N*N for
            one component. Example: [Ca, Cb, Cc], where Ca is a list of 9
            file paths (3x3).
        grid_dim:
            If given, enforce that each component has exactly grid_dim*grid_dim
            tiles. If None, infer N = int(sqrt(len(component))).
        normalize:
            Optional normalization applied per component signal:
              - None: no normalization
              - "l2":   divide by L2 norm
              - "minmax": map to [0, 1] using per-signal min/max

    Returns:
        components: np.ndarray of shape (K, D)
            K = number of components, D = length of the processed signal
            returned by `signal_process(..., flag="component")`.

    Notes:
        - This expects a function `signal_process(tiles, flag="component")`
          available in scope that accepts a list of N*N grayscale arrays and
          returns a 1D signal for that component.
        - Tiles are read with `cv2.imread(path, 0)` if strings are provided.
    """

    def _load_gray(x):
        if isinstance(x, np.ndarray):
            # assume already grayscale
            return x
        if isinstance(x, str):
            img = cv2.imread(x, 0)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {x}")
            return img
        raise TypeError(f"Tile must be a path or numpy array, got {type(x)}")

    processed = []
    for idx, comp in enumerate(components_files):
        comp = list(comp)
        if grid_dim is None:
            n = int(np.sqrt(len(comp)))
            if n * n != len(comp):
                raise ValueError(
                    f"Component #{idx}: length {len(comp)} is not a perfect square; "
                    "please pass grid_dim explicitly."
                )
        else:
            n = int(grid_dim)
            if len(comp) != n * n:
                raise ValueError(
                    f"Component #{idx}: expected {n*n} tiles, got {len(comp)}."
                )

        # reshape only for clarity (we iterate by row/col below)
        _ = np.reshape(comp, (n, n))

        tiles = []
        for i in range(n):
            for j in range(n):
                tiles.append(_load_gray(_[i][j]))

        vec = signal_process(tiles, flag="component", pattern_height=h, pattern_width=w, slice_x=slice_x, slice_y=slice_y)  # expects 1D vector
        vec = np.asarray(vec).ravel()

        if normalize is not None:
            if normalize.lower() == "l2":
                denom = np.linalg.norm(vec) + 1e-12
                vec = vec / denom
            elif normalize.lower() == "minmax":
                vmin, vmax = np.min(vec), np.max(vec)
                if vmax > vmin:
                    vec = (vec - vmin) / (vmax - vmin)
                else:
                    vec = np.zeros_like(vec)
            else:
                raise ValueError(f"Unknown normalize='{normalize}'")

        processed.append(vec)

    components = np.stack(processed, axis=0)  # (K, D)
    return components

def get_eds_average(pos_X, pos_Y, edax, type= 'component'):
    """
    get the eds average value for each element within one component/position
    """
    elements = ['oxygen', 'Mg', 'Al', 
                'Si', 'Ti', 'Mn', 'Fe']
    
    if type == 'component' and isinstance(pos_X, tuple) and isinstance(pos_Y, tuple):
        averages = []
        for element in elements:
            try:
                eds_data = edax.inav[pos_X[0]:pos_X[1], pos_Y[0]:pos_Y[1]].xmap.prop[element]
                if eds_data.size > 0:
                    avg = np.nanmean(eds_data)
                    averages.append(round(avg, 4))
                else:
                    averages.append(np.nan)
            except KeyError:
                print(f"Warning: {element} data not found in EDS metadata!")
                averages.append(np.nan)
        return averages
    elif type == 'roi' and isinstance(pos_X, tuple) and isinstance(pos_Y, tuple):
        width = pos_X[1] - pos_X[0]
        height = pos_Y[1] - pos_Y[0]
        total_pixels = width * height
        roi_data = np.full((total_pixels, len(elements)), np.nan)
        for col_idx, element in enumerate(elements):
            try:
                eds_2d = edax.inav[pos_X[0]:pos_X[1], pos_Y[0]:pos_Y[1]].xmap.prop[element]
                
                
                eds_flat = eds_2d.flatten(order='F')
                

                roi_data[:, col_idx] = eds_flat
                
            except KeyError:
                print(f"Warning: {element} data not found in EDS metadata!")
        
        return roi_data
    else:
        point_data = []
        for element in elements:
            try:
                eds_data = edax.inav[pos_X:(pos_X+1), pos_Y:(pos_Y+1)].xmap.prop[element]
                point_data.append(eds_data)
            except KeyError:
                print(f"Warning: {element} data not found in EDS metadata!")
                point_data.append(None)
        return point_data
        


def add_gaussian_noise_to_kikuchi_patterns(image_paths, noise_std=25, output_folder=None, auto_detect_circle=True):
    """
    Add Gaussian noise to Kikuchi patterns (only within circular signal region) and save the noisy images.
    Only processes the first channel of the input image, but outputs 3-channel grayscale images.
    
    Parameters:
    -----------
    image_paths : list
        List of paths to the original Kikuchi pattern images (should be 31*31=961 images)
    noise_std : float, default=25
        Standard deviation of the Gaussian noise (higher value = more noise)
    output_folder : str, optional
        Output folder path. If None, uses './Noise_scan'
    auto_detect_circle : bool, default=True
        Whether to automatically detect the circular region
    
    Returns:
    --------
    np.ndarray : Array of noisy images with shape (961, height, width, 3)
    list : List of output file paths
    """
    
    # Set output folder
    if output_folder is None:
        output_folder = './Noise_scan'
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    noisy_images = []
    output_paths = []
    
    print(f"Processing {len(image_paths)} Kikuchi patterns...")
    print(f"Adding Gaussian noise with std={noise_std} (only within circular signal region)")
    print(f"Output folder: {output_folder}")
    
    for i, img_path in enumerate(tqdm(image_paths, desc="Adding noise")):
        try:
            # Read the first channel only (grayscale)
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                print(f"Error: Could not read image {img_path}")
                continue
                
            # Convert to float for noise addition
            img_float = img_gray.astype(np.float32)
            
            # Create circular mask for signal region
            if auto_detect_circle:
                mask = detect_circular_signal_region(img_gray)
            else:
                mask = create_noise_mask_for_circular_pattern(img_gray.shape)
            
            # Generate Gaussian noise for the first channel
            noise = np.random.normal(0, noise_std, img_float.shape).astype(np.float32)
            
            # Apply noise only to the circular signal region
            noisy_img_float = img_float + (noise * mask)
            
            # Clip values to valid range [0, 255]
            noisy_img_float = np.clip(noisy_img_float, 0, 255)
            
            # Convert back to uint8 (single channel)
            noisy_gray = noisy_img_float.astype(np.uint8)
            
            # Create 3-channel grayscale image (all channels same)
            noisy_img = cv2.cvtColor(noisy_gray, cv2.COLOR_GRAY2BGR)
            
            # Store the noisy image
            noisy_images.append(noisy_img)
            
            # Generate output filename with noise_ prefix
            original_filename = Path(img_path).name
            noisy_filename = f"{original_filename}"
            output_path = output_dir / noisy_filename
            
            # Save the noisy image as 3-channel grayscale
            cv2.imwrite(str(output_path), noisy_img)
            
            output_paths.append(str(output_path))
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Convert list to numpy array
    noisy_images_array = np.array(noisy_images)
    
    print(f"\nCompleted processing!")
    print(f"Generated {len(noisy_images)} noisy images")
    print(f"Output shape: {noisy_images_array.shape}")
    print(f"Saved to: {output_folder}")
    
    return noisy_images_array, output_paths

def detect_circular_signal_region(image, threshold=30):
    """
    Automatically detect the circular signal region in Kikuchi patterns
    
    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image (single channel)
    threshold : int, default=30
        Threshold for separating signal from background
    
    Returns:
    --------
    np.ndarray : Binary mask (1 for signal region, 0 for background)
    """
    # Create mask based on intensity threshold (assuming background is darker)
    mask = image > threshold
    
    # Find contours to get the circular region
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (should be the circular signal region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask from the largest contour
        mask_refined = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask_refined, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Convert to float and normalize
        mask_refined = mask_refined.astype(np.float32) / 255.0
        return mask_refined
    else:
        # Fallback to simple threshold-based mask
        return mask.astype(np.float32)

def create_noise_mask_for_circular_pattern(image_shape, center=None, radius=None):
    """
    Create a mask for circular Kikuchi patterns to apply noise only to the signal region
    
    Parameters:
    -----------
    image_shape : tuple
        Shape of the image (height, width)
    center : tuple, optional
        Center coordinates (y, x). If None, uses image center
    radius : int, optional
        Radius of the circular region. If None, uses min(height, width) // 2 - 10
    
    Returns:
    --------
    np.ndarray : Binary mask (1 for signal region, 0 for background) of shape (height, width)
    """
    height, width = image_shape
    
    if center is None:
        center = (height // 2, width // 2)
    
    if radius is None:
        radius = min(height, width) // 2 - 10  # Slightly smaller to avoid edge effects
    
    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    
    # Calculate distance from center
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Create circular mask
    mask = distance <= radius
    
    return mask.astype(np.float32)
