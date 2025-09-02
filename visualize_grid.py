#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assisting module to locate the region of interest/components etc with respect to
a reference image of the microstructure.

Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""

import os
import re as re
from Nat_sort import natsorted
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import utils

coords = {}  # global variable to be used by all functions.


def read_data(path_to_kikuchi, path_to_phasemap):
    """
    Read Kikuchi pattern filenames, extract their spatial grid coordinates, and
    visualize the grid overlay on a phase map.

    Args:
        path_to_kikuchi (str): Path to the folder containing Kikuchi pattern image files.
            Filenames must encode coordinates that `_coordinate_extract` can parse.
            Only files ending with ".jpg" or ".tiff" will be considered.
        path_to_phasemap (str): Path to the phase map image file (e.g., .png, .jpg, .tiff).
            This will be used as the background for plotting the extracted grid.

    Returns:
        np.ndarray:
            A 2D numpy array of shape (N, 2) containing the (x, y) coordinates extracted
            from Kikuchi pattern filenames.

    Side Effects:
        - Plots the grid overlaid on the phase map image using `utils._plot_grid`.
        - Prints the grid dimensions `(xdim+1, ydim+1)`.

    Notes:
        - Relies on `_get_grid(names)` to parse coordinates from filenames.
        - The dimensions reported (`xdim+1`, `ydim+1`) are based on the maximum extracted
          coordinates and assume 0-based indexing.
    """

    file_names = os.listdir(path_to_kikuchi)
    names = natsorted(file_names)  # sorted numerically instead of ASCII
    grid, xdim, ydim = _get_grid(names)

    img = mpimg.imread(path_to_phasemap)

    utils._plot_grid(grid, img)
    print("x dimensions", xdim+1, "\n", "y dimensions", ydim+1, "\n", "in multiples of 1")

    return grid


# Set components

#select the pattern within the ROI
#return list of (path+key)= the location of each EBSP
def set_component(x_range, y_range, path, grid, image_path, plot_flag= True):
    x_l = x_range[0]
    x_u = x_range[1]

    y_l = y_range[0]
    y_u = y_range[1]

    roi = []
    for key, value in coords.items():
        xy = np.asarray(list(value))
        x = xy[0]
        y = xy[1]
        # print(x,y)
        if (x >= x_l and x <= x_u) and (y >= y_l and y <= y_u):
            # print(key, value)
            # roi[key].append(value)
            # path+key = file location?
            roi.append((path + key, x, y))
    img = mpimg.imread(image_path)
    if plot_flag:
        utils._plot_component(roi, grid, img)
    else:
        pass
    # print(roi)
    return [lis[0] for lis in roi], np.array([lis[1:] for lis in roi])


def set_ROI(x_range, y_range, path, grid, path_to_phasemap):
    """
    Select and visualize a rectangular Region of Interest (ROI) within the EBSD/Kikuchi grid.

    Args:
        x_range (tuple[int, int]): Lower and upper bounds for the x-coordinate (inclusive).
        y_range (tuple[int, int]): Lower and upper bounds for the y-coordinate (inclusive).
        path (str): Directory path prefix to be prepended to each filename in the ROI.
        grid (np.ndarray): The full grid of coordinates as returned by `read_data` / `_get_grid`.
        path_to_phasemap (str): Path to the phase map image, used as background for plotting.

    Returns:
        tuple[list[str], list[tuple[int, int]]]:
            - List of full file paths of Kikuchi patterns within the ROI.
            - List of corresponding (x, y) coordinates of those files.

    Side Effects:
        - Overlays the ROI on top of the phase map grid using `utils._plot_ROI`.
        - Reads and displays the phase map image.

    Notes:
        - Relies on the global dictionary `coords`, which maps filenames → (x, y) coordinates.
        - The selection includes all points with coordinates satisfying
          `x_range[0] <= x <= x_range[1]` and `y_range[0] <= y <= y_range[1]`.
    """
    x_l, x_u = x_range
    y_l, y_u = y_range

    roi = []
    index = []
    count = -1

    for key, value in coords.items():
        xy = np.asarray(list(value))
        x, y = xy[0], xy[1]
        count += 1

        if (x_l <= x < x_u) and (y_l <= y < y_u):
            roi.append((path + key, x, y))
            index.append(count)

    img = mpimg.imread(path_to_phasemap)
    utils._plot_ROI(roi, grid, img)

    return [lis[0] for lis in roi], np.array([lis[1:] for lis in roi])

def _get_grid(names):
    """
    Parse image filenames, extract spatial coordinates, and compute grid dimensions.

    Args:
        names (list of str): List of file names (e.g., from EBSD/EDS export). 
            Only files ending with ".jpg" or ".tiff" are considered.

    Returns:
        tuple:
            - coord_grid (np.ndarray): Array of shape (N, 2), containing (x, y) coordinates 
              extracted from filenames.
            - x_dim (int): Maximum x coordinate found in the grid.
            - y_dim (int): Maximum y coordinate found in the grid.

    Notes:
        - This function depends on `_coordinate_extract(file)` which should return (x, y).
        - `coords` is assumed to be a dictionary mapping filenames → (x, y).
    """

    for file in names:
        if file.endswith(".jpg") or file.endswith(".tiff") or file.endswith(".jpeg"):
            coords[file] = _coordinate_extract(file)

    # Convert dict values to numpy array
    coord_grid = np.asarray(list(coords.values()))  # shape (N, 2)

    # Determine EBSD grid dimensions
    x_dim = int(coord_grid[:, 0].max())
    y_dim = int(coord_grid[:, 1].max())

    return coord_grid, x_dim, y_dim


def _coordinate_extract(filename: str):
    """
    Extract (x, y) coordinates from a filename.

    Supported patterns (case-insensitive, with optional spaces/extra underscores):
      1) Labeled: ... x_<int> _ y_<int> .ext  -> returns (x, y)
         e.g. "abc_x_12_y_34.jpg"
      2) Unlabeled two-integer suffix: ... <int> _ <int> .ext  -> interprets as (y, x), returns (x, y) = (second, first)
         e.g. "34_12.tiff"  -> returns (12, 34)

    Extensions supported: jpg, jpeg, tif, tiff.

    Args:
        filename (str): The filename (basename or full path).

    Returns:
        tuple[int, int]: (x, y) coordinates parsed from the filename.

    Raises:
        ValueError: If no supported coordinate pattern is found.
    """
    name = filename.strip()

    patterns = [
        # Labeled: ... x_<int> _ y_<int> .ext  -> (x, y) = (g1, g2)
        (re.compile(r"x_+\s*(\d+)[_\s]*y_+\s*(\d+)\.(?:jpg|jpeg|tif|tiff)$", re.IGNORECASE), "labeled"),
        # Unlabeled: ... <int> _ <int> .ext  -> interpret as (y, x) = (g1, g2), return (x, y) = (g2, g1)
        (re.compile(r"(\d+)[_\s]+(\d+)\.(?:jpg|jpeg|tif|tiff)$", re.IGNORECASE), "unlabeled_yx"),
    ]

    for pat, kind in patterns:
        m = pat.search(name)
        if m:
            if kind == "labeled":
                x = int(m.group(1))
                y = int(m.group(2))
            else:  # unlabeled_yx
                y = int(m.group(1))
                x = int(m.group(2))
            return (x, y)

    raise ValueError(
        f"Could not extract coordinates from filename: '{filename}'. "
        "Expected patterns like 'x_<int>_y_<int>.jpg' or '<int>_<int>.tiff' (interpreted as y_x)."
    )

