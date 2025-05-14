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


def read_data(path):
    """
    Read the file names and plot the spatial space into a corresponding grid.
    return: NDarray, with all (x,y)
    """

    file_names = os.listdir(path)
    names = natsorted(file_names) #sorted by the real value not by ascii
    grid, xdim, ydim = _get_grid(names)
    img = mpimg.imread(path + "Scan3_cropped_phasemap.png")
    #print(grid)
    utils._plot_grid(grid, img)
    print("x dimensions", xdim+1, "\n", "y dimensions", ydim+1, "\n", "in multiples of 1")

    return grid


# Set components

#select the pattern within the ROI
#return list of (path+key)= the location of each EBSP
def set_component(x_range, y_range, path, grid, plot_flag= True):
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
    img = mpimg.imread(path + "Scan3_cropped_phasemap.png")
    if plot_flag:
        utils._plot_component(roi, grid, img)
    else:
        pass
    # print(roi)
    return [lis[0] for lis in roi], [lis[1:] for lis in roi]


# Set region of interest (ROI)
def set_ROI(x_range, y_range, path, grid):
    # this function will overlay the ROI over the plot.
    x_l = x_range[0]
    x_u = x_range[1]

    y_l = y_range[0]
    y_u = y_range[1]

    

    roi = []
    index = []
    count =-1
    for key, value in coords.items():
        xy = np.asarray(list(value))
        x = xy[0]
        y = xy[1]
        count+=1
        # print(x,y)
        if (x >= x_l and x <= x_u) and (y >= y_l and y <= y_u):
            # print(key, value)
            # roi[key].append(value)
            roi.append((path + key, x, y))
            index.append(count)

    img = mpimg.imread(path + "Scan3_cropped_phasemap.png")
    utils._plot_ROI(roi, grid, img)

    return [lis[0] for lis in roi], [lis[1:] for lis in roi]


def _get_grid(names):
    """
    Sorts and reads the file names as a grid
    return a dic(filename,max_x,max_y)
    """

    for file in names:

        if file.endswith(".jpg"):
            coords[file] = _coordinate_extract(file)
            # coords.append(coordinate_extract(file))
    
    
    #coords is a dict    
    coord_grid = np.asarray(list(coords.values()))  # to extract only the coordinates
    #print(coord_grid)
    # get dimensions of the EBSD space, indexed points.
    x_dim = max(coord_grid[:, 0])
    y_dim = max(coord_grid[:, 1])

    return (coord_grid, x_dim, y_dim)


def _coordinate_extract(filename):
    """
    Extract the patterns of the files for reading and using as grid

    """
    pattern = re.compile(
        r"x_+(\d+)_y_+(\d+)(\.jpg)"
    )  # searches for this particular pattern only.
    #a regular expression object

    mo = pattern.search(filename)
    
    x = int(mo.group(1))
    y = int(mo.group(2))
    
    return (x, y)
