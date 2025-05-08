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
    

def signal_process(temp, flag):
    
    if flag == "component":
        c1 = kp.signals.EBSD(np.reshape(temp, (3, 3, 239, 239)))  #Input:the dimension of scan points and pattern height, width; Return: EBSD Object
        c2 = c1.isig[45:195, 45:195] #slice to focus on the specific parts of the kikuchi patterns? remove the edges with artifacts?
        c2.average_neighbour_patterns(window="gaussian", std=1) #average the EBSP with it neighbors
        c2.remove_dynamic_background(
            operation="subtract",  # Default
            filter_domain="frequency",  # Default
            std=8,  # Default is 1/8 of the pattern width
            truncate=4,  # Default
            inplace=True,
        ) #correct the dynamic background variation in EBSD
        comp = c2.inav[0, 0].data #select the first EBSP in the 3*3 windows
        comp = img_normalization(comp).flatten() #return a copy of array collapsed into one dimension pixel width * height

        return comp

    elif flag == "ROI":
    #return normalized EBSD object, every EBSD has the feature dimensions of 300*300
        # file = [lis[0] for lis in temp]
        file = temp
        temp = cv2.imread(file, 0)
        #print(file)
        temp2 = (kp.signals.EBSD(temp)).isig[45:195, 45:195]
        temp3 = temp2.remove_dynamic_background(
            operation="subtract",  # Default
            filter_domain="frequency",  # Default
            std=8,  # Default is 1/8 of the pattern width
            truncate=4,  # Default
            inplace=False,
        )

        dump = img_normalization(temp3.data).flatten() #dump (90000,)
        #print(np.shape(dump))
        shape = np.shape(dump)[0] 
        input_X = np.reshape(dump, (1, shape))

        return input_X

# Input: Ca, Cb are lists of location of EBSPs
def get_components(Ca, Cb):
    """
    to get the normalized intensities of averaged EBSPs within selected points/ components
    """
    dim1 = int(np.sqrt(len(Ca)))
    dim2 = int(np.sqrt(len(Cb)))
    if dim1 != 3 or dim2 != 3:
        print("The component grid dimensions is not 3x3. Must be 3x3 ")

    # component 1
    file_list_arr = np.reshape(Ca, (dim1, dim2))  # only Ca is read
    print("The component C1 grid shape is", np.shape(file_list_arr))
    temp = []
    for i in range(0, 3):
        for j in range(0, 3):
            # print(file_list_arr[i][j])
            temp_file = cv2.imread(file_list_arr[i][j], 0) #grey value
            # print(temp_file)
            temp.append(temp_file)

    comp1 = signal_process(temp, flag="component")

    # component 2
    file_list_arr = np.reshape(Cb, (dim1, dim2))  # only Cb is read
    print("The component C2 grid shape is", np.shape(file_list_arr))
    temp = []
    for i in range(0, 3):
        for j in range(0, 3):
            temp_file = cv2.imread(file_list_arr[i][j], 0)
            temp.append(temp_file)
    comp2 = signal_process(temp, flag="component")

    components = np.stack((comp1, comp2)) #Join a sequence of arrays along a new axis

    return components


def get_eds_average(pos_X, pos_Y, edax):
    """
    get the eds average value for each element within one component/position
    """
    elements = ['oxygen', 'Mg', 'Al', 
                'Si', 'Ti', 'Mn', 'Fe']
    
    averages = []
    
    if isinstance(pos_X, tuple):
        for element in elements:
            try:
                # get the eds value
                eds_data = edax.inav[pos_X[0]:pos_X[1],pos_Y[0]:pos_Y[1]].xmap.prop[element]
                
                # calculate the average value
                if eds_data.size > 0:
                    avg = np.nanmean(eds_data)
                    averages.append(round(avg, 4))
                else:
                    averages.append(np.nan)
                    
            except KeyError:
                print(f"Warning: {element} data not found in EDS metadata!")
                averages.append(np.nan)
    else:
        for element in elements:
            eds_data = edax.inav[pos_X:(pos_X+1),pos_Y:(pos_Y+1)].xmap.prop[element]
            
            averages.append(eds_data)
        
    return averages




