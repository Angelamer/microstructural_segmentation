#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing function which uses Kikuchipy to process individual EBSPs

Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""
import numpy as np
import kikuchipy as kp
import cv2
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm




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
def preprocess_features(all_data):
    """
    Preprocessing contrast learning input features

    Parameters:
        pca_scores: PCA score matrix (n_samples, n_pca_components)
        cnmf_weights: cNMF weight matrix (n_samples, n_components)
        element_content: element content matrix (n_samples, n_elements)

    Returns:
        Preprocessed tuple (pca_processed, cnmf_processed, element_processed)
    """
    processed_data = all_data.copy()
    pca_cols = [col for col in all_data.columns if col.startswith('PC_')]
    cnmf_cols = [col for col in all_data.columns if col.startswith('cNMF_')]
    element_cols = ['O', 'Mg', 'Al', 'Si', 'Ti', 'Mn', 'Fe']
    
    if pca_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        processed_data[pca_cols] = scaler.fit_transform(processed_data[pca_cols])
        
    if cnmf_cols:
        cnmf_data = all_data[cnmf_cols].values
        
        cnmf_processed = np.zeros_like(cnmf_data)
        for i in range(cnmf_data.shape[0]):
            row = cnmf_data[i, :]
            
            row = np.where(row == 0, 1e-9, row)
            geometric_mean = np.exp(np.mean(np.log(row)))
            cnmf_processed[i, :] = np.log(row / geometric_mean)
        
        processed_data[cnmf_cols] = cnmf_processed
    
    if element_cols:
        
        for col in element_cols:
            processed_data[col] = np.log1p(all_data[col])
    
    return processed_data