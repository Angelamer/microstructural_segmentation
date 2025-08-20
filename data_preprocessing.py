#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing function which uses Kikuchipy to process individual EBSPs

Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""
import numpy as np
import kikuchipy as kp
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import gmean
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd


# ---- columns ----
ELEM_COLS = ['O', 'Mg', 'Al', 'Si', 'Ti', 'Mn', 'Fe']

def get_eds_average(pos_X, pos_Y, edax, type= 'component', read_mode ='vertical'):
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
        x0, x1 = pos_X
        y0, y1 = pos_Y
        width  = x1 - x0
        height = y1 - y0
        total_pixels = width * height
        roi_data = np.full((total_pixels, len(elements)), np.nan, dtype=float)

        # choose flatten order:
        #   'C'  -> last axis (y) varies fastest  -> (0,0),(0,1),(0,2),... (vertical reading) ✅
        #   'F'  -> first axis (x) varies fastest -> (0,0),(1,0),(2,0),... (horizontal reading)
        flatten_order = 'C' if read_mode == 'vertical' else 'F'

        for col_idx, element in enumerate(elements):
            try:
                eds_1d = edax.inav[x0:x1, y0:y1].xmap.prop[element]
                # Ensure it's a NumPy array
                eds_1d = np.asarray(eds_1d)
                if eds_1d.size != total_pixels:
                    raise ValueError(f"Unexpected size for element {element}: {eds_1d.size}, expected {total_pixels}")
                # reshape 成 (width, height)，这里 axis=0 是 x，axis=1 是 y
                eds_2d = eds_1d.reshape((width, height), order='C')

                # ----------- 控制展平方向 -----------
                # vertical: y 先变 → flatten(order='F')
                # horizontal: x 先变 → flatten(order='C')
                eds_flat = eds_2d.flatten(order='F')   # 纵向读取 (0,0),(0,1),(0,2)...

                roi_data[:, col_idx] = eds_flat
            except KeyError:
                print(f"Warning: {element} not found in EDS metadata.")
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


def get_feature_cols(df, kind):
    if kind == 'pca':
        return [c for c in df.columns if c.startswith('PC_')]
    elif kind == 'cnmf':
        return [c for c in df.columns if c.startswith('cNMF_')]
    else:
        raise ValueError("kind must be 'pca' or 'cnmf'")

# ---------- compositional transform: CLR ----------
def clr_transform(arr, eps=1e-9):
    """arr: (N,K) >=0, automatically close to sum=1, then do CLR"""
    arr = np.asarray(arr, dtype=np.float64)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    comp = arr / row_sum
    comp = comp + eps
    gm = np.exp(np.mean(np.log(comp), axis=1, keepdims=True))
    clr = np.log(comp) - np.log(gm)
    return clr.astype(np.float32)

# ---------- Fitting preprocessor (on training set only) ----------
def fit_preprocessors(
    train_df: pd.DataFrame,
    pca_scale: str = 'standard',     # 'standard' | 'minmax' | 'robust' | 'none'
    cnmf_transform: str = 'clr',     # 'clr' | 'zscore' | 'none'
    elem_transform: str = 'raw'      # 'raw' | 'zscore' | 'clr' | 'log1p_zscore'
):
    transformers = {}

    # PCA
    pca_cols = get_feature_cols(train_df, 'pca')
    if pca_cols:
        if pca_scale == 'standard':
            scaler = StandardScaler().fit(train_df[pca_cols].values)
            transformers['pca'] = ('standard', pca_cols, scaler)
        elif pca_scale == 'minmax':
            scaler = MinMaxScaler().fit(train_df[pca_cols].values)
            transformers['pca'] = ('minmax', pca_cols, scaler)
        elif pca_scale == 'robust':
            scaler = RobustScaler().fit(train_df[pca_cols].values)
            transformers['pca'] = ('robust', pca_cols, scaler)
        elif pca_scale == 'none':
            transformers['pca'] = ('none', pca_cols, None)

    # cNMF
    cnmf_cols = get_feature_cols(train_df, 'cnmf')
    if cnmf_cols:
        if cnmf_transform == 'zscore':
            scaler = StandardScaler().fit(train_df[cnmf_cols].values)
            transformers['cnmf'] = ('zscore', cnmf_cols, scaler)
        elif cnmf_transform == 'clr':
            # CLR 无需拟合，但记录列名
            transformers['cnmf'] = ('clr', cnmf_cols, None)
        elif cnmf_transform == 'none':
            transformers['cnmf'] = ('none', cnmf_cols, None)

    # elements
    have_elem = all(col in train_df.columns for col in ELEM_COLS)
    if have_elem:
        if elem_transform == 'zscore':
            scaler = StandardScaler().fit(train_df[ELEM_COLS].values)
            transformers['elem'] = ('zscore', ELEM_COLS, scaler)
        elif elem_transform == 'clr':
            transformers['elem'] = ('clr', ELEM_COLS, None)
        elif elem_transform == 'log1p_zscore':
            # First log1p, then zscore (fit with train)
            tmp = np.log1p(train_df[ELEM_COLS].values)
            scaler = StandardScaler().fit(tmp)
            transformers['elem'] = ('log1p_zscore', ELEM_COLS, scaler)
        elif elem_transform == 'raw':
            transformers['elem'] = ('raw', ELEM_COLS, None)

    return transformers

# ---------- transform any df using the fitted preprocessor ----------
def apply_preprocessors(df: pd.DataFrame, transformers: dict) -> pd.DataFrame:
    out = df.copy()

    # PCA
    if 'pca' in transformers:
        mode, cols, obj = transformers['pca']
        X = out[cols].values.astype(np.float32)
        if mode in ('standard', 'minmax', 'robust'):
            X = obj.transform(X)
        # 'none' 
        out[cols] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # cNMF
    if 'cnmf' in transformers:
        mode, cols, obj = transformers['cnmf']
        X = out[cols].values
        if mode == 'zscore':
            X = obj.transform(X)
        elif mode == 'clr':
            X = clr_transform(X)
        # 'none' 
        out[cols] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # elements
    if 'elem' in transformers:
        mode, cols, obj = transformers['elem']
        X = out[cols].values
        if mode == 'raw':
            pass
        elif mode == 'zscore':
            X = obj.transform(X)
        elif mode == 'clr':
            X = clr_transform(X)
        elif mode == 'log1p_zscore':
            X = np.log1p(X)
            X = obj.transform(X)
        out[cols] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return out