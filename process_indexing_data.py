import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R
import kikuchipy as kp
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colorbar import Colorbar
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from orix.quaternion import Orientation
from sklearn.cross_decomposition import CCA
from scipy import stats
from matplotlib.colors import ListedColormap, to_hex, to_rgba
from matplotlib.patches import Rectangle, Patch
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.lines import Line2D



def compute_roi_misorientation_map_from_xmap(
        xmap,
        roi_xrange=(0, 10),
        roi_yrange=(0, 10),
        step_size=0.05,
        phase_ref_points=None,  # e.g., {0: (ix, iy)}, ix,iy are pixel indices
        mis_tolerance=1.0,
        default_cluster_cmap_names=None,
        show_plot=True):
    """
    Compute and plot misorientation map for a rectangular ROI from orix xmap.

    Args:
        xmap: orix EBSD xmap object
        roi_xrange: (xmin, xmax) in pixel indices (not physical positions)
        roi_yrange: (ymin, ymax) in pixel indices
        step_size: real spacing of grid, e.g. 0.05 (μm or other units)
        phase_ref_points: {phase: (ix, iy)}, where (ix, iy) are pixel indices.
        show_plot: if True, show the ROI misorientation map.

    Returns:
        result_table: array, each row: [x, y, phase_index, misorientation]
        misorientation_map: 2D array, misorientation values for ROI
        phase_map: 2D (ny, nx) phase index values
    """

    # 1. Extract x, y, phase_id, Euler for each pixel, per phase
    euler_list = []
    phase_id_list = []
    x_list = []
    y_list = []
    orientation_list = []
    symmetry_list = []  # Store symmetry for each phase

    # Create a mapping from phase_id to phase name
    phase_idx_to_name = {phase_id: phase_obj.name for phase_id, phase_obj in xmap.phases}
    phase_idx_to_obj  = {phase_id: phase_obj for phase_id, phase_obj in xmap.phases}

    phase_symmetry = {}
    for phase_id, phase_obj in xmap.phases:
        # Use point group from phase object if available
        if hasattr(phase_obj, 'point_group') and phase_obj.point_group is not None:
            phase_symmetry[phase_id] = phase_obj.point_group
        else:
            # Fallback to cubic symmetry if not specified
            from orix.quaternion.symmetry import get_point_group
            phase_symmetry[phase_id] = get_point_group(432)  # Cubic symmetry
    for phase_id, phase_obj in xmap.phases:
        mask = (xmap.phase_id == phase_id)
        # Phase-specific orientation array
        ori_this_phase = xmap[phase_obj.name].orientations
        # Assign symmetry to orientations
        if phase_id in phase_symmetry:
            sym = phase_symmetry[phase_id]
            ori_this_phase.symmetry = sym
        else:
            # Fallback to phase's point group or cubic symmetry
            if hasattr(phase_obj, 'point_group') and phase_obj.point_group is not None:
                ori_this_phase.symmetry = phase_obj.point_group
            else:
                from orix.quaternion.symmetry import get_point_group
                ori_this_phase.symmetry = get_point_group(432)  # Cubic symmetry
        # Convert to Euler angles
        if hasattr(ori_this_phase, 'to_euler'):
            euler_this_phase = ori_this_phase.to_euler('zxz').data
        else:
            euler_this_phase = np.array([o.to_euler('zxz') for o in ori_this_phase])
        # print(euler_this_phase)
        while euler_this_phase.ndim > 2:
            euler_this_phase = euler_this_phase[0]
        if euler_this_phase.ndim == 1:
            euler_this_phase = euler_this_phase[None, :]
        euler_this_phase = np.rad2deg(euler_this_phase)
        # Corresponding phase_id, x, y
        phase_id_arr = np.full(len(euler_this_phase), phase_id)
        x_arr = np.asarray(xmap[phase_obj.name].x).ravel()
        y_arr = np.asarray(xmap[phase_obj.name].y).ravel()
        # Ensure x_arr and y_arr are also shape (n,)
        if x_arr.ndim > 1:
            x_arr = x_arr.ravel()
        if y_arr.ndim > 1:
            y_arr = y_arr.ravel()
        euler_list.append(euler_this_phase)
        phase_id_list.append(phase_id_arr)
        x_list.append(x_arr)
        y_list.append(y_arr)
        orientation_list.append(ori_this_phase)
        symmetry_list.append(ori_this_phase.symmetry)
        # print(orientation_list)
        # print(np.shape(euler_list))
    # Stack everything
    all_euler = np.vstack(euler_list)        # (N, 3)
    all_phase = np.concatenate(phase_id_list)
    # print(all_phase)
    all_x = np.concatenate(x_list)
    all_y = np.concatenate(y_list)
    # quats = np.vstack([ori_obj.data for ori_obj in orientation_list])
    #sym = orientation_list[0].symmetry   # or pick appropriate phase symmetry
    #all_ori = Orientation(quats, symmetry=sym)

    # 2. ROI mask using physical positions
    xmin, xmax = roi_xrange
    ymin, ymax = roi_yrange
    x_min_real, x_max_real = xmin * step_size, xmax * step_size
    y_min_real, y_max_real = ymin * step_size, ymax * step_size

    mask_roi = (
        (all_x >= x_min_real) & (all_x < x_max_real) &
        (all_y >= y_min_real) & (all_y < y_max_real)
    )

    roi_x = all_x[mask_roi]
    roi_y = all_y[mask_roi]
    roi_phase = all_phase[mask_roi]
    roi_euler = all_euler[mask_roi]
    # roi_ori = all_ori[mask_roi]

    # Create a list of orientations for ROI points
    roi_ori = []
    for i in range(len(roi_x)):
        phase = roi_phase[i]
        # Find the original orientation index
        phase_mask = (all_phase == phase) & (all_x == roi_x[i]) & (all_y == roi_y[i])
        if np.any(phase_mask):
            idx = np.where(phase_mask)[0][0]
            # Find which phase list this belongs to
            for j, (phase_arr, ori_list) in enumerate(zip(phase_id_list, orientation_list)):
                if phase in phase_arr:
                    local_idx = np.where(phase_arr == phase)[0]
                    if idx in local_idx:
                        roi_ori.append(ori_list[local_idx.tolist().index(idx)])
                        break
            else:
                # Fallback: create a new orientation
                euler_deg = roi_euler[i]
                euler_rad = np.deg2rad(euler_deg)
                if phase in phase_symmetry:
                    sym = phase_symmetry[phase]
                else:
                    sym = symmetry_list[0]  # Fallback to first symmetry
                roi_ori.append(Orientation.from_euler(euler_rad, symmetry=sym))
        else:
            # Create new orientation if not found
            euler_rad = np.deg2rad(roi_euler[i])
            if phase in phase_symmetry:
                sym = phase_symmetry[phase]
            else:
                sym = symmetry_list[0]  # Fallback to first symmetry
            roi_ori.append(Orientation.from_euler(euler_rad, symmetry=sym))
    
    # 3. Determine grid shape of ROI
    n_x = xmax - xmin
    n_y = ymax - ymin

    # 4. For each phase in ROI, determine reference orientation
    phase_indices = np.unique(roi_phase)
    n_phases = len(phase_indices)
    if default_cluster_cmap_names is None:
        default_cluster_cmap_names = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges',
                                    'YlOrBr', 'BuGn', 'PuRd', 'Greys'][:n_phases]
    cmap_names = []
    for i in range(n_phases):
        cmap_names.append(default_cluster_cmap_names[i % len(default_cluster_cmap_names)])
    cmaps = [get_cmap(name) for name in cmap_names]
    phase_ref_ori = {}
    
    for phase in phase_indices:
        mask = (roi_phase == phase)
        phase_indices_arr = np.where(mask)[0]
        ori_this_phase = [roi_ori[i] for i in phase_indices_arr]
        
        if phase_ref_points and phase in phase_ref_points:
            # Provided reference as pixel (ix, iy) within ROI
            ref_ix, ref_iy = phase_ref_points[phase]
            ref_x = x_min_real + ref_ix * step_size
            ref_y = y_min_real + ref_iy * step_size
            dists = (roi_x[mask] - ref_x)**2 + (roi_y[mask] - ref_y)**2
            min_idx = np.argmin(dists)
            
            if ori_this_phase and min_idx < len(ori_this_phase):
                ref_ori = ori_this_phase[min_idx]
            else:
                # Fallback: use the first orientation
                ref_ori = ori_this_phase[0]
        else:
            # Use mean orientation considering symmetry
            try:
                ref_ori = Orientation.mean(ori_this_phase)
            except:
                # Fallback: use the first orientation
                ref_ori = ori_this_phase[0]
                
        phase_ref_ori[phase] = ref_ori

    # 5. Compute misorientation for each pixel in ROI considering crystal symmetry
    misorientation = np.zeros(len(roi_phase))
    
    for i in range(len(roi_phase)):
        phase = roi_phase[i]
        ref_ori = phase_ref_ori[phase]
        point_ori = roi_ori[i]
        
        # Calculate misorientation considering crystal symmetry
        misorientation[i] = point_ori.angle_with(ref_ori).data[0] * 180 / np.pi

    # 6. Map to 2D grid (pixel index, relative to ROI)
    mis_map = np.full((n_y, n_x), np.nan)
    phase_map = np.full((n_y, n_x), -1)
    
    for i in range(len(roi_x)):
        # Convert position to grid indices
        ix = int(round((roi_x[i] - x_min_real) / step_size))
        iy = int(round((roi_y[i] - y_min_real) / step_size))
        if 0 <= iy < n_y and 0 <= ix < n_x:
            mis_map[iy, ix] = misorientation[i]
            phase_map[iy, ix] = roi_phase[i]

    # 7. Visualization
    if show_plot:
        # Create figure with better layout
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0.05, 0.1, 0.6, 0.8])  # Main plot area
        
        base_img = np.zeros((n_y, n_x, 4))
        
        # Set grey for values >1° or NaN
        mask_high = (mis_map > mis_tolerance) | np.isnan(mis_map)
        base_img[mask_high] = [0.7, 0.7, 0.7, 1]  # RGBA for grey
        
        # Create composite image with per-phase coloring for values <=1°
        rgba_img = np.zeros((n_y, n_x, 4))
        phase_norms = {}
        
        for i, phase in enumerate(phase_indices):
            mask_phase = (phase_map == phase)
            mask_low = mask_phase & (mis_map <= mis_tolerance)
            
            if np.any(mask_low):
                phase_mis = mis_map[mask_low]
                vmin, vmax = 0, mis_tolerance  # Fixed range for all phases
                phase_norms[phase] = (vmin, vmax)
                normed = (phase_mis - vmin) / (vmax - vmin + 1e-8)
                cmap = cmaps[i % len(cmaps)]
                rgba_img[mask_low] = cmap(normed)[:, :4]
                rgba_img[mask_low, 3] = 1.0
        
        # Combine images: base (grey) + colored phases
        composite_img = np.where(rgba_img[..., 3:4] > 0, rgba_img, base_img)
        
        # Plot the composite image
        im = ax.imshow(composite_img, interpolation='nearest', origin='upper',
                    extent=[0, n_x, n_y, 0])
        ax.set_title("ROI Misorientation Map", fontsize=16, pad=20)
        ax.set_xlabel("X (ROI pixel index)", fontsize=12)
        ax.set_ylabel("Y (ROI pixel index)", fontsize=12)
        ax.set_xlim(0, n_x)
        ax.set_ylim(n_y, 0)
        # Add grid lines for better orientation
        ax.grid(True, color='w', linestyle=':', linewidth=0.5, alpha=0.3)
        
        # Create colorbars for each phase
        if phase_indices.size > 0:
            
            cax_width0 = 0.02
            cax_spacing0 = 0.05
            max_total_width = 0.25
            total_required = len(phase_indices) * cax_width0 + (len(phase_indices)-1) * cax_spacing0
            if total_required > max_total_width:
                scale_factor = max_total_width / total_required
                cax_width = cax_width0 * scale_factor
                cax_spacing = cax_spacing0 * scale_factor
            else:
                cax_width = cax_width0
                cax_spacing = cax_spacing0
                
            total_cbar_width = len(phase_indices) * cax_width + (len(phase_indices)-1) * cax_spacing
            start_x = 0.65 + (0.25 - total_cbar_width) / 2
            
            cax_height = 0.8
            cax_top = 0.9
            cax_bottom = cax_top - cax_height
            
            for i, phase in enumerate(phase_indices):
                
                vmin, vmax = 0, mis_tolerance
                cax = fig.add_axes([start_x + i*(cax_width + cax_spacing), 
                                    cax_bottom, 
                                    cax_width, 
                                    cax_height])
                
                cmap = cmaps[i]  # Colormap assigned to this phase
                norm = Normalize(vmin=vmin, vmax=vmax)
                sm = ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                
                cbar = Colorbar(cax, sm, orientation='vertical')
                n_ticks = 6
                ticks = np.linspace(0, mis_tolerance, n_ticks)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f'{t:.2f}°' for t in ticks])
                
                # Add phase name above each colorbar
                phase_name = phase_idx_to_name.get(int(phase), f"Phase {int(phase)}")
                cbar.ax.text(0.5, 1.05, phase_name, 
                            transform=cbar.ax.transAxes,
                            ha='center', va='bottom', fontsize=10)
                
                # Only add label to the first colorbar
                if i == 0:
                    cbar.set_label("Misorientation [deg]", fontsize=12, labelpad=10)
        
        # Add text annotation for grey areas
        fig.text(0.75, 0.05, f"Grey areas: >{mis_tolerance}° or no data", 
                fontsize=10, ha='center', va='center')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    result_table = np.column_stack([roi_x, roi_y, roi_phase, misorientation])
    return result_table, mis_map, phase_map

def normalize_misorientation_by_phase_map(mis_map, phase_map):
    """
    Normalize the orientation difference within each phase (0-1 range) for 2D grid data

    Parameters:
        mis_map: 2D array containing the orientation difference of each pixel (in degrees)
        phase_map: 2D array containing the phase ID of each pixel

    Returns:
        normalized_map: normalized orientation difference map (0-1 range)
        phase_min_max: dictionary recording the original minimum and maximum values of each phase {phase: (min, max)}
    
    """
    
    
    normalized_map = np.full_like(mis_map, np.nan, dtype=float)
    
    # Get the IDs of all phases (excluding negative values, which usually indicate background or no data)
    unique_phases = np.unique(phase_map)
    unique_phases = unique_phases[unique_phases >= 0]
    
    phase_min_max = {}
    
    # Normalize each phase separately
    for phase in unique_phases:
        # mask for phase
        mask = (phase_map == phase) & ~np.isnan(mis_map)
        
        
        if not np.any(mask):
            continue
            
        
        phase_mis = mis_map[mask]
        
        min_val = np.min(phase_mis)
        max_val = np.max(phase_mis)
        
        
        phase_min_max[phase] = (min_val, max_val)
        
        
        if max_val - min_val < 1e-8:
            
            normalized_map[mask] = 0.5
        else:
            normalized_map[mask] = (mis_map[mask] - min_val) / (max_val - min_val)
    
    return normalized_map, phase_min_max
def extract_roi_quaternions(xmap, roi_xrange, roi_yrange, step_size):
    """
    Extract the quaternions within roi from xmap
    
    Args:
        xmap: orix CrystalMap object
        roi_xrange: (xmin, xmax) 
        roi_yrange: (ymin, ymax) 
        step_size
    
    Returns:
        roi_coords:  [(x1, y1), (x2, y2), ...]
        roi_quats: (N, 4)
    """
    xmin, xmax = roi_xrange
    ymin, ymax = roi_yrange
    x_min_real = round(xmin * step_size,2)
    x_max_real = round(xmax * step_size,2)
    y_min_real = round(ymin * step_size,2)
    y_max_real = round(ymax * step_size,2)
    
    roi_coords = []
    roi_quats = []
    
    
    for phase_id, phase_obj in xmap.phases:
        
        phase_data = xmap[phase_obj.name]
        
        # Obtain the coordinates and orientations
        x_coords = phase_data.x.flatten()
        y_coords = phase_data.y.flatten()
        orientations = phase_data.orientations
        
    
        for i in range(len(x_coords)):
            x = x_coords[i]
            y = y_coords[i]
            
            
            if (x >= x_min_real and x < x_max_real and 
                y >= y_min_real and y < y_max_real):
                
                coord_key = (np.round(x, 2), np.round(y, 2))
                quat = orientations[i].data
                #print(quat)
                roi_coords.append(coord_key)
                roi_quats.append(quat)
    #print(roi_coords, roi_quats)
    return np.array(roi_coords), np.array(roi_quats)

def match_data_by_coords(roi_coords, roi_quats, variations, variation_coords, scores, step_size):
    """
    Match the quaternions, intra-cluster variations, and scores based on coordinates
    
    Args:
        roi_coords: Coordinates from ROI (N, 2)
        roi_quats: Quaternions corresponding to roi_coords (N, 4)
        variations: Variation values (M,)
        variation_coords: Coordinates for variations (M, 2)
        scores: PCA scores or other feature vectors (M, K)
    
    Returns:
        matched_quats: Matched quaternions
        matched_variations: Matched variations
        matched_scores: Matched scores
    """
    variation_coords= variation_coords * step_size
    # Create mapping from coordinates to index in variation data
    coord_to_idx = {}
    for i, coord in enumerate(variation_coords):
        # Use rounded coordinates as keys to handle floating-point precision
        key = (np.round(coord[0], 6), np.round(coord[1], 6))
        coord_to_idx[key] = i
    
    matched_quats = []
    matched_variations = []
    matched_scores = []
    matched_indices = []  # For debugging
    
    # Traverse ROI coordinates to find matching data
    for i, coord in enumerate(roi_coords):
        key = (np.round(coord[0], 6), np.round(coord[1], 6))
        if key in coord_to_idx:
            idx = coord_to_idx[key]
            matched_quats.append(roi_quats[i])
            matched_variations.append(variations[idx])
            matched_scores.append(scores[idx])
            matched_indices.append(idx)
    
    # Convert to arrays for easier handling
    matched_quats = np.array(matched_quats)
    matched_variations = np.array(matched_variations)
    matched_scores = np.array(matched_scores)
    
    # Print matching statistics
    n_matched = len(matched_quats)
    n_roi = len(roi_coords)
    match_percentage = (n_matched / n_roi) * 100 if n_roi > 0 else 0
    
    print(f"Matched {n_matched} points out of {n_roi} in ROI ({match_percentage:.2f}%)")
    
    return matched_quats, matched_variations, matched_scores

def analyze_quaternion_variations_relationship(quats, variations):
    """
    Analyze the relations between quaternions and variations
    Args:
        quats:  (N, 4)
        variations: variations (N,)
    
    Returns:
        cca_result: CCA
        correlations: Correlations between each component of quaternions and variations
    """
    if quats.ndim != 2 or quats.shape[1] != 4:
        quats = quats.reshape(-1, 4)
    # 1. Calculate the pearsonr correlation between quaternions and variations
    correlations = []
    for i in range(4):
        #print(np.shape(quats[:,i]))
        corr, p_value = stats.pearsonr(quats[:, i], variations)
        correlations.append({
            'component': i,
            'correlation': corr,
            'p_value': p_value
        })
    
    # 2. CCA
    # reshape variations into 2-D array
    X = quats  # (N, 4)
    Y = variations.reshape(-1, 1)  # (N, 1)
    
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    
    # transform X,Y
    X_c, Y_c = cca.transform(X, Y)
    
    # calculate the ccc
    canon_corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
    
    return {
        'correlations': correlations,
        'canonical_correlation': canon_corr,
        'x_weights': cca.x_weights_,
        'y_weights': cca.y_weights_,
        'transformed_x': X_c,
        'transformed_y': Y_c
    }

def analyze_quaternion_scores_relationship(quats, scores, n_components=3):
    """
    analyze the relations between quaternions and scores
    
    Args:
        quats: (N, 4)
        scores: scores (N, reduced_components)
        n_components: CCA components
    
    Returns:
        cca_result
    """
    if quats.ndim != 2 or quats.shape[1] != 4:
        quats = quats.reshape(-1, 4)
    # CCA
    cca = CCA(n_components=n_components)
    cca.fit(quats, scores)
    
    
    X_c, Y_c = cca.transform(quats, scores)
    

    canon_corrs = []
    for i in range(n_components):
        corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        canon_corrs.append(corr)
    
    return {
        'canonical_correlations': canon_corrs,
        'x_weights': cca.x_weights_,
        'y_weights': cca.y_weights_,
        'transformed_x': X_c,
        'transformed_y': Y_c
    }

def plot_phase_heatmap(coor_dict, boundary_loc_label_dict=None,
    anomalies_loc_label_dict=None,
    coor_phase_map=None,
    boundary_loc_label_map=None,
    anomalies_loc_label_map=None, image_size=(31, 31)):
    """
    Plot a phase index heatmap from coordinate-phase dictionary, and highlight boundary and anomaly points.
    
    Args:
        coor_dict (dict): mapping (x, y) -> phase_index for each of the 31x31 samples.
        boundary_loc_label_dict (dict, optional): {(x, y): label_idx, ...}
        anomalies_loc_label_dict (dict, optional): {(x, y): label_idx, ...}
        coor_phase_map (dict): {label_idx: phase_name, ...}
        boundary_loc_label_map (dict): {label_idx: phase_name, ...}
        anomalies_loc_label_map (dict): {label_idx: phase_name, ...}
        image_size (tuple): (n_rows, n_cols) for the ROI grid, default (31,31).
    """
    n_rows, n_cols = image_size

    # phase_map
    phase_map = np.full((n_rows, n_cols), fill_value=-1, dtype=int)
    all_coords = np.array(list(coor_dict.keys()))
    min_x, min_y = np.min(all_coords, axis=0)
    max_x, max_y = np.max(all_coords, axis=0)
    scale_x = (n_cols - 1) / (max_x - min_x) if max_x != min_x else 1
    scale_y = (n_rows - 1) / (max_y - min_y) if max_y != min_y else 1

    # Map coordinates to grid
    coord_to_grid = {}
    for (x, y), phase in coor_dict.items():
        grid_x = int(round((x - min_x) * scale_x))
        grid_y = int(round((y - min_y) * scale_y))
        if 0 <= grid_x < n_cols and 0 <= grid_y < n_rows:
            phase_map[grid_y, grid_x] = phase
            coord_to_grid[(x, y)] = (grid_x, grid_y)

    # All phase names
    all_phase_names = set()
    def _collect_phases(dict_, phase_map_):
        if dict_ and phase_map_:
            for idx in set(dict_.values()):
                all_phase_names.add(phase_map_.get(idx, f"Phase {idx}"))
    if coor_phase_map:
        for idx in set(coor_dict.values()):
            all_phase_names.add(coor_phase_map.get(idx, f"Phase {idx}"))
    else:
        for idx in set(coor_dict.values()):
            all_phase_names.add(f"Phase {idx}")
    _collect_phases(boundary_loc_label_dict, boundary_loc_label_map)
    _collect_phases(anomalies_loc_label_dict, anomalies_loc_label_map)
    all_phase_names = sorted(list(all_phase_names))
    n_phases = len(all_phase_names)

    # Set the color
    if n_phases == 2:
        color_list = ['#2196F3', '#E53935']  # 蓝/红
    elif n_phases == 3:
        color_list = ['#2196F3', '#E53935', '#43A047']  # 蓝/红/绿
    else:
        base_cmap = plt.cm.get_cmap('tab20', n_phases)
        color_list = [to_hex(base_cmap(i)) for i in range(n_phases)]
    phase_color_dict = {p: color_list[i % len(color_list)] for i, p in enumerate(all_phase_names)}

    # 4. phase index -> phase name -> colormap index
    def _get_phase_idx(idx):
        if coor_phase_map is not None:
            phase = coor_phase_map.get(idx, f"Phase {idx}")
        else:
            phase = f"Phase {idx}"
        return all_phase_names.index(phase)
    phase_idx_map = np.vectorize(_get_phase_idx)
    plot_map = np.full_like(phase_map, fill_value=-1)
    mask = (phase_map >= 0)
    plot_map[mask] = phase_idx_map(phase_map[mask])
    
    phase_alpha = 0.25
    phase_colors_rgba = [to_rgba(phase_color_dict[p], alpha=phase_alpha) for p in all_phase_names]
    cmap = ListedColormap(phase_colors_rgba)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(plot_map, cmap=cmap, origin='upper', vmin=0, vmax=n_phases-1)
    ax.set_title('Phase Map with Anomaly and Boundary')
    ax.set_xticks([]); ax.set_yticks([])


    def _add_boxes_label_dict(label_dict, label_map, hatch, lw=2, alpha=1.0, zorder=2):
        if not (label_dict and label_map): return
        for (x, y), label in label_dict.items():
            phase = label_map.get(label, f"Phase {label}") if label_map else f"Phase {label}"
            color = phase_color_dict.get(phase, 'black')
            grid_coord = coord_to_grid.get((x, y))
            if grid_coord is None: continue
            grid_x, grid_y = grid_coord
            rect = Rectangle(
                (grid_x - 0.5, grid_y - 0.5), 1, 1,
                linewidth=lw, edgecolor=color, facecolor='none',
                hatch=hatch, alpha=alpha, linestyle='-', zorder=zorder
            )
            ax.add_patch(rect)

    # Overlapping
    boundary_set = set(boundary_loc_label_dict) if boundary_loc_label_dict else set()
    anomaly_set = set(anomalies_loc_label_dict) if anomalies_loc_label_dict else set()
    overlap_set = boundary_set & anomaly_set

    if boundary_loc_label_dict and boundary_loc_label_map:
        only_boundary = {k: v for k, v in boundary_loc_label_dict.items() if k not in overlap_set}
        _add_boxes_label_dict(only_boundary, boundary_loc_label_map, hatch='////', lw=2, alpha=1, zorder=3)
    if anomalies_loc_label_dict and anomalies_loc_label_map:
        only_anomaly = {k: v for k, v in anomalies_loc_label_dict.items() if k not in overlap_set}
        _add_boxes_label_dict(only_anomaly, anomalies_loc_label_map, hatch='xxx', lw=2, alpha=1, zorder=4)
    
    
    for k in overlap_set:
        label_b = boundary_loc_label_dict[k]
        label_a = anomalies_loc_label_dict[k]
        _add_boxes_label_dict({k: label_b}, boundary_loc_label_map, hatch='////', lw=2, alpha=1, zorder=5)
        _add_boxes_label_dict({k: label_a}, anomalies_loc_label_map, hatch='xxx', lw=2, alpha=1, zorder=6)

    # legend：classify phase、anomaly/boundary and hatch
    legend_elements = []
    for phase in all_phase_names:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', markerfacecolor=phase_color_dict[phase],
                markersize=12, label=f"{phase}")
        )
    legend_elements.append(
        Line2D([0], [0], marker='s', color='k', markerfacecolor='none',
            markersize=12, label='Boundary (hatch=////)', linewidth=2)
    )
    legend_elements.append(
        Line2D([0], [0], marker='s', color='k', markerfacecolor='none',
            markersize=12, label='Anomaly (hatch=xxx)', linewidth=2)
    )
    if overlap_set:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='k', markerfacecolor='none',
                markersize=12, label='Overlap (//// + xxx)', linewidth=2)
        )

    seen = set()
    legend_unique = []
    for h in legend_elements:
        if h.get_label() not in seen:
            legend_unique.append(h)
            seen.add(h.get_label())
    ax.legend(handles=legend_unique, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    plt.tight_layout()
    plt.show()
    
    
    

def plot_element_vs_pca(pca_scores, element_df, loc, coord_dict, phase_map, figsize=(10, 6)):
    """
    Plot element content versus PCA scores (dual Y-axis)

    Args:
        pca_scores: Array of PCA scores (n_samples,)
        element_df: DataFrame of element content containing 'O' and 'Fe' columns
        loc: Array of coordinate tuples [(x1, y1), (x2, y2), ...]
        coord_dict: Dictionary mapping coordinates to phase IDs {(x, y): phase_id}
        phase_map: Dictionary mapping phase IDs to phase names {phase_id: phase_name}
        figsize: Figure size (default 12×8 inches)
    
    """
    # Validate input lengths
    if len(pca_scores) != len(element_df) or len(pca_scores) != len(loc):
        raise ValueError("All input arrays must have the same length")
    
    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    # Extract element content
    o_content = element_df['O'].values
    fe_content = element_df['Fe'].values
    
    # Define marker symbols for phases
    marker_symbols = ['o', 's', '^', 'D', '*', 'p', 'X', 'v', '<', '>']
    phase_markers = {}
    
    # Assign markers to unique phases
    unique_phases = set(phase_map.values())
    for i, phase in enumerate(unique_phases):
        phase_markers[phase] = marker_symbols[i % len(marker_symbols)]
    
    # Create legend handles
    element_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
            markersize=10, label='O Content'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
            markersize=10, label='Fe Content')
    ]
    
    phase_handles = []
    for phase, marker in phase_markers.items():
        phase_handles.append(
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray', 
                markersize=10, label=phase)
        )
    
    # Plot each point with phase-specific marker
    for i, coord in enumerate(loc):
        # Get phase information
        phase_id = coord_dict.get(coord)
        phase_name = phase_map.get(phase_id, "Unknown")
        marker = phase_markers.get(phase_name, 'o')
        
        # Plot O content (left axis)
        ax1.scatter(pca_scores[i], o_content[i], 
                c='blue', alpha=0.7, marker=marker,
                s=60, edgecolors='w')
        
        # Plot Fe content (right axis)
        ax2.scatter(pca_scores[i], fe_content[i], 
                c='red', alpha=0.7, marker=marker,
                s=60, edgecolors='w')
    
    # Configure axes
    ax1.set_xlabel('PC 1', fontsize=12)
    ax1.set_ylabel('O Content', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2.set_ylabel('Fe Content', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add grid and legend
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(handles=element_handles + phase_handles, loc='best', fontsize=10)
    
    # Add title and display
    plt.title('Element Content vs PC Score by Phase', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    