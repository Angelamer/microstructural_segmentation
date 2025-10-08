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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from utils import _safe_savefig

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

def plot_phase_heatmap(
    coor_dict,                         # {(x,y): phase_id}
    boundary_loc_label_dict=None,      # {(x,y): cluster_id or label_id}
    anomalies_loc_label_dict=None,     # {(x,y): cluster_id or label_id}
    coor_phase_map=None,               # {phase_id: phase_name}
    boundary_loc_label_map=None,       # {label_id: phase_name} (optional)
    anomalies_loc_label_map=None,      # {label_id: phase_name} (optional)
    image_size=(31, 31),
    roi_xrange=None,                   # (xmin, xmax) half-open [xmin, xmax)
    roi_yrange=None,                   # (ymin, ymax) half-open [ymin, ymax)
    cluster_name_map=None,              # {label_id: phase_name} — if given, overrides boundary_loc_label_map
    save_dir=None, filename=None, dpi=300, show=False
):
    """
    Draw a phase heatmap from (x,y)->phase_id and overlay boundary/anomaly boxes.
    IMPORTANT: overlays compare by *phase name* — overlay boxes are colored using the
    same color as that phase name (if present in the phase map); otherwise a fallback.
    """

    n_rows, n_cols = image_size

    # ---- 1) ROI filter (half-open ranges) ----
    def in_roi(x, y):
        okx = True if roi_xrange is None else (roi_xrange[0] <= x < roi_xrange[1])
        oky = True if roi_yrange is None else (roi_yrange[0] <= y < roi_yrange[1])
        return okx and oky

    coor_dict = { (x, y): pid for (x, y), pid in coor_dict.items() if in_roi(x, y) }
    if boundary_loc_label_dict is not None:
        boundary_loc_label_dict = { (x, y): lab for (x, y), lab in boundary_loc_label_dict.items() if in_roi(x, y) }
    if anomalies_loc_label_dict is not None:
        anomalies_loc_label_dict = { (x, y): lab for (x, y), lab in anomalies_loc_label_dict.items() if in_roi(x, y) }

    if not coor_dict and not boundary_loc_label_dict and not anomalies_loc_label_dict:
        print("[warn] No points inside ROI to plot.")
        return

    # ---- 2) Coordinate → grid mapping ----
    # Collect all coordinates (including boundary and anomaly points)
    all_coords = list(coor_dict.keys())
    if boundary_loc_label_dict:
        all_coords.extend(boundary_loc_label_dict.keys())
    if anomalies_loc_label_dict:
        all_coords.extend(anomalies_loc_label_dict.keys())
    
    all_xy = np.array(all_coords, dtype=float)
    min_x, min_y = np.min(all_xy, axis=0)
    max_x, max_y = np.max(all_xy, axis=0)
    sx = (n_cols - 1) / (max_x - min_x) if max_x > min_x else 1.0
    sy = (n_rows - 1) / (max_y - min_y) if max_y > min_y else 1.0

    def to_grid(x, y):
        gx = int(round((x - min_x) * sx))
        gy = int(round((y - min_y) * sy))
        gx = max(0, min(n_cols - 1, gx))
        gy = max(0, min(n_rows - 1, gy))
        return gx, gy  # (col, row)

    # ---- 3) Phase names present in data ----
    def pid_to_name(pid):
        if coor_phase_map is not None:
            return coor_phase_map.get(pid, f"Phase {pid}")
        return f"Phase {pid}"

    phase_ids_present = sorted(set(coor_dict.values())) if coor_dict else []
    phase_names_present = [pid_to_name(pid) for pid in phase_ids_present]

    # Also include any names appearing in boundary/anomaly label maps (to color match by name)
    def _collect_label_names(label_dict, label_map):
        names = set()
        if label_dict is not None:
            for lab in set(label_dict.values()):
                if label_map is not None:
                    names.add(label_map.get(lab, str(lab)))
                else:
                    names.add(str(lab))
        return names

    # Allow cluster_name_map to override boundary map (per your note)
    boundary_name_map = cluster_name_map if cluster_name_map is not None else boundary_loc_label_map
    anomaly_name_map  = anomalies_loc_label_map

    extra_names = set()
    extra_names |= _collect_label_names(boundary_loc_label_dict, boundary_name_map)
    extra_names |= _collect_label_names(anomalies_loc_label_dict, anomaly_name_map)

    # Build master list of phase names used for the colormap
    all_phase_names = list(phase_names_present)  # keep order of phases in coor_dict first
    for nm in sorted(extra_names):
        if nm not in all_phase_names:
            all_phase_names.append(nm)

    # ---- 4) Colors (match 'not_indexed' / -1 to white if present) ----
    base_cmap = plt.cm.get_cmap('tab20', max(1, len(all_phase_names)))
    color_list = [to_hex(base_cmap(i)) for i in range(len(all_phase_names))]

    # Force not_indexed to white if present by name (or by pid=-1 already included)
    try:
        idx_not = all_phase_names.index("not_indexed")
        color_list[idx_not] = "#FFFFFF"
    except ValueError:
        pass  # not present

    # Slight transparency for non-white
    rgba_colors = []
    for c in color_list:
        if c.upper() == "#FFFFFF":
            rgba_colors.append(to_rgba(c, 1.0))
        else:
            rgba_colors.append(to_rgba(c, 0.25))
    cmap = ListedColormap(rgba_colors)
    name_to_idx = {name: i for i, name in enumerate(all_phase_names)}
    name_to_facecolor = {name: color_list[i] for i, name in enumerate(all_phase_names)}

    # ---- 5) Build the phase index grid ----
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)  # NaN=transparent
    coord_to_grid = {}
    for (x, y), pid in coor_dict.items():
        pname = pid_to_name(pid)
        pidx  = name_to_idx[pname]
        gx, gy = to_grid(x, y)
        grid[gy, gx] = float(pidx)
        coord_to_grid[(x, y)] = (gx, gy)

    cmap = cmap.copy()
    cmap.set_bad((0, 0, 0, 0))  # transparent for empty grid cells

    # ---- 6) Plot base heatmap ----
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, cmap=cmap, origin='upper', vmin=0, vmax=len(all_phase_names)-1)
    # ax.set_title("Phase Map with Boundary/Anomaly Overlays (compare by phase name)")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- 7) Overlay helpers (compare by phase name) ----
    def _draw_boxes(label_dict, label_map, hatch, lw=2, alpha=1.0, zorder=3):
        """
        label_dict: {(x,y): label_id}
        label_map: {label_id: phase_name}  (or None -> str(label_id))
        We color the box edge using the SAME color as that phase name if the name exists
        in all_phase_names; otherwise we fall back to black.
        """
        if not label_dict:
            return
        for (x, y), lab in label_dict.items():
            # Calculate grid coordinates for this point
            gx, gy = to_grid(x, y)
            
            # map label_id -> name
            if label_map is not None:
                lname = label_map.get(lab, str(lab))
            else:
                lname = str(lab)
            edge_col = name_to_facecolor.get(lname, 'black')  # match by phase name
            rect = Rectangle(
                (gx - 0.5, gy - 0.5), 1, 1,
                linewidth=lw, edgecolor=edge_col, facecolor='none',
                hatch=hatch, alpha=alpha, linestyle='-', zorder=zorder
            )
            ax.add_patch(rect)

    # Overlap handling (draw both hatches on the same cell)
    b_set = set(boundary_loc_label_dict) if boundary_loc_label_dict else set()
    a_set = set(anomalies_loc_label_dict) if anomalies_loc_label_dict else set()
    overlap = b_set & a_set

    if boundary_loc_label_dict:
        only_b = {k: v for k, v in boundary_loc_label_dict.items() if k not in overlap}
        _draw_boxes(only_b, boundary_name_map, hatch='////', lw=2, alpha=1.0, zorder=4)

    if anomalies_loc_label_dict:
        only_a = {k: v for k, v in anomalies_loc_label_dict.items() if k not in overlap}
        _draw_boxes(only_a, anomalies_loc_label_map, hatch='xxx', lw=2, alpha=1.0, zorder=5)

    for k in overlap:
        _draw_boxes({k: boundary_loc_label_dict[k]}, boundary_name_map,       hatch='////', lw=2, alpha=1.0, zorder=6)
        _draw_boxes({k: anomalies_loc_label_dict[k]}, anomalies_loc_label_map, hatch='xxx',  lw=2, alpha=1.0, zorder=7)

    # ---- 8) Legend (phase colors + hatch meaning) ----
    legend_elems = []
    for name in all_phase_names:
        legend_elems.append(
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=name_to_facecolor[name], markersize=12, label=name)
        )
    legend_elems.append(Line2D([0], [0], marker='s', color='k', markerfacecolor='none',
                               markersize=12, label='Boundary (////)', linewidth=2))
    legend_elems.append(Line2D([0], [0], marker='s', color='k', markerfacecolor='none',
                               markersize=12, label='Anomaly (xxx)', linewidth=2))
    if overlap:
        legend_elems.append(Line2D([0], [0], marker='s', color='k', markerfacecolor='none',
                                   markersize=12, label='Overlap (//// + xxx)', linewidth=2))

    # de-duplicate legend entries
    seen, uniq = set(), []
    for h in legend_elems:
        if h.get_label() not in seen:
            uniq.append(h); seen.add(h.get_label())
    ax.legend(handles=uniq, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

    fig.tight_layout()

    saved = _safe_savefig(fig, save_dir, filename, dpi)
    if show: plt.show()
    else: plt.close(fig)
    return saved
    

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
    
    
    
def plot_kam_with_learning_boundary(
    df,
    gb_col: str,                                   # e.g. "GB_0.1" / "GB_2" / "GB_5" / ...
    roi_xrange: tuple,                              # (xmin, xmax) in same units as df[x_col]
    roi_yrange: tuple,                              # (ymin, ymax)
    boundary_loc_label_dict=None,                   # {(x,y): label_idx, ...}
    anomalies_loc_label_dict=None,                  # {(x,y): label_idx, ...}
    boundary_loc_label_map=None,                    # {label_idx: "name"} (optional)
    anomalies_loc_label_map=None,                   # {label_idx: "name"} (optional)
    figsize=(12, 9),
    kam_col="KAM",
    x_col="x",
    y_col="y",
    cmap_colors=("#fff5f5","#ff9b9b","#b21a1a"),    # low → mid → high (compatible with your palette)
    kam_vmax_percentile=95,                         # robust upper color cap
    contour_kwargs=None,                             # e.g. dict(colors="black", linewidths=1.0)
    save_dir=None, filename=None, dpi=300, show=False
):
    """
    Plot a KAM heatmap for a *region of interest* and overlay:
      • one GB column (boolean) as a contour,
      • boundary markers,
      • anomaly markers.

    Notes
    -----
    - Uses the same color solution as `plot_kam_with_overlays`:
      KAM warm colormap, and GB/boundary colors/linestyles.
    - ROI is specified in the coordinate units of df[x_col]/df[y_col].
    - Works for integer-indexed grids or irregular (float) grids by mapping
      unique values inside the ROI to a tight grid.
    """
    # ---------- styles synced with plot_kam_with_overlays ----------
    default_styles = {
        "phase_boundary":       dict(color="#111111", linestyle="-",  linewidth=1.2, alpha=0.12),
        "points_on_the_border": dict(color="#6e6e6e", linestyle="--", linewidth=0.9, alpha=0.10),

        "GB_0.1":               dict(color="#17becf", linestyle=":",  linewidth=1.0, alpha=0.10),  # teal dotted
        "GB_0.2":               dict(color="#8c564b", linestyle="-.", linewidth=1.0, alpha=0.10),  # brown dash-dot
        "GB_0.5":               dict(color="#e377c2", linestyle="--", linewidth=1.0, alpha=0.10),  # pink dashed

        "GB_1":                 dict(color="#1f77b4", linestyle="-",  linewidth=1.0, alpha=0.10),  # blue solid
        "GB_2":                 dict(color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.10),  # green dashed
        "GB_5":                 dict(color="#d62728", linestyle="-.", linewidth=1.2, alpha=0.10),  # red dash-dot
        "GB_10":                dict(color="#9467bd", linestyle=":",  linewidth=1.2, alpha=0.10),  # purple dotted
    }
    # If user gave contour kwargs, merge with the style for this GB if available
    if contour_kwargs is None:
        contour_kwargs = {}
    gb_style = default_styles.get(gb_col, {})
    # Fill defaults if not specified by caller
    contour_kwargs = {
        "colors": gb_style.get("color", contour_kwargs.get("colors", "black")),
        "linestyles": gb_style.get("linestyle", contour_kwargs.get("linestyles", "-")),
        "linewidths": gb_style.get("linewidth", contour_kwargs.get("linewidths", 1.0)),
        **{k: v for k, v in contour_kwargs.items() if k not in {"colors","linestyles","linewidths"}}
    }

    # ---------- 1) Filter DF to ROI ----------
    xmin, xmax = roi_xrange
    ymin, ymax = roi_yrange
    mask_roi = (
        (df[x_col] >= xmin) & (df[x_col] < xmax) &
        (df[y_col] >= ymin) & (df[y_col] < ymax)
    )
    dfr = df.loc[mask_roi].copy()
    if dfr.empty:
        print("[warn] No points fall inside the specified ROI.")
        return

    if kam_col not in dfr.columns:
        raise ValueError(f"'{kam_col}' column not found in DataFrame.")
    if gb_col not in dfr.columns:
        raise ValueError(f"'{gb_col}' column not found in DataFrame.")

    # ---------- 2) Pull arrays (ROI only) ----------
    x = dfr[x_col].to_numpy()
    y = dfr[y_col].to_numpy()
    kam = dfr[kam_col].to_numpy().astype(float)
    gb_vals = dfr[gb_col].to_numpy().astype(float)

    # ---------- 3) Decide grid & mapping for ROI ----------
    def _is_intlike(arr):
        return np.allclose(arr, np.round(arr))

    if _is_intlike(x) and _is_intlike(y):
        # integer-indexed grid → tight min..max inclusive grid
        gx_min, gx_max = int(np.min(x)), int(np.max(x))
        gy_min, gy_max = int(np.min(y)), int(np.max(y))
        n_cols = gx_max - gx_min + 1
        n_rows = gy_max - gy_min + 1

        def to_grid_ix(xv, yv):
            gx = int(round(xv)) - gx_min
            gy = int(round(yv)) - gy_min
            return int(np.clip(gx, 0, n_cols - 1)), int(np.clip(gy, 0, n_rows - 1))
    else:
        # irregular coords → map via unique sorted values inside ROI
        xs = np.unique(x)
        ys = np.unique(y)
        n_cols = len(xs)
        n_rows = len(ys)
        x_to_ix = {v: i for i, v in enumerate(xs)}
        y_to_iy = {v: i for i, v in enumerate(ys)}
        def to_grid_ix(xv, yv):
            gx = x_to_ix.get(xv, int(np.argmin(np.abs(xs - xv))))
            gy = y_to_iy.get(yv, int(np.argmin(np.abs(ys - yv))))
            return int(gx), int(gy)

    # ---------- 4) Build KAM/GB grids (ROI) ----------
    kam_grid = np.full((n_rows, n_cols), np.nan, dtype=float)
    gb_grid  = np.zeros((n_rows, n_cols), dtype=float)

    for xv, yv, kv, gv in zip(x, y, kam, gb_vals):
        gx, gy = to_grid_ix(xv, yv)
        kam_grid[gy, gx] = kv
        gb_grid[gy, gx]  = gv

    # ---------- 5) KAM colormap/norm (same warmth as before) ----------
    kam_cmap = LinearSegmentedColormap.from_list(
        'kam', [
            "#fff5f5",  # very pale pink
            "#ffdada",  # soft warm pink
            "#ffbcbc",  # light rose
            "#ff9b9b",  # muted coral
            "#ff7b7b",  # medium dusty red
            "#ff5c5c",  # warm red
            "#e64545",  # bright red
            "#cc3030",  # deep red
            "#b21a1a"   # dark warm red
        ]
    )

    finite_kam = kam_grid[np.isfinite(kam_grid)]
    if finite_kam.size:
        vcenter = float(np.nanmedian(finite_kam))
        vmax = float(np.nanpercentile(finite_kam, kam_vmax_percentile))
        vmax = max(vmax, vcenter + 1e-8)
    else:
        vcenter, vmax = 0.0, 1.0
    norm = TwoSlopeNorm(vmin=0.0, vcenter=vcenter, vmax=vmax)

    # ---------- 6) Plot base KAM ----------
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        kam_grid, cmap=kam_cmap, norm=norm,
        origin="upper", interpolation="nearest"
    )
    # ax.set_title(
    #     f"KAM map (ROI: {x_col}∈[{xmin},{xmax}], {y_col}∈[{ymin},{ymax}])  +  {gb_col} contour"
    # )
    ax.set_xticks([]); ax.set_yticks([])

    legend_handles = []

    # ---------- 7) GB contour (boolean → level=0.5) ----------
    if np.any(gb_grid > 0):
        cs = ax.contour(gb_grid, levels=[0.5], **contour_kwargs)
        legend_handles.append(Line2D(
            [0],[0],
            color=contour_kwargs.get("colors", "black"),
            linestyle=contour_kwargs.get("linestyles", "-"),
            linewidth=contour_kwargs.get("linewidths", 1.0),
            label=gb_col
        ))

    # ---------- helper to draw cells at ROI grid coords ----------
    def draw_cells(coords, color, lw=2, hatch=None, alpha=1.0, z=6, label=None):
        if not coords:
            return False
        any_drawn = False
        for (cx, cy) in coords:
            # keep only points inside ROI (in original coordinate units)
            if not (xmin <= cx < xmax and ymin <= cy < ymax):
                continue
            gx, gy = to_grid_ix(cx, cy)
            rect = Rectangle((gx - 0.5, gy - 0.5), 1, 1,
                             linewidth=lw, edgecolor=color, facecolor='none',
                             hatch=hatch, alpha=alpha, zorder=z)
            ax.add_patch(rect)
            any_drawn = True
        if any_drawn and label:
            legend_handles.append(Line2D([0],[0], color=color, lw=2, label=label))
        return any_drawn

    # ---------- 8) Boundary markers ----------
    if boundary_loc_label_dict:
        b_items = [(k, v) for k, v in boundary_loc_label_dict.items()
                   if xmin <= k[0] < xmax and ymin <= k[1] < ymax]
        if b_items:
            b_labels = np.array([v for _, v in b_items])
            b_uniq = np.unique(b_labels)
            b_cmap = plt.cm.get_cmap('tab20', len(b_uniq))
            for i, lab in enumerate(b_uniq):
                coords = [xy for (xy, L) in b_items if L == lab]
                color = b_cmap(i)
                lbl   = boundary_loc_label_map.get(lab, f"Boundary label {lab}") if boundary_loc_label_map else f"Boundary label {lab}"
                draw_cells(coords, color=color, lw=2, hatch='////', alpha=0.95, z=7, label=lbl)

    # ---------- 9) Anomaly markers ----------
    if anomalies_loc_label_dict:
        a_items = [(k, v) for k, v in anomalies_loc_label_dict.items()
                   if xmin <= k[0] < xmax and ymin <= k[1] < ymax]
        if a_items:
            a_labels = np.array([v for _, v in a_items])
            a_uniq = np.unique(a_labels)
            a_cmap = plt.cm.get_cmap('tab10', len(a_uniq))
            for i, lab in enumerate(a_uniq):
                coords = [xy for (xy, L) in a_items if L == lab]
                color = a_cmap(i)
                lbl   = anomalies_loc_label_map.get(lab, f"Anomaly label {lab}") if anomalies_loc_label_map else f"Anomaly label {lab}"
                draw_cells(coords, color=color, lw=2, hatch='xxx', alpha=0.95, z=8, label=lbl)

    # ---------- 10) Colorbar + legend ----------
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('KAM (degrees)')
    if legend_handles:
        # de-duplicate legend labels
        uniq = {}
        for h in legend_handles:
            uniq[h.get_label()] = h
        ax.legend(handles=list(uniq.values()), loc="upper left", bbox_to_anchor=(-0.05, 1.15),fontsize=9, frameon=True)

    fig.tight_layout()
    saved = _safe_savefig(fig, save_dir, filename, dpi)
    if show: plt.show()
    else: plt.close(fig)
    return saved