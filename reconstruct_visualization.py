import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from constrainedmf.nmf.models import NMF
import numpy as np
import torch
import os
import matplotlib.patches as mpatches

@torch.no_grad()
def get_latent_features(model, dataloader, device, phase_dict=None, max_points=None, return_logvar=False):
    """
    Collect latent means (mu) from model.encode() along with (x,y) and optional phase ids.

    Args
    ----
    model : VAE-like model with .encode() -> (mu, logvar)
    dataloader : yields ((x_batch, y_batch), image_batch) OR (coords, image_batch)
    device : torch.device
    phase_dict : dict[(x,y) -> phase_id] or None
    max_points : int or None, stop after collecting this many samples
    return_logvar : bool, also return logvar if True

    Returns
    -------
    latents : (N, D) float32
    labels  : (N,) int or None  (phase ids from phase_dict, -1 if missing)
    xs, ys  : lists of ints (length N)
    (optional) logvars : (N, D) float32, if return_logvar=True
    """
    model.eval()

    latents = []
    logvars = [] if return_logvar else None
    labels  = [] if phase_dict is not None else None
    xs_all, ys_all = [], []

    seen = 0
    for batch in dataloader:
        # Unpack: dataset yields ((x_batch, y_batch), image_batch)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            (xb, yb), data = batch
        else:
            # Fallback: try to infer
            raise RuntimeError("Expected dataloader to yield ((x_batch,y_batch), image_batch)")

        # Convert coords to CPU numpy lists
        if torch.is_tensor(xb): x_list = xb.cpu().tolist()
        else:                   x_list = list(xb)
        if torch.is_tensor(yb): y_list = yb.cpu().tolist()
        else:                   y_list = list(yb)

        data = data.to(device, dtype=torch.float32, non_blocking=True)

        # IMPORTANT: call encode() (not forward) so we only get mu, logvar
        mu, log_var = model.encode(data)
        mu_np = mu.detach().cpu().numpy().astype(np.float32)
        latents.append(mu_np)

        if return_logvar:
            logvars.append(log_var.detach().cpu().numpy().astype(np.float32))

        # optional phase ids
        if phase_dict is not None:
            batch_labels = [phase_dict.get((int(x), int(y)), -1) for x, y in zip(x_list, y_list)]
            labels.extend(batch_labels)

        xs_all.extend([int(x) for x in x_list])
        ys_all.extend([int(y) for y in y_list])

        seen += len(x_list)
        if max_points is not None and seen >= max_points:
            # trim last chunk to exact max_points if needed
            extra = seen - max_points
            if extra > 0:
                latents[-1] = latents[-1][:-extra]
                xs_all = xs_all[:-extra]
                ys_all = ys_all[:-extra]
                if phase_dict is not None:
                    labels  = labels[:-extra]
                if return_logvar:
                    logvars[-1] = logvars[-1][:-extra]
            break

    if len(latents) == 0:
        raise RuntimeError("No latents collected. Check dataloader / model.encode().")

    latents = np.concatenate(latents, axis=0)
    labels  = np.array(labels) if labels is not None else None
    if return_logvar:
        logvars = np.concatenate(logvars, axis=0)

    print(f"Collected {latents.shape[0]} latent vectors (mu), dim={latents.shape[1]}")
    if return_logvar:
        return latents, labels, xs_all, ys_all, logvars
    return latents, labels, xs_all, ys_all

# --- Image Reconstruction Visualization ---
def reconstruct_and_visualize(model, device, dataloader, num_images=5):
    """
    select several original images from dataloader,
    and display them with the reconstructed ones
    """
    model.eval() # Set model to evaluation mode
    data_iter = iter(dataloader)
    coords, data = next(data_iter)
    data = data.to(device)
    
    n = min(num_images, data.shape[0]) # Show up to n_samples or batch size
    with torch.no_grad():
        recon_images, _, _ = model(data)
        
    # move data and reconstructions to CPU for plotting
    data = data.cpu()
    recon_images = recon_images.cpu()

    # rescale images from [-1, 1] to [0, 1] for display
    data_01 = (data + 1.0) / 2.0
    recon_images_01 = (recon_images + 1.0) / 2.
    
    # Select the coordinates of dataloader
    x_indices = coords[0][:6].numpy().astype(int)
    y_indices = coords[1][:6].numpy().astype(int)
    
    
    # plotting：The first row: original; the second row: reconstructed images第一行原图，第二行重建图
    fig, axes = plt.subplots(2, num_images, figsize=(n*2, 4))
    fig.suptitle("Original vs. Reconstructed Images", fontsize=16)
    for i in range(n):
        # Original
        ax = axes[0,i]
        ax.imshow(data_01[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f"Original Figure\n({x_indices[i]},{y_indices[i]})\n",fontsize=9)
        ax.axis('off')
        # Reconstructed
        ax = axes[1,i]
        ax.imshow(recon_images_01[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f"Reconstructed Figure\n({x_indices[i]},{y_indices[i]})\n",fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Latent Space Visualization ---
def latent_space_visualize(model, dataloader, device, phase_dict, method='tsne', max_points=10, components_coords=None):
    """Generates latent vectors (mu) and visualizes them using t-SNE."""
    latents, all_labels, all_x_indices, all_y_indices = get_latent_features(model, dataloader, device, phase_dict, max_points)

    if latents.shape[0] < 2:
        print("Not enough points to perform t-SNE.")
        return

    # Apply t-SNE/PCA for dimensionality reduction
    n_components = 2
    if method == 'tsne':
        print("Applying t-SNE... (This might take a while for many points)")
        # Adjust perplexity based on number of samples
        perplexity_value = min(30, max(5, latents.shape[0] - 1))
        # tsne
        tsne = TSNE(n_components=n_components, perplexity=perplexity_value, n_iter=350, random_state=42, init='pca', learning_rate='auto')
        mu_2d = tsne.fit_transform(latents)
        print(f"t-SNE finished. Reduced shape: {mu_2d.shape}") # Shape: (num_points, 2)
            
    elif method == 'pca':
        if latents.shape[1] < n_components:
            print(f"Warning: Latent dimension ({latents.shape[1]}) is less than requested components ({n_components}). Plotting original dimensions.")
        else:
            pca = PCA(n_components=n_components)
            mu_2d = pca.fit_transform(latents)
        print(f"PCA finished. Reduced shape: {mu_2d.shape}") # Shape: (num_points, 2)
    elif method == 'cnmf':
        # at least two constraints
        if not components_coords or len(components_coords) < 2:
            raise ValueError("At least two component coordinates required for cNMF")
        components = []
        for coord in components_coords:
            # Find matching indices
            idx = np.where(
                (np.array(all_x_indices) == coord[0]) & 
                (np.array(all_y_indices) == coord[1])
            )[0]
                
            if len(idx) == 0:
                raise ValueError(f"Coordinate {coord} not found in collected data")
            components.append(latents[idx[0], :])
                
        # NMF input X
        Input_X = torch.tensor(latents, dtype=torch.float32)
        nmf = NMF(
            Input_X.shape,
            n_components=len(components),
            initial_components=[torch.tensor(c[None, :], dtype=torch.float32) for c in components],
            fix_components=[True]*len(components),
        )
        nmf.fit(Input_X, beta=2)
            
        # extract weights
        weights = nmf.W.detach().numpy() # transpose to (samples, constraints_number)
            
        mu_2d = weights[:, :2]
            
            
    # Plotting the 2D latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=all_labels, cmap='Set1', s=10, alpha=0.7) # s=size, alpha=transparency
    plt.title(f'Latent Space Visualization ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
        
    name_map = {
    1: 'Fe3O4',
    2: 'FeO',
    3: 'Fe'
    }
    unique_ids = sorted(set(all_labels.tolist()))
        # pi and phase name mapping
    handles = []
    for pid in unique_ids:
        if pid in name_map:
            color = scatter.cmap(scatter.norm(pid))
            patch = mpatches.Patch(color=color, label=name_map[pid])
            handles.append(patch)
    plt.legend(handles=handles, title='Phase')
    plt.show()
    
def visualize_latent_maps(
    latents, xs, ys,
    latent_idx=None,                # int | list[int] | None (None=all dims)
    cmap='viridis',
    suptitle='Latent Feature Maps',
    save_idx=None,                  # int | list[int] | list of ints | None
    save_dir='.',
    save_prefix='latent_dim',
    origin='upper',                 # 'upper' (image-like) or 'lower' (math axes)
    roi_xrange=None,                # (xmin, xmax) inclusive on integer x, or None
    roi_yrange=None                 # (ymin, ymax) inclusive on integer y, or None
):
    """
    Build 2D maps for each latent dimension using provided (x,y) integer coordinates.
    Works for any x/y offset; the grid is defined by unique xs/ys after ROI filtering.

    Args
    ----
    latents : (N, D) array
    xs, ys  : length-N integer arrays/lists (coordinates)
    latent_idx : which latent dims to plot (None => all)
    origin : 'upper' puts top-left at (min_x, min_y) like images; 'lower' puts it bottom-left.
    roi_xrange, roi_yrange : optional inclusive ROI filters on x and y

    Returns
    -------
    (fig, axes), maps_dict, axes_info
      maps_dict : {dim: 2D array of shape (len(uniq_y), len(uniq_x))}
      axes_info : {"uniq_x": uniq_x, "uniq_y": uniq_y}
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    latents = np.asarray(latents)
    xs = np.asarray(xs, dtype=int)
    ys = np.asarray(ys, dtype=int)
    N, D = latents.shape
    assert xs.shape[0] == N and ys.shape[0] == N, "xs/ys must match latents N"

    # ----- pick latent dims -----
    if latent_idx is None:
        dims = list(range(D))
    elif isinstance(latent_idx, int):
        dims = [latent_idx]
    else:
        dims = list(latent_idx)

    # ----- ROI filtering (inclusive) -----
    mask = np.ones(N, dtype=bool)
    if roi_xrange is not None:
        xmin, xmax = roi_xrange
        mask &= (xs >= xmin) & (xs < xmax)
    if roi_yrange is not None:
        ymin, ymax = roi_yrange
        mask &= (ys >= ymin) & (ys < ymax)

    if not np.any(mask):
        raise ValueError("No points remain after ROI filtering.")

    xs_f = xs[mask]
    ys_f = ys[mask]
    lat_f = latents[mask, :]

    # ----- build grid axes from unique coords -----
    uniq_x = np.unique(xs_f)
    uniq_y = np.unique(ys_f)
    nx, ny = len(uniq_x), len(uniq_y)
    if nx == 0 or ny == 0:
        raise ValueError("Empty grid after ROI filtering.")

    x_to_ix = {x: i for i, x in enumerate(uniq_x)}
    y_to_iy = {y: i for i, y in enumerate(uniq_y)}

    # ----- accumulate (avg if duplicates exist) -----
    counts = np.zeros((ny, nx), dtype=np.int32)
    maps = {d: np.zeros((ny, nx), dtype=np.float32) for d in dims}

    for k in range(xs_f.shape[0]):
        ix = x_to_ix[xs_f[k]]
        iy = y_to_iy[ys_f[k]]
        counts[iy, ix] += 1
        for d in dims:
            maps[d][iy, ix] += float(lat_f[k, d])

    # average
    safe_counts = counts.copy()
    safe_counts[safe_counts == 0] = 1
    for d in dims:
        maps[d] = maps[d] / safe_counts

    # ----- plot -----
    n = len(dims)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.2, n_rows*3.0))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # image extent so ticks show real coords; +1 so each integer cell spans 1 unit
    extent = [uniq_x.min(), uniq_x.max() + 1, uniq_y.min(), uniq_y.max() + 1]

    for ax_i, d in enumerate(dims):
        ax = axes[ax_i]
        im = ax.imshow(maps[d], cmap=cmap, origin=origin, extent=extent, aspect='auto')
        ax.set_title(f'Latent dim #{d}', fontsize=10)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # save single images for selected dims
        if save_idx is not None:
            save_list = [save_idx] if isinstance(save_idx, int) else list(save_idx)
            if d in save_list:
                os.makedirs(save_dir, exist_ok=True)
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                im2 = ax2.imshow(maps[d], cmap=cmap, origin=origin, extent=extent, aspect='auto')
                ax2.set_title(f'Latent dim #{d}', fontsize=12)
                ax2.set_xlabel('x'); ax2.set_ylabel('y')
                fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                fname = f"{save_prefix}_{d}.png"
                fig2.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
                plt.close(fig2)

    # hide unused axes
    for j in range(len(dims), len(axes)):
        axes[j].axis('off')

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    axes_info = {"uniq_x": uniq_x, "uniq_y": uniq_y}
    return (fig, axes), maps, axes_info
#def latent_space_visualize_cnmf(model, dataloader, device, phase_dict, constraints_mu, max_points=100):
    
    """
    Generates latent vectors (cNMF weights W) and visualizes them.
    It processes VAE's mu with constrained_nmf function
    using constraints_mu and visualizes the resulting NMF weights (W).

    Args:
        model: The VAE model with an `encode` method.
        dataloader: DataLoader for input data.
        device: Torch device ('cuda' or 'cpu').
        phase_dict: Dictionary mapping (x,y) coordinates to phase IDs.
        max_points (int): Maximum number of data points to process and display.
    """
#   model.eval()  # Set model to evaluation mode
    

#   collected_features_for_plotting = [] # Will store VAE's mu or cNMF's W
#   all_labels = []
    
#   print(f"Method: {method.upper()}. Generating features for visualization (up to {max_points} points)...")
    