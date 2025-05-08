import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from constrainedmf.nmf.models import NMF
import numpy as np
import torch
import matplotlib.patches as mpatches



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
    model.eval() # Set model to evaluation mode
    latents = []
    all_labels = []
    all_x_indices = []
    all_y_indices = []
    
    print(f"Generating latent vectors (mu) for visualization (up to {max_points} points)...")

    count = 0
    with torch.no_grad():
        
        for batch_idx, (batch_indices, data) in enumerate(dataloader):
            
            x_idx_tensor, y_idx_tensor = batch_indices
            data_batch = data.to(device)
            
            # Get mu from the encoder
            mu, _ = model.encode(data_batch) # Pass through encoder only
            latents.append(mu.cpu().numpy())
            
            #get the phase id
            x_list = x_idx_tensor.tolist()
            y_list = y_idx_tensor.tolist()
            batch_labels = [
            phase_dict.get((x, y), -1)
            for x, y in zip(x_list, y_list)
            ]
            all_labels.extend(batch_labels)
            all_x_indices.extend(x_list)
            all_y_indices.extend(y_list)

            count += len(data_batch)
            if count >= max_points:
                print(f"Reached {count} points, stopping latent vector generation.")
                break

    # Concatenate all mu vectors
    latents = np.concatenate(latents, axis=0)
    all_labels = np.array(all_labels)
    print(f"Collected {latents.shape[0]} latent vectors (mu) with dimension {latents.shape[1]}")

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
    