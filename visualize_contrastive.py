import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE

def visualize_loss_curve(
    pca_model, 
    cnmf_model, 
    test_data,
    pca_train_losses, 
    pca_test_losses,  
    cnmf_train_losses, 
    cnmf_test_losses   
):
    """Visualization of contrastive learning"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract test set features and elements
    pca_features = test_data[[col for col in test_data.columns if 'PC_' in col]].values
    cnmf_features = test_data[[col for col in test_data.columns if 'cNMF_' in col]].values
    elements = test_data[[col for col in test_data.columns if col in ['O', 'Mg', 'Al', 'Si', 'Ti', 'Mn', 'Fe']]].values
    
    if pca_features.shape[1] != pca_model.feature_dim:
        print(f"Warning: PCA feature dimension mismatch. Expected {pca_model.feature_dim}, got {pca_features.shape[1]}")
        if pca_features.shape[1] > pca_model.feature_dim:
            pca_features = pca_features[:, :pca_model.feature_dim]
            print(f"Using first {pca_model.feature_dim} PCA features")
        else:
            raise ValueError("PCA feature dimension mismatch cannot be resolved automatically")
    if cnmf_features.shape[1] != cnmf_model.feature_dim:
        print(f"Warning: cNMF feature dimension mismatch. Expected {cnmf_model.feature_dim}, got {cnmf_features.shape[1]}")
        if cnmf_features.shape[1] > cnmf_model.feature_dim:
            cnmf_features = cnmf_features[:, :cnmf_model.feature_dim]
            print(f"Using first {cnmf_model.feature_dim} cNMF features")
        else:
            raise ValueError("cNMF feature dimension mismatch cannot be resolved automatically")
    
    if elements.shape[1] != pca_model.element_dim:
        print(f"Warning: Element dimension mismatch. Expected {pca_model.element_dim}, got {elements.shape[1]}")
        # Assume all using the same elements
        if elements.shape[1] > pca_model.element_dim:
            elements = elements[:, :pca_model.element_dim]
            print(f"Using first {pca_model.element_dim} elements")
        else:
            raise ValueError("Element dimension mismatch cannot be resolved automatically")
    # Transform to tensor
    pca_features_tensor = torch.tensor(pca_features, dtype=torch.float32).to(device)
    cnmf_features_tensor = torch.tensor(cnmf_features, dtype=torch.float32).to(device)
    elements_tensor = torch.tensor(elements, dtype=torch.float32).to(device)
    
    # Obtain the embedding representation
    pca_model.eval()
    with torch.no_grad():
        pca_proj, _ = pca_model(pca_features_tensor, elements_tensor)
        pca_emb = pca_proj.cpu().numpy()
    
    cnmf_model.eval()
    with torch.no_grad():
        cnmf_proj, _ = cnmf_model(cnmf_features_tensor, elements_tensor)
        cnmf_emb = cnmf_proj.cpu().numpy()
    
    # Loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(pca_train_losses, label='PCA Train Loss')
    plt.plot(pca_test_losses, label='PCA Test Loss', linestyle='--')
    plt.plot(cnmf_train_losses, label='cNMF Train Loss')
    plt.plot(cnmf_test_losses, label='cNMF Test Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Contrastive Learning Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_comparison.png', dpi=300)
    plt.show()
    
def visualize_contrastive_embeddings(
    model, 
    test_data,
    model_name,
    element_names=['O', 'Mg', 'Al', 'Si', 'Ti', 'Mn', 'Fe']
):
    """Visualize the embedding space for features and elements"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == 'pca':
        feature_cols = [col for col in test_data.columns if 'PC_' in col]
    elif model_name =='cnmf':
        feature_cols =[col for col in test_data.columns if 'cNMF_' in col]
    features = test_data[feature_cols].values
    elements = test_data[element_names].values
    
    
    # Transform to Tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    elements_tensor = torch.tensor(elements, dtype=torch.float32).to(device)
    
    # Obtain the embedding representation
    model.eval()
    with torch.no_grad():
        feature_proj, element_proj = model(features_tensor, elements_tensor)
        feature_emb = feature_proj.cpu().numpy()
        element_emb = element_proj.cpu().numpy()
    
    # Combat to tsne
    combined_emb = np.vstack((feature_emb, element_emb))
    
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(combined_emb)
    
    
    feature_tsne = tsne_results[:len(features)]
    element_tsne = tsne_results[len(features):]
    
    # Color for each sample
    num_samples = len(features)
    colors = plt.cm.hsv(np.linspace(0, 1, num_samples))
    

    plt.figure(figsize=(12, 8))
    

    plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], 
                c=colors, marker='o', s=50, alpha=0.7,
                label='Feature Embeddings')
    

    plt.scatter(element_tsne[:, 0], element_tsne[:, 1], 
                c=colors, marker='x', s=70, alpha=0.9,
                label='Element Embeddings')

    for i in range(num_samples):
        plt.plot([feature_tsne[i, 0], element_tsne[i, 0]], 
                [feature_tsne[i, 1], element_tsne[i, 1]], 
                color=colors[i], alpha=0.2, linestyle='--')

    plt.legend()
    

    plt.title(f'{model_name} - Feature and Element Embeddings (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    

    plt.grid(True, alpha=0.3)
    

    plt.tight_layout()
    # plt.savefig(f'{model_name.lower()}_contrastive_embedding_space.png', dpi=300)
    plt.show()
    

    distances = np.linalg.norm(feature_tsne - element_tsne, axis=1)
    avg_distance = np.mean(distances)
    print(f"Average distance between feature and element embeddings: {avg_distance:.4f}")
    
    return feature_tsne, element_tsne, distances


def visualize_contrastive_embeddings_by_phase(
    model, 
    test_data,
    loc_data,
    coor_dict,
    phase_map,
    model_name,
    element_names=['O', 'Mg', 'Al', 'Si', 'Ti', 'Mn', 'Fe'],
    perplexity=30
):
    """Visualize embeddings colored by material phase"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == 'pca':
        feature_cols = [col for col in test_data.columns if 'PC_' in col]
    elif model_name == 'cnmf':
        feature_cols = [col for col in test_data.columns if 'cNMF_' in col]
    
    features = test_data[feature_cols].values
    elements = test_data[element_names].values
    
    # Transform to Tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    elements_tensor = torch.tensor(elements, dtype=torch.float32).to(device)
    
    # Obtain the embedding representation
    model.eval()
    with torch.no_grad():
        feature_proj, element_proj = model(features_tensor, elements_tensor)
        feature_emb = feature_proj.cpu().numpy()
        element_emb = element_proj.cpu().numpy()
    
    # Get phase information for each sample
    phase_ids = []
    phase_names = []
    
    # Iterate through each coordinate tuple
    for coord_tuple in loc_data:
        # coord_tuple tuple (x,y)
        phase_id = coor_dict.get(coord_tuple, -1)  # -1 for unknown phase
        phase_ids.append(phase_id)
        phase_names.append(phase_map.get(phase_id, "Unknown"))
    
    
    # Convert to numpy arrays
    phase_ids = np.array(phase_ids)
    phase_names = np.array(phase_names)
    
    # Get unique phases and assign colors
    unique_phases = np.unique(phase_names)
    phase_to_color = {phase: plt.cm.tab10(i) for i, phase in enumerate(unique_phases)}
    
    # Map each sample to its phase color
    phase_colors = [phase_to_color[name] for name in phase_names]
    
    # t-SNE transformation
    combined_emb = np.vstack((feature_emb, element_emb))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(combined_emb)
    
    feature_tsne = tsne_results[:len(features)]
    element_tsne = tsne_results[len(features):]
    
    plt.figure(figsize=(12, 8))
    
    # Create legend handles
    legend_handles = []
    
    # Plot feature embeddings (circles)
    for phase in unique_phases:
        mask = (phase_names == phase)
        if np.any(mask):
            sc = plt.scatter(feature_tsne[mask, 0], feature_tsne[mask, 1], 
                            c=[phase_to_color[phase]], marker='o', s=50, alpha=0.7,
                            label=f'{phase} (Feature)')
            legend_handles.append(sc)
    
    # Plot element embeddings (crosses)
    for phase in unique_phases:
        mask = (phase_names == phase)
        if np.any(mask):
            sc = plt.scatter(element_tsne[mask, 0], element_tsne[mask, 1], 
                            c=[phase_to_color[phase]], marker='x', s=70, alpha=0.9,
                            label=f'{phase} (Element)')
            # Don't add to legend again to avoid duplication
    
    # Connect same sample points
    for i in range(len(features)):
        plt.plot([feature_tsne[i, 0], element_tsne[i, 0]], 
                 [feature_tsne[i, 1], element_tsne[i, 1]], 
                 color=phase_colors[i], alpha=0.2, linestyle='--')
    
    # Create custom legend
    phase_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=phase_to_color[phase], 
                    markersize=10, label=phase) for phase in unique_phases]
    
    marker_legend = [
        plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='gray', markersize=10, label='Feature Embedding'),
        plt.Line2D([0], [0], marker='x', color='k', markersize=10, label='Element Embedding')
    ]
    
    # Add both legends
    first_legend = plt.legend(handles=phase_legend, title="Phases", loc='upper right')
    plt.gca().add_artist(first_legend)
    plt.legend(handles=marker_legend, loc='lower right')
    
    plt.title(f'{model_name} - Feature and Element Embeddings by Phase (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_embedding_by_phase.png', dpi=300)
    plt.show()
    
    # Calculate average distance
    distances = np.linalg.norm(feature_tsne - element_tsne, axis=1)
    avg_distance = np.mean(distances)
    print(f"Average distance between feature and element embeddings: {avg_distance:.4f}")
    
    # Calculate distance by phase
    print("\nDistance by phase:")
    for phase in unique_phases:
        mask = (phase_names == phase)
        if np.any(mask):
            phase_dist = np.mean(distances[mask])
            print(f"{phase}: {phase_dist:.4f}")
    
    return feature_tsne, element_tsne, distances

def visualize_contrastive_embeddings_by_element(
    model, 
    test_data,
    model_name,
    element_to_plot='Fe',
    element_names=['O', 'Mg', 'Al', 'Si', 'Ti', 'Mn', 'Fe'],
    perplexity=30
):
    """Visualize embeddings colored by element content"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == 'pca':
        feature_cols = [col for col in test_data.columns if 'PC_' in col]
    elif model_name == 'cnmf':
        feature_cols = [col for col in test_data.columns if 'cNMF_' in col]
    
    features = test_data[feature_cols].values
    elements = test_data[element_names].values
    
    # Get the index of the element to plot
    try:
        element_idx = element_names.index(element_to_plot)
        
    except ValueError:
        print(f"Element '{element_to_plot}' not found in element names. Using 'Fe' instead.")
        element_idx = element_names.index('Fe') if 'Fe' in element_names else 0
    
    element_values = elements[:, element_idx]
    
    # Create a color map for the element values
    norm = plt.Normalize(vmin=np.min(element_values), vmax=np.max(element_values))
    cmap = plt.cm.viridis
    colors = cmap(norm(element_values))
    
    # Transform to Tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    elements_tensor = torch.tensor(elements, dtype=torch.float32).to(device)
    
    # Obtain the embedding representation
    model.eval()
    with torch.no_grad():
        feature_proj, element_proj = model(features_tensor, elements_tensor)
        feature_emb = feature_proj.cpu().numpy()
        element_emb = element_proj.cpu().numpy()
    
    # t-SNE transformation
    combined_emb = np.vstack((feature_emb, element_emb))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(combined_emb)
    
    feature_tsne = tsne_results[:len(features)]
    element_tsne = tsne_results[len(features):]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot feature embeddings (circles)
    scatter1 = plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], 
                c=colors, marker='o', s=50, alpha=0.7,
                label='Feature Embeddings')
    
    # Plot element embeddings (crosses)
    scatter2 = plt.scatter(element_tsne[:, 0], element_tsne[:, 1], 
                c=colors, marker='x', s=70, alpha=0.9,
                label='Element Embeddings')
    
    # Connect same sample points
    for i in range(len(features)):
        plt.plot([feature_tsne[i, 0], element_tsne[i, 0]], 
                [feature_tsne[i, 1], element_tsne[i, 1]], 
                color=colors[i], alpha=0.2, linestyle='--')
    
    # Add colorbar
    cbar = fig.colorbar(scatter1, ax=ax)
    cbar.set_label(f'{element_to_plot} Content', rotation=270, labelpad=20)
    ax.legend()
    ax.set_title(f'{model_name} - Feature and Element Embeddings by {element_to_plot} Content (t-SNE)')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_embedding_by_{element_to_plot.lower()}.png', dpi=300)
    plt.show()
    
    # Calculate average distance
    distances = np.linalg.norm(feature_tsne - element_tsne, axis=1)
    avg_distance = np.mean(distances)
    print(f"Average distance between feature and element embeddings: {avg_distance:.4f}")
    
    # Calculate correlation between distance and element content
    corr = np.corrcoef(distances, element_values)[0, 1]
    print(f"Correlation between distance and {element_to_plot} content: {corr:.4f}")
    
    return feature_tsne, element_tsne, distances