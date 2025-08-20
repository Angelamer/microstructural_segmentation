import os
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional UMAP
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

torch.manual_seed(42)
np.random.seed(42)

class ProjectionHead(nn.Module):
    """Project Head for projecting multiple modes to the same space"""
    def __init__(self, input_dim, output_dim=64, hidden=256, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, output_dim),
        )
    
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=1)

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim, element_dim, latent_dim=16, projection_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.element_dim = element_dim
        # Reduced features for kikuchi patterns (pca scores, cnmf weights, latent representation...)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        # element (chemical) encoder
        self.element_encoder = nn.Sequential(
            nn.Linear(element_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        # Shared projection
        self.feature_head = ProjectionHead(latent_dim, projection_dim, hidden=256, dropout=0.1)
        self.element_head = ProjectionHead(latent_dim, projection_dim, hidden=256, dropout=0.1)

        # init (helps stability)
        self.apply(self._init)
        
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, features, elements):
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {features.shape[1]}")
        if elements.shape[1] != self.element_dim:
            raise ValueError(f"Expected {self.element_dim} elements, got {elements.shape[1]}")
        feature_emb = self.feature_encoder(features)
        
        element_emb = self.element_encoder(elements)
        
        feature_proj = self.feature_head(feature_emb)
        element_proj = self.element_head(element_emb)
        # Normalize embeddings
        return feature_proj, element_proj
    

    
class ClipLikeLoss(nn.Module):
    """
    Bidirectional InfoNCE/CLIP loss.
    If learnable_temp=True, temperature is learnable (logit_scale).
    Otherwise supports set_temperature(T) to fix/anneal temperature.
    """
    def __init__(self, temperature=0.07, learnable_temp=True, min_temp=0.01, max_temp=1.0):
        super().__init__()
        self.learnable_temp = learnable_temp
        if learnable_temp:
            # Initialize logit_scale with log(1/temperature)
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
            self.min_temp = min_temp
            self.max_temp = max_temp
        else:
            self.register_buffer("_fixed_scale", torch.tensor(1.0 / float(temperature)))
        self.ce = nn.CrossEntropyLoss()
        
    def get_temperature(self):
        """Get the current temperature value"""
        if self.learnable_temp:
            scale = self.logit_scale.exp().clamp(min=1.0/self.max_temp, max=1.0/self.min_temp)
            return 1.0 / scale.item()
        else:
            return 1.0 / self._fixed_scale.item() 
           
    def set_temperature(self, T: float):
        """Manually set temperature (useful for annealing)"""
        T = float(max(1e-6, T))
        if self.learnable_temp:
            # Overwrite learnable scale for annealing
            with torch.no_grad():
                self.logit_scale.copy_(torch.log(torch.tensor(1.0 / T)))
        else:
            self._fixed_scale = torch.tensor(1.0 / T, device=self._fixed_scale.device)
                
    def forward(self, z_k, z_e):
        # z_k, z_e already normalized
        B = z_k.size(0)
        if self.learnable_temp:
            scale = self.logit_scale.exp().clamp(min=1.0/self.max_temp, max=1.0/self.min_temp)
        else:
            scale = self._fixed_scale

        logits_k2e = scale * z_k @ z_e.t()      # (B,B)
        logits_e2k = scale * z_e @ z_k.t()
        labels = torch.arange(B, device=z_k.device)

        loss = (self.ce(logits_k2e, labels) + self.ce(logits_e2k, labels)) / 2
        return loss

class FeatureElementDataset(Dataset):
        def __init__(self, data, feature_type):
            self.data = data
            self.feature_type = feature_type
            
            if feature_type == 'pca':
                self.feature_cols = [col for col in data.columns if 'PC_' in col]
            else:  # 'cnmf'
                self.feature_cols = [col for col in data.columns if 'cNMF_' in col]
                
            self.element_cols = [col for col in data.columns if col in ['O', 'Mg', 'Al', 'Si', 'Ti', 'Mn', 'Fe']]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            features = torch.tensor(self.data.iloc[idx][self.feature_cols].values, dtype=torch.float32)
            elements = torch.tensor(self.data.iloc[idx][self.element_cols].values, dtype=torch.float32)
            return features, elements
        

def train_and_evaluate(feature_type, train_data, test_data, output_dim =2,latent_dim=16, num_epochs=100, batch_size=32, lr=1e-3, 
                    weight_decay=1e-4,
                    temperature=0.07,
                    learnable_temp=True,
                    temp_anneal=True,
                    temp_init=0.1,
                    temp_floor=0.01,
                    temp_decay=0.95,
                    temp_step=10,
                    grad_clip_norm=None,
                    min_temp=0.01, max_temp=2.0
                    
                    ):
    """
    Train with CLIP-like loss. Optionally anneal temperature if learnable_temp=False.

    Args:
        feature_type: 'pca' or 'cnmf'
        train_data: training data DataFrame
        test_data: test data DataFrame
        num_epochs: number of training epochs
        batch_size: batch size
        lr: learning rate

    Returns:
        model: trained model
        train_losses: training loss history
        test_losses: test loss history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Create the dataset and loader
    train_dataset = FeatureElementDataset(train_data, feature_type)
    test_dataset = FeatureElementDataset(test_data, feature_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and loss
    feature_dim = len(train_dataset.feature_cols)
    element_dim = len(train_dataset.element_cols)
    model = ContrastiveModel(feature_dim, element_dim, latent_dim=latent_dim,projection_dim=output_dim).to(device)
    criterion = ClipLikeLoss(temperature=temperature, learnable_temp=learnable_temp,min_temp=min_temp, max_temp=max_temp)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': [criterion.logit_scale]} if learnable_temp else []
    ], lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))  # For mixed precision training
    

    train_losses = []
    test_losses = []
    
    final_alignment, final_uniformity = float('nan'), float('nan')
    for epoch in range(num_epochs):
        # Temperature display
        if learnable_temp:
            temp_display = criterion.get_temperature()
        else:
            # Handle non-learnable temperature with annealing if needed
            if temp_anneal:
                current_temp = max(temp_floor, temp_init * (temp_decay ** (epoch // temp_step)))
                criterion.set_temperature(current_temp)
                temp_display = current_temp
            else:
                temp_display = temperature
        # Training
        model.train()
        epoch_train_loss = 0.0
        for features, elements in train_loader:
            features = features.to(device)
            elements = elements.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision context
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                feature_proj, element_proj = model(features, elements)
                loss = criterion(feature_proj, element_proj)
            
            # Backpropagation
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Test
        model.eval()
        epoch_test_loss = 0.0
        zs_f, zs_e = [], []
        
        with torch.no_grad():
            for features, elements in test_loader:
                features = features.to(device)
                elements = elements.to(device)
                
                # Forward
                feature_proj, element_proj = model(features, elements)
                
                # Calculate the loss
                loss = criterion(feature_proj, element_proj)
                epoch_test_loss += loss.item()
                zs_f.append(feature_proj)
                zs_e.append(element_proj)
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        # Analyze alignment every 10 epochs
        do_log_metrics = ((epoch % 10) == 0) or (epoch == num_epochs - 1)
        if do_log_metrics:
            with torch.no_grad():
                k_proj = torch.cat(zs_f, dim=0)   # features embeddings
                e_proj = torch.cat(zs_e, dim=0)   # elements embeddings

                
                alignment = compute_alignment(k_proj, e_proj)
                all_proj = torch.cat([k_proj, e_proj], dim=0)
                uniformity = compute_uniformity(all_proj, t=2.0)

                # Save the last epoch
                final_alignment = alignment
                final_uniformity = uniformity

                rn_f = k_proj.norm(dim=1).mean().item()
                rn_e = e_proj.norm(dim=1).mean().item()
                
                
                
            print(f"Epoch {epoch+1}: Alignment={alignment:.4f}, Uniformity={uniformity:.4f} | "
                f"||zf||={rn_f:.3f} ||ze||={rn_e:.3f}")

        # ---- epoch summary ----
        print(f"Epoch [{epoch+1}/{num_epochs}], {feature_type.upper()} - "
            f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
            f"Temp: {temp_display:.4f}")
        
    
    return model, train_losses, test_losses, final_alignment, final_uniformity


@torch.no_grad()
def get_embeddings(model: nn.Module, features_np: np.ndarray, elements_np: np.ndarray, device=None):
    """Run model forward and return (feature_emb, element_emb) as numpy arrays. Outputs are L2-normalized."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    f = torch.tensor(features_np, dtype=torch.float32, device=device)
    e = torch.tensor(elements_np, dtype=torch.float32, device=device)
    zf, ze = model(f, e)
    return zf.cpu().numpy(), ze.cpu().numpy()



def cosine_sim_matrix(zf: np.ndarray, ze: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (N,N). Assumes rows are L2-normalized."""
    return zf @ ze.T


def retrieval_metrics(sim: np.ndarray, topk=(1, 5, 10)):
    """Cross-modal retrieval accuracy: for each feature row, rank element cols by similarity."""
    N = sim.shape[0]
    ranks = np.argsort(-sim, axis=1)
    gt = np.arange(N)
    acc = {k: float(np.mean([gt[i] in ranks[i, :k] for i in range(N)])) for k in topk}
    true_ranks = np.argmax(ranks == gt[:, None], axis=1) + 1
    return acc, float(np.median(true_ranks))

def logit_gap(sim: np.ndarray):
    """Diagonal (true pair) vs. hardest negative gap per row."""
    sim = np.asarray(sim)
    diag = np.diag(sim)
    tmp = sim.copy()
    np.fill_diagonal(tmp, -np.inf)
    hardest = tmp.max(axis=1)
    gap = diag - hardest
    return gap, float(np.mean(gap)), float(np.median(gap))


def paired_cosine_distance(zf: np.ndarray, ze: np.ndarray):
    """Paired cosine distance (1 - cosine). Lower is better alignment."""
    cos = np.sum(zf * ze, axis=1)
    return 1.0 - cos


def debug_embedding_collapse(zf, ze, name=""):
    def stats(z, tag):
        z = np.asarray(z)
        print(f"[{name}] {tag}: shape={z.shape}, "
            f"row-norm(mean±std)={np.mean(np.linalg.norm(z,axis=1)):.4f}±{np.std(np.linalg.norm(z,axis=1)):.4f}, "
            f"per-dim std(mean)={np.mean(np.std(z,axis=0)):.6f}, "
            f"pairwise-dist(mean)={np.mean(np.linalg.norm(z[:,None,:]-z[None,:,:],axis=2)):.6f}")
        # Check if all the same
        uniq = np.unique(np.round(z, decimals=4), axis=0).shape[0]
        print(f"[{name}] {tag}: approx unique rows (4dp) = {uniq}")
    stats(zf, "features zf")
    stats(ze, "elements ze")

# -------------------------
# Plots
# -------------------------
def plot_similarity_heatmap(S: np.ndarray, phase_names=None, title="Cosine similarity (features vs elements)"):
    """Heatmap of cross-modal similarity. If phase_names provided, sort by phase for block patterns."""
    order = np.arange(S.shape[0])
    if phase_names is not None:
        order = np.argsort(np.array(phase_names, dtype=str))
    S_sorted = S[order][:, order]
    plt.figure(figsize=(7, 6))
    im = plt.imshow(S_sorted, vmin=-1, vmax=1, cmap='coolwarm')
    plt.title(title)
    plt.xlabel("elements"); plt.ylabel("features")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()


def plot_pair_distance_hist(zf: np.ndarray, ze: np.ndarray, bins=40):
    """Histogram of paired cosine distances (embedding space, not t-SNE)."""
    d = paired_cosine_distance(zf, ze)
    plt.figure(figsize=(5, 3))
    plt.hist(d, bins=bins, alpha=0.85)
    plt.xlabel("Paired cosine distance (1 - cosine)")
    plt.ylabel("Count")
    plt.title("Paired distance distribution")
    plt.tight_layout(); plt.show()


def joint_2d_pca_plot(zf: np.ndarray, ze: np.ndarray, max_lines=200, alpha=0.6):
    """Stable 2D PCA joint plot of feature/element embeddings with optional pair lines."""
    Z = np.vstack([zf, ze])
    Zs = StandardScaler(with_mean=True, with_std=True).fit_transform(Z)
    xy = PCA(n_components=2, random_state=42).fit_transform(Zs)
    N = zf.shape[0]
    a, b = xy[:N], xy[N:]
    plt.figure(figsize=(7, 6))
    plt.scatter(a[:, 0], a[:, 1], s=20, alpha=alpha, label="features")
    plt.scatter(b[:, 0], b[:, 1], s=20, alpha=alpha, marker='x', label="elements")
    idx = np.arange(N)
    if N > max_lines:
        idx = np.random.choice(N, max_lines, replace=False)
    for i in idx:
        plt.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color='k', alpha=0.08, lw=0.8)
    plt.legend(); plt.title("Joint 2D PCA of embeddings")
    plt.tight_layout(); plt.show()


def joint_umap_plot(zf: np.ndarray, ze: np.ndarray, n_neighbors=15, min_dist=0.1, max_lines=200):
    """UMAP joint plot (if umap is available). Pre-scales with StandardScaler."""
    if not _HAS_UMAP:
        print("[warn] umap is not installed; skip joint_umap_plot.")
        return
    Z = np.vstack([zf, ze])
    Zs = StandardScaler().fit_transform(Z)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    xy = reducer.fit_transform(Zs)
    N = zf.shape[0]
    a, b = xy[:N], xy[N:]
    plt.figure(figsize=(7, 6))
    plt.scatter(a[:, 0], a[:, 1], s=20, alpha=0.6, label='features')
    plt.scatter(b[:, 0], b[:, 1], s=20, alpha=0.6, marker='x', label='elements')
    idx = np.arange(N)
    if N > max_lines:
        idx = np.random.choice(N, max_lines, replace=False)
    for i in idx:
        plt.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color='k', alpha=0.06, lw=0.8)
    plt.legend(); plt.title(f"Joint UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")
    plt.tight_layout(); plt.show()


@torch.no_grad()
def compute_alignment(zf: torch.Tensor, ze: torch.Tensor) -> float:
    """
    Alignment = E[ || zf - ze ||^2 ]  (assuming zf, ze are L2-normalized)
    """
    return torch.mean(torch.sum((zf - ze) ** 2, dim=1)).item()

@torch.no_grad()
def compute_uniformity(z: torch.Tensor, t: float = 2.0, chunk: int = 4096) -> float:
    """
    Uniformity = log E_{i!=j} [ exp( -t * ||zi - zj||^2 ) ]  (Wang & Isola, 2020)
    z: (N, D) (will be L2-normalized inside)
    t: usually 2.0
    chunk: to limit memory (pairwise O(N^2))
    """
    z = F.normalize(z, dim=1)
    N = z.size(0)
    acc_sum, acc_cnt = 0.0, 0
    for i in range(0, N, chunk):
        zi = z[i:i+chunk]                 # (b, D)
        sims = zi @ z.t()                 # (b, N)
        sqdist = 2.0 - 2.0 * sims.clamp(-1, 1)   # ||zi - zj||^2 on the unit sphere
        # mask diagonal in this block
        mask = torch.ones_like(sqdist, dtype=torch.bool)
        bi = zi.size(0)
        idx = torch.arange(bi, device=z.device)
        # Note: Only the diagonal corners of the current block are masked here; the overall approximation is sufficient for diagnosis
        mask[idx, i + idx] = False
        vals = torch.exp(-t * sqdist[mask])
        acc_sum += vals.sum().item()
        acc_cnt += vals.numel()
    uniform = np.log(acc_sum / max(acc_cnt, 1))
    return float(uniform)

def evaluate_contrastive(model, features_np, elements_np, loc_testdata,coor_dict,phase_map, do_plots=True):
    """
    Compute key diagnostics:
    - cosine similarity matrix
    - retrieval accuracy (Top-1/5/10) & median rank
    - logit gap mean/median
    - paired cosine distance stats
    Optionally draw heatmap + distance histogram + PCA/UMAP plots.
    """
    if hasattr(loc_testdata, "to_numpy"):
        arr = loc_testdata.to_numpy()
    else:
        arr = np.asarray(loc_testdata, dtype=object)
    # Case 1: already (N,2)
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2:
        coords = arr
    # Case 2: (N,) of tuple-like
    elif isinstance(arr, np.ndarray) and arr.ndim == 1:
        coords = np.array([tuple(v) for v in arr], dtype=float)  # cast to float first
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Could not parse loc_testdata into (N,2); got shape {coords.shape}")
    else:
        # Fallback: treat as iterable of tuple-like
        coords = np.array([tuple(v) for v in arr], dtype=float)
    coords = coords.astype(int)  # (N,2) int
    if coords.shape[0] != features_np.shape[0]:
        raise ValueError(
            f"loc_testdata N={coords.shape[0]} does not match features N={features_np.shape[0]}"
        )

    phase_names = []
    for i in range(coords.shape[0]):
        x_i, y_i = int(coords[i, 0]), int(coords[i, 1])
        pid = coor_dict.get((x_i, y_i), -1)
        if pid in phase_map:
            pname = str(phase_map[pid])
        else:
            pname = "Unknown" if pid == -1 else f"Phase {pid}"
        phase_names.append(pname)
    phase_names = np.array(phase_names, dtype=object)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zf_n, ze_n = get_embeddings(model, features_np, elements_np, device=device)
    debug_embedding_collapse(zf_n, ze_n, name="eval")
    S = cosine_sim_matrix(zf_n, ze_n)
    acc, medrank = retrieval_metrics(S, topk=(1, 5, 10))
    gap_vec, gap_mean, gap_median = logit_gap(S)
    pair_dist = paired_cosine_distance(zf_n, ze_n)

    zf = torch.from_numpy(zf_n).float().to(device)
    ze = torch.from_numpy(ze_n).float().to(device)

    alignment = compute_alignment(zf, ze)
    all_proj = torch.cat([zf, ze], dim=0)
    uniformity = compute_uniformity(all_proj, t=2.0)

    print("[Retrieval] Top-1/5/10:", acc, "| Median rank:", medrank)
    print("[Logit gap] mean/median:", gap_mean, gap_median)
    print("[Pair cosine distance] mean/std/min/max:",
        float(pair_dist.mean()), float(pair_dist.std()),
        float(pair_dist.min()), float(pair_dist.max()))
    print(f"[W&I] Alignment={alignment:.4f}  Uniformity={uniformity:.4f}")

    if do_plots:
        plot_similarity_heatmap(S, phase_names=phase_names)
        plot_pair_distance_hist(zf_n, ze_n)
        joint_2d_pca_plot(zf_n, ze_n)
        if _HAS_UMAP:
            joint_umap_plot(zf_n, ze_n)

    return {
        "similarity": S,
        "retrieval": acc,
        "median_rank": medrank,
        "logit_gap": {"vector": gap_vec, "mean": gap_mean, "median": gap_median},
        "pair_cos_dist": pair_dist,
        "alignment": alignment,
        "uniformity": uniformity,
        "phase_names": phase_names,
    }