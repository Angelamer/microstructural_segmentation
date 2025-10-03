# vae_representation

## Files Covered

- `analyze_bandcontrast.py`
- `cluster_latent_bc.py`
- `data_prepare.py`
- `latent_map_selection.py`
- `reconstruct_visualization.py`
- `vae_cnn.py`
- `vae_training.py`
- `visualize_vae_latents.py`

---

## Module Summaries

### 1) `analyze_bandcontrast.py`
**Role**: Reading and analyzing EBSD **BandContrast (BC)** maps. Typical workflow:
- Load `(x, y, bandcontrast)` from CSV.
- Optional ROI selection or thresholding to define **valid pixels** and export **kept coordinates** for downstream tasks.
- Produce **histograms** / **gray-scale or heatmap** plots of BC, and export filtered coordinate lists used for training/inference alignment.

**I/O (typical)**
- **Inputs** — `*_bandcontrast.csv` with columns `x,y,bandcontrast`.
- **Saves** — ROI masks / kept coordinates (CSV/NPY) and BC visualization PNGs.

---

### 2) `cluster_latent_bc.py`
**Role**: **Joint clustering** over **latent** values and **BandContrast** with **coordinated visualizations**.
- Supports **GMM**, **KMeans**, **DBSCAN**, **HDBSCAN** for 2D clustering on `(latent, bandcontrast)`.
- Can select an **ROI** via `roi_xrange` / `roi_yrange` from a **full-field** `coords_csv`.
- **Left panel**: spatial latent map where **each cluster is a fixed color**, and **brightness encodes the magnitude of latent within that cluster** (dark→light).
- **Right panel**: scatter of `(latent, bandcontrast)` using **exactly the same cluster colors** (noise = gray).  
- Exports **cluster assignments** and a **JSON report** for reproducibility.

**I/O (typical)**
- **Inputs** — `coords_csv` (full field; must contain `x,y`, may contain `bandcontrast`), ROI ranges, `latent_roi` (NPY/CSV single column matching the ROI length), optional `bandcontrast` vector (from the CSV ROI or separate file).
- **Saves** — see the “Saved outputs” section below.

---

### 3) `data_prepare.py`
**Role**: Dataset and pre-processing **hub**.
- `KikuchiH5Dataset` to read Kikuchi patterns from **HDF5** with normalization/cropping.
- `FilterByCoordsDataset` to **restrict** samples to a coordinate set (e.g., BC-filtered pixels or an ROI).
- CSV utilities for building coordinate-phase mappings and generating `keep_xy` lists from BC, etc.

**I/O (typical)**
- **Inputs** — H5 path(s), keep-xy lists, element/BC CSVs.
- **Saves** — none directly; provides **DataLoader-ready datasets** and index/alignment helpers.

---

### 4) `latent_map_selection.py`
**Role**: **Associations** between VAE latents and multiple references (PCA/cNMF, element maps, BandContrast) with **ranking** and **visualization**.
- **PCA/cNMF vs. latents**: GMM with **auto-`k`** via BIC/AIC; compare clusters in ROI using **ARI**/**NMI** to find top-N latent dims that best match reference partitions.
- **Elements vs. latents**: build **synthetic element maps** (normalized per-channel, weighted sum *or* RGB) and compute multiple metrics (**Pearson/Spearman/Kendall/MI/DistanceCorr/HSIC/partial-Pearson removing spatial trends**); rank top-N and save plots.
- **BandContrast vs. latents**: same multi-metric ranking; save top-N figures and reports.
- **Full-field latent maps**: reassemble ROI-only latents back to the **full coordinate grid** with NaNs elsewhere; support NaN rendering (e.g., black/white).

**I/O (typical)**
- **Inputs** — latents (full-field order with NaNs for untrained), full `x,y`, ROI spec, PCA/cNMF/element/BC data.
- **Saves** — many **PNGs**, **NPYs** (ROI-aligned latent vectors), **CSVs** (score tables), and **JSON** reports (top-N rankings, parameters).

---

### 5) `reconstruct_visualization.py`
**Role**: **Inference & visualization** helpers for a trained VAE.
- Run forward passes, **collect latents** (per-coordinate) into dict or full-field arrays.
- Plot latent heatmaps (choose dims, set colormap), and optionally produce **reconstruction vs. input** comparisons.
- Utilities for **reading coordinates** and **grid assembly** from irregular points.

**I/O (typical)**
- **Inputs** — VAE model, DataLoader, coordinate lists.
- **Saves** — latent maps, reconstruction previews (PNGs) and intermediate arrays for downstream steps.

---

### 6) `vae_cnn.py`
**Role**: CNN **architecture** for the VAE.
- Convolutional **encoder** → latent **mean/logvar** → **reparameterization**.
- Convolutional **decoder** → reconstruction (often **Tanh** output).
- Forward returns reconstruction and terms needed for the loss (recon+KL).

**I/O (typical)**
- **Inputs** — `torch` tensors.
- **Saves** — none directly; consumed by training/inference scripts.

---

### 7) `vae_training.py`
**Role**: **End-to-end training** entry point.
- Builds dataset(s) from `data_prepare.py` (with optional **BC/ROI filtering**), wraps in `DataLoader`.
- Instantiates `vae_cnn.py` with hyperparameters (latent dim, LR, batch size, device).
- Training loop: forward → compute recon/regularization losses → backprop/optim.
- Saves **weights** and optionally **loss curves** / sample **reconstructions** for quick sanity checks.

**I/O (typical)**
- **Inputs** — H5 data, ROI or keep-xy coordinates from BC/ROI selection, VAE hyperparameters.
- **Saves** — see the “Saved outputs” section below.

---

### 8) `visualize_vae_latents.py`
**Role**: **Latent-space visualization** and derived analyses (post-training).
- Load the trained weights; iterate your dataset to **collect full-field latents** (fill NaNs for missing/untrained pixels).
- Batch-export **latent maps** (per dimension) to a target folder.
- Optionally trigger parts of `latent_map_selection.py` for correlation/ranking.

**I/O (typical)**
- **Inputs** — trained `.pth`, dataset (often BC-filtered ROI), full coordinates.
- **Saves** — latent maps (`latent_maps_*` folders), latent arrays (`.npy`), and optional CSV/JSON reports.

---

## What gets **saved to disk** (the three main scripts)

### A) `vae_training.py`
- **Model weights** — e.g., `vae_model_*.pth` (there is also a sample `vae_model_leaky_tanh.pth` in the repo for inference).
- **Training logs/curves** *(if enabled)* — `loss.csv`/`loss.json` and/or `loss_curve.png`.
- **Reconstruction previews** *(optional)* — side-by-side input vs. reconstruction snapshots for a few batches.
- **Config snapshot** *(optional)* — `config.json` with hyperparameters, ROI, normalization, etc., to ensure reproducibility.

### B) `visualize_vae_latents.py`
- **Latent heatmaps** — a folder per experiment/ROI/latent-dim choice, e.g.:
  - `latent_maps_filtered_selected_dim_512_x_20_90_y_190_260/`
  - `latent_maps_selected_dim_256/`
- **Latent arrays** — `latents_full.npy` (full-field order, NaN for missing) and/or `latents_roi.npy`.
- **Coordinate CSVs** — `coords.csv` or `all_coords.csv` for alignment.
- **(If coupled with `latent_map_selection.py`)** CSV/JSON score tables and top-N rankings alongside corresponding PNGs.

### C) `cluster_latent_bc.py`
- **Main visualization** — `cluster_latent_bc__{tag}.png`:
  - Left: **spatial latent map** colored by **cluster** (each cluster has a fixed hue) with **within-cluster brightness** proportional to **latent magnitude**.
  - Right: **(latent, BandContrast)** scatter using the **same colors**. DBSCAN/HDBSCAN noise (`-1`) is gray.
- **Cluster assignments** — `cluster_assignments__{tag}.csv` with columns `x, y, latent, bandcontrast, cluster`.
- **Run report** — `cluster_report__{tag}.json` capturing method, parameters (`k`, `eps`, `min_samples`, `min_cluster_size`, etc.), counts, and the output paths.
- **ROI handling** — if `coords_csv` is **full-field**, you can pass `roi_xrange=(xmin,xmax)` and `roi_yrange=(ymin,ymax)`; the script will slice the ROI rows, extract the corresponding **BandContrast** for the ROI, and verify that `latent_roi` is **exactly** the same length.

> Practical tip: Always ensure `(latent_roi, bandcontrast_roi)` are **one-to-one and in the same order**. If `bandcontrast` is supplied as a full-length vector, it must be **masked** by the same ROI before clustering.

---

## How these modules fit together

1. **Pre-processing**  
	   `analyze_bandcontrast.py` & `data_prepare.py` → define **valid pixels/ROI** and build **datasets**.

2. **Train**  
	   `vae_training.py` → train VAE on the selected pixels; save **`*.pth`** and optional logs/figures.

3. **Infer & Map**  
	   `visualize_vae_latents.py` + `reconstruct_visualization.py` → collect latents and export **latent maps**/**arrays**.

4. **Analyze & Compare**  
	   `latent_map_selection.py` → correlations/rankings vs PCA/cNMF/elements/BC with **multi-metric** reports.  
	   `cluster_latent_bc.py` → **clustering** of `(latent, BC)` with synchronized **spatial + scatter** plots and CSV/JSON artifacts.

---

