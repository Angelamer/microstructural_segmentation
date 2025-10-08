import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import numpy as np
import pandas as pd

from visualize_grid import read_data, set_ROI, set_component
from data_processing import get_components, coord_phase_dict_from_dataframe
from cNMF import set_global_determinism, run_cNMF, plot_weight_map_cnmf_with_anomalies, plot_weight_histograms, plot_weight_sum_histogram, get_intersection_points_y, get_intersection_points_x, get_weight_map, find_all_intersections_xy
from process_indexing_data import plot_phase_heatmap, plot_kam_with_learning_boundary
from PCA import run_PCA, detect_anomalies_pca
from cluster_analysis import (gmm_clustering, plot_cluster_heatmap, calculate_cluster_metrics,plot_intra_cluster_variation_map, plot_gmm_clusters, 
                            evaluate_clustering_metrics, plot_cluster_distances_ranking, find_best_reference_window, plot_cnmf_weights_projected)

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
set_global_determinism(seed=42, use_cuda=False)


#=== RoI ===
path = "/home/users/zhangqn8/storage/Partially reduced oxides 20 minutes Arbeitsbereich 3 Elementverteilungsdaten 5/Images_Valid/"
# path = "/Volumes/T7/ebsd_data/Images_Valid/"
path_to_phase_map = "/home/users/zhangqn8/workflow/ebsd_kikuchi/phase_map.png"
#kikuchi pattern args
height= 512
width = 672
slice_x = (86,586)
slice_y = (56,456)

# RoI region
# roi_1
# roi_xrange_x1 = (250,300)
# roi_yrange_x1 = (30,70)

# roi_2
# roi_xrange_x1 = (350,420)
# roi_yrange_x1 = (70,140)

# roi_3
# roi_xrange_x1 = (20,90)
# roi_yrange_x1 = (190,260)

# roi_4
# roi_xrange_x1 = (250,270)
# roi_yrange_x1 = (40,60)

# roi_5
# roi_xrange_x1 = (20,40)
# roi_yrange_x1 = (190,210)


roi_xrange_x1 = (365,385)
roi_yrange_x1 = (115,135)

roi_xb, roi_xt = roi_xrange_x1[0], roi_xrange_x1[1]
r_w = roi_xt-roi_xb
roi_yb, roi_yt = roi_yrange_x1[0], roi_yrange_x1[1]
r_h = roi_yt-roi_yb

# Save folder path
save_folder = f"./output_figure/x_{roi_xb}-{roi_xt}_y_{roi_yb}-{roi_yt}/"
# path_to_phase_map = "phase_map.png"
grid = read_data(path, path_to_phase_map, save_path=save_folder+"phase_map_grid.png")


X1, loc_1 = set_ROI(roi_xrange_x1,roi_yrange_x1, path, grid, path_to_phase_map, save_path=save_folder+"RoI_location.png")
# X_all, loc_all = set_ROI((0,495), (0,266), path, grid, path_to_phase_map, None)

#===software results===
df = pd.read_csv("~/workflow/process_experiment_data/ebsd_processed_with_grain_boundary.csv")
# df = pd.read_csv("ebsd_processed_with_grain_boundary.csv")

# coordinates: phase_id
coord_phase_dict = coord_phase_dict_from_dataframe(df, True)


pca_scores1, pca1, k, evr = run_PCA(X1, components="auto", h=height,w= width, slice_x=slice_x, slice_y= slice_y,
                                    incremental=False, batch_size=4096, save_curve_path=save_folder+"roi_pca_cumvar.png")


gmm_model1, cluster_coords_1, coord_to_label1, cluster_label1, optimal_n_pca, silhouette = gmm_clustering(pca_scores1, loc_1, n_components=None, max_components=10, selection="silhouette")

#anomalies_cluster_pca_scores1, anomalies_cluster_pca_coords1, anomalies_labels_pca1 = detect_anomalies_pca(pca_scores, coord_to_label1, loc_relative)
anomalies_cluster_pca_scores1, anomalies_cluster_pca_coords1, anomalies_coords_label1 = detect_anomalies_pca(pca_scores1, coord_to_label1, loc_1)
centers1, covs1, variations1 = calculate_cluster_metrics(gmm_model1, cluster_label1, pca_scores1)

top_samples_per_cluster, _= plot_cluster_distances_ranking(gmm_model1, cluster_label1, pca_scores1, loc_1, save_dir=save_folder, filename="Mahalanobis_ranking_clusterwise")
best_window = find_best_reference_window(top_samples_per_cluster, cluster_label1, variations1, loc_1)

plot_gmm_clusters(pca_scores1, cluster_label1, variations1, dim=2, anomalies=anomalies_cluster_pca_scores1, reference_windows=best_window,ellipse_alpha=0.3, save_dir=save_folder, filename=f'GMM_Clustering_for_PCA_scores_(n_clusters={optimal_n_pca})', dpi=300, show=False)

plot_cluster_heatmap(cluster_coords_1,img_shape=(r_h,r_w),save_dir=save_folder, filename="Cluster_Distribution_Heatmap(PCA)",dpi=300, show=False)


phase_labels = {
    -1: 'not_indexed',
    1: 'Iron bcc (old)',
    3: 'Hematite',
    4: 'Magnetite',
    5: 'Wuestite'
}
evaluate_results = evaluate_clustering_metrics(coord_phase_dict, coord_to_label1,name_map=phase_labels,cluster_name_map=None, print_table=True, mapping_mode="majority", save_json_path=save_folder+"pca_clustering_metrics.json")
mapped_phaseid= evaluate_results['detailed_results']['mapped_phaseid']
plot_intra_cluster_variation_map(loc_1, variations1, cluster_label1, (r_h,r_w), None, None, anomalies_cluster_pca_coords1, None, None, best_window, save_dir=save_folder, filename="Intra-Cluster_Variation_(Mahalanobis_Distance)", dpi=300, show=False)


R_list = []
ranges_list = []   # [(x_range, y_range, center_loc, key), ...]
ref_pos_list = []

# Sorted iteration for reproducibility
for key in sorted(best_window.keys(), key=lambda k: int(k)):
    entry = best_window[key]
    cx, cy = map(int, entry['center_loc'])  # center_loc: array([x, y])

    x0, x1 = cx - 1, cx + 1
    y0, y1 = cy - 1, cy + 1

    
    x_range = (x0, x1)
    y_range = (y0, y1)

    R, ref_pos = set_component(x_range, y_range, path, grid, path_to_phase_map, save_path=None)

    R_list.append(R)
    ref_pos_list.append(ref_pos)
    ranges_list.append((x_range, y_range, (cx, cy), int(key)))

# Build final components object
components = get_components(R_list, 3, None, height, width, slice_x, slice_y)


# run cNMF
weights1,mse,r_square = run_cNMF(X1, components, height, width, slice_x, slice_y)

# Sum of all weights
row_sums_minus_1 = weights1.sum(axis=1) - 1
abs_values = np.abs(row_sums_minus_1)

# Min and Max for the sum -1
max_abs = np.max(abs_values)
min_abs = np.min(abs_values)
max_abs_index = np.argmax(abs_values)
min_abs_index = np.argmin(abs_values)

print(f"Max absolute value of (weight sum-1): {max_abs} (In row {max_abs_index})")
print(f"Min absolute value of (weight sum-1): {min_abs} (In row {min_abs_index})")

gmm_model_cnmf1, cluster_coords_cnmf1, coord_to_label_cnmf1, cluster_labels_cnmf1, optimal_n_cnmf, silhouette = gmm_clustering(weights1, loc_1, optimal_n_pca, 10)

plot_cluster_heatmap(cluster_coords_cnmf1, img_shape=(r_h,r_w), save_dir=save_folder, filename="Cluster_Distribution_Heatmap(cNMF)",dpi=300, show=False)

evaluate_results = evaluate_clustering_metrics(coord_phase_dict, coord_to_label_cnmf1, name_map=phase_labels, cluster_name_map=None, print_table=True, save_json_path=save_folder+"cnmf_clustering_metrics.json")
cluster_name_map = evaluate_results['mapping_names']

plot_weight_histograms(weights1, loc_1, coord_to_label_cnmf1, cluster_name_map, None, None, save_dir=save_folder, filename="Weight_Distribution_for_Phases",dpi=300, show=False)
plot_weight_sum_histogram(weights1, 1, 20, save_dir=save_folder, filename="Weight_Sum_Histogram_Distribution",dpi=300, show=False)

weight_maps = get_weight_map(weights1, loc_1, r_h, r_w)
intersections1, fig1, _ = get_intersection_points_y(weight_maps, 0, roi_xrange_x1[0], roi_yrange_x1[0], r_h, r_w, save_dir=save_folder, filename='Weight_Map_(Feature_1)_and_Weight_Values_at_Highlighted_Row_0',show=False)
intersections2, fig2, _ = get_intersection_points_y(weight_maps, 5, roi_xrange_x1[0], roi_yrange_x1[0], r_h, r_w, save_dir=save_folder, filename='Weight_Map_(Feature_1)_and_Weight_Values_at_Highlighted_Row_5',show=False)
intersections3, fig3, _ = get_intersection_points_y(weight_maps, 10, roi_xrange_x1[0], roi_yrange_x1[0], r_h, r_w, save_dir=save_folder, filename='Weight_Map_(Feature_1)_and_Weight_Values_at_Highlighted_Row_10',show=False)
intersections4, fig4, _ = get_intersection_points_y(weight_maps, 15, roi_xrange_x1[0], roi_yrange_x1[0], r_h, r_w, save_dir=save_folder, filename='Weight_Map_(Feature_1)_and_Weight_Values_at_Highlighted_Row_15',show=False)

intersections1, fig1, _ = get_intersection_points_x(weight_maps, 0, roi_xrange_x1[0], roi_yrange_x1[0], r_h, r_w, save_dir=save_folder, filename='Weight_Map_(Feature_1)_and_Weight_Values_at_Highlighted_Column_0',show=False)
intersections2, fig2, _ = get_intersection_points_x(weight_maps, 5, roi_xrange_x1[0], roi_yrange_x1[0], r_h, r_w, save_dir=save_folder, filename='Weight_Map_(Feature_1)_and_Weight_Values_at_Highlighted_Column_5',show=False)
intersections3, fig3, _ = get_intersection_points_x(weight_maps, 10, roi_xrange_x1[0], roi_yrange_x1[0], r_h, r_w, save_dir=save_folder, filename='Weight_Map_(Feature_1)_and_Weight_Values_at_Highlighted_Column_10',show=False)
intersections4, fig4, _ = get_intersection_points_x(weight_maps, 15, roi_xrange_x1[0], roi_yrange_x1[0], r_h, r_w, save_dir=save_folder, filename='Weight_Map_(Feature_1)_and_Weight_Values_at_Highlighted_Column_15',show=False)

all_intersections = find_all_intersections_xy(weights1, loc_1, r_h, r_w)

#=== boundary detected both in the direction x and y ====

boundary_pairs = set([(point[0], point[1]) for point in all_intersections['combined']])
boundary_label_dict = {coord: coord_to_label_cnmf1[coord] 
                for coord in boundary_pairs if coord in coord_to_label_cnmf1}
jaccard, overlap_coefficient, _= plot_weight_map_cnmf_with_anomalies(weights1, loc_1, r_h, r_w,anomalies_dict= anomalies_coords_label1, ref_pos_list=ref_pos_list, component=0, boundary_locs=boundary_pairs, save_dir=save_folder, filename=f"Component_1_Weight_Map_with_Annotations", show=False)

# plot_cnmf_weights_projected(weights1, loc_1, cluster_label1, (0,1), '2d', boundary_pairs, ref_pos_list, anomalies_coords_label1)
plot_phase_heatmap(coord_phase_dict, boundary_loc_label_dict= boundary_label_dict, anomalies_loc_label_dict= anomalies_coords_label1, coor_phase_map=phase_labels, boundary_loc_label_map=cluster_name_map, cluster_name_map=cluster_name_map, image_size=(r_h,r_w), roi_xrange= roi_xrange_x1, roi_yrange=roi_yrange_x1, save_dir=save_folder, filename="Phase_Map_with_Boundary/Anomaly_Overlays_(compare_by_phase_name)",show=False)
plot_phase_heatmap(coord_phase_dict, boundary_loc_label_dict= boundary_label_dict, anomalies_loc_label_dict= None, coor_phase_map=phase_labels, boundary_loc_label_map=cluster_name_map, cluster_name_map=cluster_name_map, image_size=(r_h,r_w), roi_xrange= roi_xrange_x1, roi_yrange=roi_yrange_x1, save_dir=save_folder, filename="Phase_Map_with_Boundary_Overlays_(compare_by_phase_name)",show=False)

plot_kam_with_learning_boundary(df, "phase_boundary", roi_xrange_x1, roi_yrange_x1, boundary_loc_label_dict=boundary_label_dict, anomalies_loc_label_dict=anomalies_coords_label1, boundary_loc_label_map=cluster_name_map, anomalies_loc_label_map=cluster_name_map, figsize=(10,8), kam_col="KAM", x_col="x_indice", y_col="y_indice", save_dir=save_folder, filename=f"KAM_map_(ROI_x∈[{roi_xb},{roi_xt}]_y∈[{roi_yb},{roi_yt}])_phase_boundary_contour",show=False)
plot_kam_with_learning_boundary(df, "GB_0.5", roi_xrange_x1, roi_yrange_x1, boundary_loc_label_dict=boundary_label_dict, anomalies_loc_label_dict=None, boundary_loc_label_map=cluster_name_map, anomalies_loc_label_map=cluster_name_map, figsize=(10,8), kam_col="KAM", x_col="x_indice", y_col="y_indice", save_dir=save_folder, filename=f"KAM_map_(ROI_x∈[{roi_xb},{roi_xt}]_y∈[{roi_yb},{roi_yt}])_GB0.5_contour",show=False)
plot_kam_with_learning_boundary(df, "GB_0.2", roi_xrange_x1, roi_yrange_x1, boundary_loc_label_dict=boundary_label_dict, anomalies_loc_label_dict=None, boundary_loc_label_map=cluster_name_map, anomalies_loc_label_map=cluster_name_map, figsize=(10,8), kam_col="KAM", x_col="x_indice", y_col="y_indice", save_dir=save_folder, filename=f"KAM_map_(ROI_x∈[{roi_xb},{roi_xt}]_y∈[{roi_yb},{roi_yt}])_GB0.2_contour",show=False)
