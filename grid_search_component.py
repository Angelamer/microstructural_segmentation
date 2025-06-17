"""
Select the optimal reference component for cnmf
Licensed under GNU GPL3, see license file LICENSE_GPL3.
"""

import numpy as np
import pandas as pd
import matplotlib as plt
from tqdm import tqdm  # Progress tool
from visualize_grid import set_component
from data_processing import get_components
from cNMF import run_cNMF

# define the searching range for 
def generate_candidates(x_all_range, y_all_range, grid_size=2, step=1):
    """generate the component candidate within the selected part"""
    candidates = []
    
    for x_start in range(x_all_range[0], x_all_range[1] - grid_size, step):
        for y_start in range(y_all_range[0], y_all_range[1] - grid_size, step):
            x_range = (x_start, x_start+grid_size)
            y_range = (y_start, y_start+grid_size)
            candidates.append((x_range, y_range))
    return candidates

def generate_paired_candidates(x_range_r1, y_range_r1, x_range_r2, y_range_r2, grid_size=2, step=1):
    r1_candidates = generate_candidates(x_range_r1, y_range_r1,grid_size,step)
    r2_candidates = generate_candidates(x_range_r2, y_range_r2,grid_size,step)
    print(r1_candidates)
    paired_candidates = []
    for r1 in r1_candidates:
        for r2 in r2_candidates:
            paired_candidates.append({
                'R1': {
                    'x_range': r1[0],
                    'y_range': r1[1]
                },
                'R2': {
                    'x_range': r2[0],
                    'y_range': r2[1]
                }
            })
    
    return paired_candidates
# training and evaluate
def evaluate_components(candidate_pair, ROI, path, grid):
    
    x_range_r1, y_range_r1 = candidate_pair['R1']['x_range'], candidate_pair['R1']['y_range']
    x_range_r2, y_range_r2 = candidate_pair['R2']['x_range'], candidate_pair['R2']['y_range']
    print(x_range_r1)
    # obtain the component path
    R1, ref1_pos = set_component(x_range_r1,
        y_range_r1, path, grid, False)
    R2, ref2_pos = set_component(x_range_r2,
        y_range_r2, path, grid, False) 
    
    # generate a set of components
    try:
        components = get_components(R1, R2)
    except Exception as e:
        print(f"Error generating components: {str(e)}")
        return None, None, None
    
    # cnmf
    weights, mse, r_square = run_cNMF(ROI, components)
    # print(weights)
    # calculate the average errors
    avg_mse = np.mean(mse)
    avg_r2 = np.mean(r_square)
    
    return {
        'R1_x_range': candidate_pair['R1']['x_range'], 
        'R1_y_range': candidate_pair['R1']['y_range'], 
        'R2_x_range': candidate_pair['R2']['x_range'],
        'R2_y_range': candidate_pair['R2']['y_range'],
        'mse': avg_mse,
        'r2': avg_r2,
        'components': components,
        'weights' : weights
    }


# optimization cycle
def optimize_paired_components(ROI, path, grid, x_range_r1, y_range_r1, x_range_r2, y_range_r2, grid_size=2, step=1, top_k=3):
    
    """
    Search and optimize paired regions (component pairs), evaluate the model performance (such as MSE and R2) for each pair,
    and return the best results.
    """
    paired_candidates = generate_paired_candidates(x_range_r1, y_range_r1, x_range_r2, y_range_r2, grid_size, step) 
    # print(paired_candidates)
    results = []
    progress_bar = tqdm(paired_candidates, desc="Optimizing paired components")
    print(f"The number of component pair candidate: {len(paired_candidates)}")
    for pair in progress_bar:
        
        try:
            metrics = evaluate_components(pair, ROI, path, grid)
            # print(metrics)
            if metrics:
                results.append(metrics)
                progress_bar.set_postfix( current_mse=metrics['mse'], current_r2=metrics['r2'])
        except Exception as e:
            print(f"Error processing: {str(e)}")
    
    # assess the metrics based on the mse and r2
    df = pd.DataFrame(results)
    # print(df)
    df = df.sort_values(by=['mse', 'r2'], ascending=[True, False])
    

    # return the first top ranking results
    return df.head(top_k)

# result visualization
def visualize_results(df_result):
    """visualize the best result"""
    # the best R1/R2 pair distribution
    print("\nBest R1 component:")
    print(df_result['R1_range'].value_counts().head())
    
    print("\nBest R1 component:")
    print(df_result['R2_range'].value_counts().head())
    # plot the mse distribution
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df_result['R1_range'].astype(str),
        y=df_result['R2_range'].astype(str),
        hue=df_result['mse'],
        size=df_result['r2'],
        palette='viridis'
    )
    plt.title("R1-R2 pair component distribution")
    plt.xticks(rotation=45)
    plt.show()
