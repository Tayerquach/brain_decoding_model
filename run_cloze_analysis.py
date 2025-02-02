import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.plot_helpers import plot_erp_2cons_results
from utils.techniques import run_erps_analysis
from utils.analysis_helpers import prepare_data_cloze
from utils.eeg_helpers import get_channel_name_ids
from utils.config import CHANNEL_NAMES, INTERVALS, T_MAX, T_MIN, best_clusters



if __name__ == '__main__':
    # Create the time vector
    t_min = T_MIN * 1000
    t_max = T_MAX * 1000
    step  = INTERVALS * 1000
    times = np.arange(t_min, t_max, step)   

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Cloze probability analysis')

    # Add parameters to the parser
    parser.add_argument('-cloze', type=str, help='Specify the cloze probability type e.g, high or low')
    parser.add_argument('-region', type=str, help='Specify the brain region')
    parser.add_argument('-permutation', type=bool, help='Conduct cluster-based permutation test or not')
    parser.add_argument('-p_value', type=float, help='The threshold of p-values')
    parser.add_argument('-clusterp', type=float, help='The threshold of cluster-defining p-values')
    parser.add_argument('-n_iter', type=int, help='The times for iteration.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    cloze_type = args.cloze
    name_region = args.region
    p_value = args.p_value
    cluster_permutation=args.permutation
    clusterp = args.clusterp
    n_iter = args.n_iter

    if cloze_type not in ['high', 'low']:
        raise ValueError("Invalid cloze type. Please specify either high or low")

    # Save figure
    output_folder = f"photo/{cloze_type}/univariate_analysis/{name_region}"
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    EEG_data, labels = prepare_data_cloze(cloze_type)
    indices, region_channels =  get_channel_name_ids(name_region)


    if name_region not in ['left_hemisphere', 'right_hemisphere', 'midlines', 'all', 'best']:
        raise ValueError("Invalid brain region. Please specify a valid brain region")

    # EEG_high_cloze will contain columns of data where the corresponding label is 0
    EEG_content = EEG_data[:, labels[0] == 0, :, :]* 1e6
    EEG_content = EEG_content[:, :, indices, :]

    # EEG_low_cloze will contain columns of data where the corresponding label is 1
    EEG_function = EEG_data[:, labels[0] == 1, :, :]* 1e6
    EEG_function = EEG_function[:, :, indices, :]

    # Average across trials to compute the ERP
    ERP_content = np.mean(EEG_content, axis=(1,2)) # Determine ERP for condition content words (average across trials)
    ERP_function  = np.mean(EEG_function, axis=(1,2)) # Determine ERP for condition function words (average across trials)     

    # Run ERPs analysis
    p_vals, avg1, err1, avg2, err2 = run_erps_analysis(ERP_content, ERP_function, cluster_permutation=cluster_permutation, clusterp=clusterp, iter=n_iter)

    # Visualisation
    fig = plt.figure(figsize=(13, 8))
    if cluster_permutation:
        plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=['Content',  'Function'], 
                                ylim=[-2, 2.2], p_threshold=p_value, labelpad=0, cluster_permutation=True)
    else:
        plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=['Content', 'Function'], 
                                ylim=[-2, 2.2], p_threshold=p_value, labelpad=0)
    # plt.title(f'ERPs from High (red) and Low (green) predictable words at channel {channel_names[index]}', pad=20)
    # plt.show()
    fig.savefig(output_folder + f'/best_ERPs_{cloze_type}_in_{name_region}.png', dpi=500, bbox_inches='tight')

    for chan in tqdm(indices,  position=0, leave=True):
        # EEG_high_cloze will contain columns of data where the corresponding label is 0
        EEG_content = EEG_data[:, labels[0] == 0, chan, :]* 1e6

        # EEG_low_cloze will contain columns of data where the corresponding label is 1
        EEG_function = EEG_data[:, labels[0] == 1, chan, :]* 1e6

        # Average across trials to compute the ERP
        ERP_content = np.mean(EEG_content, axis=1) # Determine ERP for condition content words (average across trials)
        ERP_function  = np.mean(EEG_function, axis=1) # Determine ERP for condition function words (average across trials)     

        # Run ERPs analysis
        p_vals, avg1, err1, avg2, err2 = run_erps_analysis(ERP_content, ERP_function, cluster_permutation=cluster_permutation, clusterp=clusterp, iter=n_iter)

        # Visualisation
        fig = plt.figure(figsize=(13, 8))
        if cluster_permutation:
            plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=[f'{cloze_type.capitalize()} - Content Words', f'{cloze_type.capitalize()} - Function Words'], 
                                ylim=[-6, 6], p_threshold=p_value, labelpad=0, cluster_permutation=True)
        else:
            plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=[f'{cloze_type.capitalize()} - Content Words', f'{cloze_type.capitalize()} - Function Words'], 
                                ylim=[-6, 6], p_threshold=p_value, labelpad=0, cluster_permutation=True)
                                
        # plt.title(f'ERPs from High (red) and Low (green) predictable words at channel {channel_names[index]}', pad=20)
        # plt.show()
        fig.savefig(output_folder + f'/{cloze_type}_channel_{CHANNEL_NAMES[chan]}.png', dpi=500, bbox_inches='tight')

    print("All saved!")