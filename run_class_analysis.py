import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.plot_helpers import plot_erp_2cons_results
from utils.techniques import run_erps_analysis
from utils.analysis_helpers import prepare_data_word_class
from utils.eeg_helpers import get_channel_name_ids
from utils.config import CHANNEL_NAMES, INTERVALS, T_MAX, T_MIN, best_clusters



if __name__ == '__main__':
    # Create the time vector
    t_min = T_MIN * 1000
    t_max = T_MAX * 1000
    step  = INTERVALS * 1000
    times = np.arange(t_min, t_max, step)   

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Word Class Analysis - ERP')

    # Add parameters to the parser
    parser.add_argument('-category', type=str, help='Specify the word type e.g, NOUN, VERB, ADJ, ADV, PRON, AUX, ADP, DET')
    parser.add_argument('-region', type=str, help='Specify the brain region')
    parser.add_argument('-permutation', type=bool, help='Conduct cluster-based permutation test or not')
    parser.add_argument('-p_value', type=float, help='The threshold of p-values')
    parser.add_argument('-clusterp', type=float, help='The threshold of cluster-defining p-values')
    parser.add_argument('-n_iter', type=int, help='The times for iteration.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    word_type = args.category
    name_region = args.region
    p_value = args.p_value
    cluster_permutation=args.permutation
    clusterp = args.clusterp
    n_iter = args.n_iter

    if word_type not in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'AUX', 'ADP', 'DET', 'content', 'function']:
        raise ValueError("Invalid word type. Please specify a valid word category")

    # Save figure
    output_folder = f"photo/{word_type}/univariate_analysis/{name_region}"
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    EEG_data, labels = prepare_data_word_class(word_type)
    


    if name_region not in ['left_hemisphere', 'right_hemisphere', 'midlines', 'all', 'best']:
        raise ValueError("Invalid brain region. Please specify a valid brain region")
    
    if name_region == 'best':
        region_channels = best_clusters[f'{word_type}_selected_chans']
        indices = [i for i, value in enumerate(CHANNEL_NAMES) if value in(region_channels)]
    else:
        indices, region_channels =  get_channel_name_ids(name_region)
    # EEG_high_cloze will contain columns of data where the corresponding label is 0
    EEG_high = EEG_data[:, labels[0] == 0, :, :]* 1e6
    EEG_high = EEG_high[:, :, indices, :]

    # EEG_low_cloze will contain columns of data where the corresponding label is 1
    EEG_low = EEG_data[:, labels[0] == 1, :, :]* 1e6
    EEG_low = EEG_low[:, :, indices, :]

    # Average across trials to compute the ERP
    ERP_high = np.mean(EEG_high, axis=(1,2)) # Determine ERP for condition content words (average across trials)
    ERP_low  = np.mean(EEG_low, axis=(1,2)) # Determine ERP for condition function words (average across trials)     

    # Run ERPs analysis
    p_vals, avg1, err1, avg2, err2 = run_erps_analysis(ERP_high, ERP_low, cluster_permutation=cluster_permutation, clusterp=clusterp, iter=n_iter)

    # Visualisation
    fig = plt.figure(figsize=(13, 8))
    if cluster_permutation:
        plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=['High Cloze',  'Low Cloze'], 
                                ylim=[-1, 1], p_threshold=p_value, labelpad=0, cluster_permutation=True)
    else:
        plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=['High Cloze', 'Low Cloze'], 
                                ylim=[-1, 1], p_threshold=p_value, labelpad=0)
    # plt.title(f'ERPs from High (red) and Low (green) predictable words at channel {channel_names[index]}', pad=20)
    # plt.show()
    fig.savefig(output_folder + f'/best_ERPs_{word_type}_in_{name_region}.png', dpi=500, bbox_inches='tight')

    for chan in tqdm(indices,  position=0, leave=True):
        # EEG_high_cloze will contain columns of data where the corresponding label is 0
        EEG_high = EEG_data[:, labels[0] == 0, chan, :]* 1e6

        # EEG_low_cloze will contain columns of data where the corresponding label is 1
        EEG_low = EEG_data[:, labels[0] == 1, chan, :]* 1e6

        # Average across trials to compute the ERP
        ERP_high = np.mean(EEG_high, axis=1) # Determine ERP for condition content words (average across trials)
        ERP_low  = np.mean(EEG_low, axis=1) # Determine ERP for condition function words (average across trials)     

        # Run ERPs analysis
        p_vals, avg1, err1, avg2, err2 = run_erps_analysis(ERP_high, ERP_low, cluster_permutation=cluster_permutation, clusterp=clusterp, iter=n_iter)

        # Visualisation
        fig = plt.figure(figsize=(13, 8))
        if cluster_permutation:
            plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=[f'{word_type.capitalize()} - High Cloze', f'{word_type.capitalize()} - Low Cloze'], 
                                ylim=[-6, 6], p_threshold=p_value, labelpad=0, cluster_permutation=True)
        else:
            plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=[f'{word_type.capitalize()} - High Cloze', f'{word_type.capitalize()} - Low Cloze'], 
                                ylim=[-6, 6], p_threshold=p_value, labelpad=0, cluster_permutation=True)
                                
        # plt.title(f'ERPs from High (red) and Low (green) predictable words at channel {channel_names[index]}', pad=20)
        # plt.show()
        fig.savefig(output_folder + f'/{word_type}_channel_{CHANNEL_NAMES[chan]}.png', dpi=500, bbox_inches='tight')

    print("All saved!")