import argparse
import os
import pickle

import numpy as np

from utils.analysis_helpers import prepare_data_word_class
from utils.eeg_helpers import get_channel_name_ids
from utils.config import CHANNEL_NAMES, INTERVALS, T_MAX, T_MIN



if __name__ == '__main__':
    # Create the time vector
    t_min = T_MIN * 1000
    t_max = T_MAX * 1000
    step  = INTERVALS * 1000
    times = np.arange(t_min, t_max, step)   

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Word class analysis')

    # Add parameters to the parser
    parser.add_argument('-category', type=str, help='Specify the word type e.g, NOUN, VERB, ADJ, ADV, PRON, AUX, ADP, DET')
    parser.add_argument('-region', type=str, help='Specify the brain region')
    parser.add_argument('-permutation', default=False, action=argparse.BooleanOptionalAction, help='Run cluster-based permutaion test or not')
    parser.add_argument('-clusterp', type=float, help='The threshold of cluster-defining p-values')
    parser.add_argument('-n_iter', type=int, help='The times for iteration.')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    word_type = args.category
    name_region = args.region
    cluster_permutation=args.permutation
    clusterp = args.clusterp
    n_iter = args.n_iter

    # Save figure
    output_folder = f"photo/{word_type}/univariate_analysis/{name_region}"
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    EEG_data, labels = prepare_data_word_class(word_type)
    indices =  get_channel_name_ids(name_region)
    # best_channels = settings['channels'][f'{word_type}_selected_chans']
    # best_indices = [i for i, value in enumerate(CHANNEL_NAMES) if value in(best_channels)]

    if name_region not in ['left_hemisphere', 'right_hemisphere', 'midlines', 'all', 'best']:
        raise ValueError("Invalid brain region. Please specify a valid brain region")

    # EEG_high_cloze will contain columns of data where the corresponding label is 0
    EEG_high_cloze = EEG_data[:, labels[0] == 0, :, :]* 1e6
    EEG_high_cloze = EEG_high_cloze[:, :, best_indices, :]

    # EEG_low_cloze will contain columns of data where the corresponding label is 1
    EEG_low_cloze = EEG_data[:, labels[0] == 1, :, :]* 1e6
    EEG_low_cloze = EEG_low_cloze[:, :, best_indices, :]

    # Average across trials to compute the ERP
    ERP_high_cloze = np.mean(EEG_high_cloze, axis=(1,2)) # Determine ERP for condition high probability words (average across trials)
    ERP_low_cloze  = np.mean(EEG_low_cloze, axis=(1,2)) # Determine ERP for condition low probability words (average across trials)     

    # Run ERPs analysis
    p_vals, avg1, err1, avg2, err2 = run_erps_analysis(ERP_high_cloze, ERP_low_cloze, cluster_permutation=cluster_permutation, clusterp=clusterp, iter=iter)

    # Visualisation
    fig = plt.figure(figsize=(13, 8))
    if cluster_permutation:
        plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=['High Cloze',  'Low Cloze'], 
                                ylim=[-1, 1.2], p_threshold=settings['statistics']['p_threshold'], labelpad=0, cluster_permutation=True)
    else:
        plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=['High Cloze', 'Low Cloze'], 
                                ylim=[-1, 1.2], p_threshold=settings['statistics']['p_threshold'], labelpad=0)
    # plt.title(f'ERPs from High (red) and Low (green) predictable words at channel {channel_names[index]}', pad=20)
    plt.show()
    fig.savefig(output_folder + f'/best_ERPs_{word_type}_in_{name_region}.png', dpi=500, bbox_inches='tight')

    for chan in tqdm(indices,  position=0, leave=True):
        # EEG_high_cloze will contain columns of data where the corresponding label is 0
        EEG_high_cloze = EEG_data[:, labels[0] == 0, chan, :]* 1e6

        # EEG_low_cloze will contain columns of data where the corresponding label is 1
        EEG_low_cloze = EEG_data[:, labels[0] == 1, chan, :]* 1e6

        # # Identify the baseline period indices (-100ms to 0ms)
        # baseline_start = 0  # Corresponding to -100ms
        # baseline_end = int(abs(settings['epochs']['tmin'] - baseline_start) * 1000)  # Corresponding to 0ms (exclusive)

        # # Compute the baseline mean for each trial and each channel
        # baseline_mean_high = np.mean(EEG_high_cloze[:, :, baseline_start:baseline_end], axis=2, keepdims=True)
        # baseline_mean_low  = np.mean(EEG_low_cloze[:, :, baseline_start:baseline_end], axis=2, keepdims=True)

        # # Subtract the baseline mean from the entire time series for each trial and each channel
        # EEG_high_cloze_corrected = EEG_high_cloze - baseline_mean_high
        # EEG_low_cloze_corrected  = EEG_low_cloze - baseline_mean_low


        # # Average across trials to compute the ERP
        # ERP_high_cloze = np.mean(EEG_high_cloze_corrected, axis=1) # Determine ERP for condition high probability words (average across trials)
        # ERP_low_cloze  = np.mean(EEG_low_cloze_corrected, axis=1) # Determine ERP for condition low probability words (average across trials)     

        # Average across trials to compute the ERP
        ERP_high_cloze = np.mean(EEG_high_cloze, axis=1) # Determine ERP for condition high probability words (average across trials)
        ERP_low_cloze  = np.mean(EEG_low_cloze, axis=1) # Determine ERP for condition low probability words (average across trials)     

        # Run ERPs analysis
        p_vals, avg1, err1, avg2, err2 = run_erps_analysis(ERP_high_cloze, ERP_low_cloze, cluster_permutation=cluster_permutation, clusterp=clusterp, iter=iter)

        # Visualisation
        fig = plt.figure(figsize=(13, 8))
        if cluster_permutation:
            plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=[f'{word_type.capitalize()} - High Cloze', f'{word_type.capitalize()} - Low Cloze'], 
                                ylim=[-6, 6], p_threshold=settings['statistics']['p_threshold'], labelpad=0, cluster_permutation=True)
        else:
            plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=[f'{word_type.capitalize()} - High Cloze', f'{word_type.capitalize()} - Low Cloze'], 
                                ylim=[-6, 6], p_threshold=settings['statistics']['p_threshold'], labelpad=0)
        # plt.title(f'ERPs from High (red) and Low (green) predictable words at channel {channel_names[index]}', pad=20)
        plt.show()
        fig.savefig(output_folder + f'/{word_type}_channel_{channel_names[chan]}.png', dpi=500, bbox_inches='tight')

    print("All saved!")