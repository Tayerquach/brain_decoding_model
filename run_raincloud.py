import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils.eeg_helpers import get_channel_name_ids
from utils.plot_helpers import plot_raincloud_diff, plot_raincloud_decoding
from utils.analysis_helpers import prepare_data_word_class
from utils.config import T_MIN, INTERVALS, CHANNEL_NAMES, best_clusters
import pandas as pd

def get_time_window(start_idx, end_idx):
    """
    Adjusts the given indices based on T_MIN, T_MAX, and INTERVALS.

    Args:
        start_idx (int): Start time window
        end_idx (int):End time window
        t_min (float): Minimum time value (e.g., -0.1).
        t_max (float): Maximum time value (e.g., 0.5).
        interval (float): Time interval step (e.g., 0.001).

    Returns:
        list: Adjusted indices corresponding to T_MIN and T_MAX.
    """
    # Compute the actual time range based on the indices
    start_time = start_idx * INTERVALS
    end_time = end_idx * INTERVALS

    # Adjust the indices relative to T_MIN
    adjusted_start = int((start_time - T_MIN) / INTERVALS)
    adjusted_end = int((end_time - T_MIN) / INTERVALS)

    return [adjusted_start, adjusted_end]

def prepare_data(group, region_type, technique, time_window):
    # Access the parameter values
    type_mapping = {
        'lexical_class': ['content', 'function'],
        'content_group': ['NOUN', 'VERB'],
        'function_group': ['DET', 'PRON']
    }
    regions = ['left_hemisphere', 'midlines', 'right_hemisphere']
    word_type = type_mapping[group]
    values = []

    if technique == 'univariate':
        if region_type == 'best':
            for category in word_type:
                output_folder = f'{technique}_data/word_class/{category}'
                # Check if the folder exists, if not create it
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                EEG_data, labels = prepare_data_word_class(category)  
                region_channels = best_clusters[f'{category}_selected_chans']
                indices = [i for i, value in enumerate(CHANNEL_NAMES) if value in(region_channels)] 
                best_indices = indices.copy()

                # EEG_data
                EEG_data_region = EEG_data[:,:, best_indices, :] * 1e6
                # EEG_high_cloze will contain columns of data where the corresponding label is 0
                EEG_high_cloze_category_region = EEG_data_region[:, labels[0] == 0, :, :]
                EEG_high_cloze_category_region_window = EEG_high_cloze_category_region[:, :, :, time_window[0]:time_window[1]] # (300-500 ms)

                # EEG_low_cloze will contain columns of data where the corresponding label is 1
                EEG_low_cloze_category_region = EEG_data_region[:, labels[0] == 1, :, :]
                EEG_low_cloze_category_region_window= EEG_low_cloze_category_region[:, :, :, time_window[0]:time_window[1]] # (300-500 ms)
                # EEG difference between high vs. low cloze
                ERP_diff_category_region_window = np.average(EEG_high_cloze_category_region_window, axis=(1,2)) - np.average(EEG_low_cloze_category_region_window, axis=(1,2))
                # Save data
                with open(output_folder + f'/erp_diff_{region_type}_window.pkl', 'wb') as file:
                    pickle.dump(ERP_diff_category_region_window, file)
                values.append(abs(ERP_diff_category_region_window.flatten()))
            # Append data to the dictionary
            n_subs, n_timepoints = ERP_diff_category_region_window.shape
            conditions = [f'{category.capitalize()}/Cloze' for category in word_type for _ in range(n_subs * n_timepoints)]

        elif region_type == 'separate':
            for name_region in regions:
                for category in word_type:
                    output_folder = f'{technique}_data/word_class/{category}'
                    # Check if the folder exists, if not create it
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    EEG_data, labels = prepare_data_word_class(category)   
                    # Regions
                    indices, region_channels =  get_channel_name_ids(name_region)
                    # best_channels = best_clusters[f'{category}_selected_chans']
                    # best_indices = [i for i, value in enumerate(region_channels) if value in(best_channels)]
                    best_indices = indices.copy()

                    # EEG_data
                    EEG_data_region = EEG_data[:,:, best_indices, :] * 1e6
                    # EEG_high_cloze will contain columns of data where the corresponding label is 0
                    EEG_high_cloze_category_region = EEG_data_region[:, labels[0] == 0, :, :]
                    EEG_high_cloze_category_region_window = EEG_high_cloze_category_region[:, :, :, time_window[0]:time_window[1]] # (300-500 ms)

                    # EEG_low_cloze will contain columns of data where the corresponding label is 1
                    EEG_low_cloze_category_region = EEG_data_region[:, labels[0] == 1, :, :]
                    EEG_low_cloze_category_region_window= EEG_low_cloze_category_region[:, :, :, time_window[0]:time_window[1]] # (300-500 ms)
                    # EEG difference between high vs. low cloze
                    ERP_diff_category_region_window = np.average(EEG_high_cloze_category_region_window, axis=(1,2)) - np.average(EEG_low_cloze_category_region_window, axis=(1,2))
                    # Save data
                    with open(output_folder + f'/erp_diff_{name_region}_window.pkl', 'wb') as file:
                        pickle.dump(ERP_diff_category_region_window, file)
                    values.append(abs(ERP_diff_category_region_window.flatten()))
            # Append data to the dictionary
            n_subs, n_timepoints = ERP_diff_category_region_window.shape
            conditions = [f'{category.capitalize()}/Cloze' for category in word_type for _ in range(n_subs * n_timepoints)]
            conditions = conditions * len(regions)
    
    elif technique == 'decoding':
        if region_type == 'best':
            for category in word_type:
                # Group accuracies
                with open(f'decoding_data/word_class/{category}/word_class_{region_type}_accuracies.pkl', 'rb') as f:
                    region_category_accuracies = pickle.load(f)
                # Prepare data
                ## Time window - Group 
                region_category_accuracies_window = region_category_accuracies[:,time_window[0]:time_window[1]]
                
                values.append(region_category_accuracies_window.flatten())  
            # Append data to the dictionary
            n_subs, n_timepoints = region_category_accuracies_window.shape 
            conditions = [f'{category.capitalize()}/Cloze' for category in word_type for _ in range(n_subs * n_timepoints)]
        
        elif region_type == 'separate':
            for name_region in regions:
                for category in word_type:
                    # Group accuracies
                    with open(f'decoding_data/word_class/{category}/word_class_{name_region}_accuracies.pkl', 'rb') as f:
                        region_category_accuracies = pickle.load(f)
                    # Prepare data
                    ## Time window - Group 
                    region_category_accuracies_window = region_category_accuracies[:,time_window[0]:time_window[1]]
                    
                    values.append(region_category_accuracies_window.flatten())  
            # Append data to the dictionary
            n_subs, n_timepoints = region_category_accuracies_window.shape 

            conditions = [f'{category.capitalize()}/Cloze' for category in word_type for _ in range(n_subs * n_timepoints)]
            conditions = conditions * len(regions)
    return values, conditions, n_subs, n_timepoints

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Word class analysis')

    # Add parameters to the parser
    parser.add_argument('-category', type=str, help='Specify the word group')
    parser.add_argument('-region_type', type=str, help='Specify the regions (e.g., separate (including left, midline, right hemisphere), best)')
    parser.add_argument('-technique', type=str, help='Specify the technique (e.g., univariate, decoding)')
    parser.add_argument('-start_window', type=int, help='Specify the beginning of time window')
    parser.add_argument('-end_window', type=int, help='Specify the end of time window')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    group = args.category
    region_type = args.region_type
    technique = args.technique
    start_window = args.start_window 
    end_window = args.end_window

    time_window = get_time_window(start_window, end_window)
    values, conditions, n_subs, n_timepoints = prepare_data(group, region_type, technique, time_window)
    
    if technique == 'univariate':
        if region_type == 'separate':
            measure = 'Diff_amp'
            data = {
            'Region': ['Left Hemisphere'] * 2 * n_subs * n_timepoints + ['Midlines'] * 2 * n_subs * n_timepoints + ['Right Hemisphere'] * 2 * n_subs * n_timepoints,
            'Condition': conditions,
            measure:  np.concatenate(values)
            }
            df = pd.DataFrame(data)
            plot_raincloud_diff(df, group)

        elif region_type == 'best':
            measure = 'Diff_amp'
            # import pdb; pdb.set_trace()
            data = {
            'Region': ['Optimal Electrodes'] * 2 * n_subs * n_timepoints,
            'Condition': conditions,
            measure:  np.concatenate(values)
            }
            df = pd.DataFrame(data)
            plot_raincloud_diff(df, group)

    elif technique == 'decoding':
        if region_type == 'best':
            measure = 'Accuracy'
            data = {
            'Region': ['Optimal Electrodes'] * 2 * n_subs * n_timepoints,
            'Condition': conditions,
            measure:  np.concatenate(values)
            }
            
            df = pd.DataFrame(data)
            plot_raincloud_decoding(df, group)

        elif region_type == 'separate':
            measure = 'Accuracy'
            data = {
            'Region': ['Optimal Electrodes'] * 2 * n_subs * n_timepoints + ['Midlines'] * 2 * n_subs * n_timepoints + ['Right Hemisphere'] * 2 * n_subs * n_timepoints,
            'Condition': conditions,
            measure:  np.concatenate(values)
            }
            
            df = pd.DataFrame(data)
            plot_raincloud_decoding(df, group)


