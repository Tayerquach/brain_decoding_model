import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils.eeg_helpers import get_channel_name_ids
from utils.analysis_helpers import prepare_data_word_class
from utils.config import T_MIN, INTERVALS, CHANNEL_NAMES, best_clusters
from ptitprince import PtitPrince as pt
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

def prepare_data(group, time_window):
    # Initialize the dictionary
    data = {
        'Region': [],
        'Condition': [],
        'Diff_amp': []
    }
    # Access the parameter values
    type_mapping = {
        'lexical_class': ['content', 'function'],
        'content_group': ['NOUN', 'VERB'],
        'function_group': ['DET', 'PRON', 'ADP', 'AUX']
    }
    regions = ['left_hemisphere', 'midlines', 'right_hemisphere']
    word_type = type_mapping[group]
    num_groups = len(group)
    num_regions = len(regions)
    values = []

    output_folder = f'univariate_data/raincloud/{group}'
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for category in word_type:
        for name_region in regions:
            EEG_data, labels = prepare_data_word_class(category)   
            # Regions
            significant_channels = best_clusters[f'{category}_selected_chans']
            best_indices = [i for i, value in enumerate(CHANNEL_NAMES) if value in(significant_channels)]
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
            with open(output_folder + f'/{group}_{name_region}_window.pkl', 'wb') as file:
                pickle.dump(ERP_diff_category_region_window, file)

            values.append(abs(ERP_diff_category_region_window.flatten()))

    # Append data to the dictionary
    n_subs, n_timepoints = ERP_diff_category_region_window.shape
                
    return values, n_subs, n_timepoints

def plot_raincloud(df):
    # Now with the group as hue
    pal = "Set2"
    sigma = .2
    ort = "v"
    dx = "Region"; dy = "Diff_amp"; dhue = "Condition"
    f, ax = plt.subplots(figsize=(12, 6))

    ax=pt.RainCloud(x = dx, y = dy, hue=dhue, data = df, palette = pal, bw = sigma, width_viol = .5, point_size = 1, width_box = .2,
                    ax = ax, orient = ort , alpha = .65, dodge = True, pointplot = True, move = 0)
    ax.set_ylabel('Absolute Difference amplitude ($\mu$V)', fontsize=20)
    # Hide x-axis labels
    ax.set_xlabel("", fontsize=14)  # Removes the x-axis label
    ax.tick_params(axis='x', labelsize=20)      # Adjust x-axis tick label font size
    ax.tick_params(axis='y', labelsize=14)      # Adjust y-axis tick label font size
    # Adjust the legend size and position inside the plot
    # Get the handles and labels from the current legend
    handles, labels = ax.get_legend_handles_labels()

    # Filter out duplicates: Keep only the first occurrence of each label
    filtered_handles = []
    filtered_labels = []
    seen_labels = set()

    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            filtered_handles.append(handle)
            filtered_labels.append(label)
            seen_labels.add(label)
        
        # Stop after adding two unique conditions
        if len(filtered_labels) == 2:
            break

    # Create the custom legend with only two unique conditions
    ax.legend(filtered_handles, filtered_labels, fontsize=20, loc='upper right', bbox_to_anchor=(1.01, 1.35), frameon=True)
    plt.title("N400 (300 - 500 ms)",  fontsize=22)
    output_folder = f'photo/raincloud/{group}'
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(f'{output_folder}/difference_amplitude_distribution.png', dpi=500, bbox_inches='tight')

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Word class analysis')

    # Add parameters to the parser
    parser.add_argument('-category', type=str, help='Specify the word group')
    parser.add_argument('-start_window', type=int, help='Specify the beginning of time window')
    parser.add_argument('-end_window', type=int, help='Specify the end of time window')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    group = args.category
    start_window = args.start_window 
    end_window = args.end_window

    time_window = get_time_window(start_window, end_window)
    values, n_subs, n_timepoints = prepare_data(group, time_window)
    data = {
    'Region': ['Left Hemisphere'] * 2 * n_subs * n_timepoints + ['Midlines'] * 2 * n_subs * n_timepoints + ['Right Hemisphere'] * 2 * n_subs * n_timepoints,
    'Condition': (['Content/Cloze'] * n_subs * n_timepoints + ['Function/Cloze'] * n_subs * n_timepoints) * 3 ,
    'Diff_amp':  np.concatenate(values)
        }
    df = pd.DataFrame(data)

    plot_raincloud(df)

