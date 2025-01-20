
import argparse

import numpy as np

from utils.analysis_helpers import prepare_data_word_class
from utils.eeg_helpers import get_channel_name_ids
from mne.stats import permutation_cluster_1samp_test, fdr_correction

def create_data(word_type, name_region):
    ## Region
    indices_region, region_channel_names =  get_channel_name_ids(name_region)
    EEG_data_class, labels_class = prepare_data_word_class(word_type)
    EEG_data_region = EEG_data_class[:,:, indices_region, :] * 1e6
    
    # EEG_high_cloze will contain columns of data where the corresponding label is 0
    EEG_high_cloze = EEG_data_region[:, labels_class[0] == 0, :, :]
    EEG_high_cloze_400 = EEG_high_cloze[:, :, :, 400:]

    # EEG_low_cloze will contain columns of data where the corresponding label is 1
    EEG_low_cloze = EEG_data_region[:, labels_class[0] == 1, :, :]
    EEG_low_cloze_400 = EEG_low_cloze[:, :, :, 400:]

    # Generate event-related potential events
    ERP1 = np.average(EEG_high_cloze_400, axis=1)
    ERP2 = np.average(EEG_low_cloze_400, axis=1)

    print("Information of EEG data", EEG_data_class.shape)
    print("Information of ERP with HIGH cloze (300-500)", ERP1.shape)
    print("Information of ERP with LOW cloze (300-500)", ERP2.shape)

    return ERP1, ERP2

def run_permutation_test(ERP1, ERP2, name_region):
    n_subjects, n_sensors, n_timepoints = ERP1.shape
    indices_region, region_channel_names =  get_channel_name_ids(name_region)
    # Calculate the difference between the two conditions
    data_diff = ERP1 - ERP2

    # Reshape the data_diff for cluster-based permutation testing
    data_diff = data_diff.reshape(n_subjects, n_sensors * n_timepoints)

    # Perform the cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        data_diff, n_permutations=10000, threshold=0.05, tail=0, n_jobs=1, seed=42, 
        adjacency=None, out_type='mask')  # Ensure to use the appropriate adjacency matrix if considering sensor adjacency
    
    # Apply FDR correction to the cluster p-values
    _, p_vals_fdr = fdr_correction(cluster_p_values, alpha=0.05)

    # Reshape the clusters and p-values back to sensor-time dimensions
    T_obs = T_obs.reshape(n_sensors, n_timepoints)

    # Create an array to store p-values for each sensor and time point
    p_vals = np.ones((n_sensors, n_timepoints))

    # Create a list to store channel names involved in significant clusters
    significant_channels = []

    # Assign p-values to each sensor and time point based on cluster results
    for cluster, p_val in zip(clusters, cluster_p_values):
        if p_val < 0.05:
            start_idx, stop_idx = cluster[0].start, cluster[0].stop
            for flat_idx in range(start_idx, stop_idx):
                sensor_index = flat_idx // n_timepoints
                timepoint = flat_idx % n_timepoints
                channel_name = region_channel_names[sensor_index]
                if channel_name not in significant_channels:
                    significant_channels.append(channel_name)

    # Print or return the significant channels
    print(name_region.capitalize())
    print("Channels in significant clusters using p_vals:", significant_channels)
    
    significant_channels_x = []
    # Assign p-values to each sensor and time point based on cluster results
    for cluster, p_val in zip(clusters, p_vals_fdr):
        if p_val < 0.05:
            start_idx, stop_idx = cluster[0].start, cluster[0].stop
            for flat_idx in range(start_idx, stop_idx):
                sensor_index = flat_idx // n_timepoints
                timepoint = flat_idx % n_timepoints
                channel_name = region_channel_names[sensor_index]
                if channel_name not in significant_channels_x:
                    significant_channels_x.append(channel_name)

    # Print or return the significant channels
    print(name_region.capitalize())
    print("Channels in significant clusters using FDR:", significant_channels_x)

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Extract the optimal channels in the 300-500ms window')
    # Add parameters to the parser
    parser.add_argument('-category', type=str, help='Specify the word type e.g, NOUN, VERB, ADJ, ADV, PRON, AUX, ADP, DET')
    parser.add_argument('-region', type=str, help='Specify the brain region')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    word_type = args.category
    name_region = args.region

    ERP1, ERP2 = create_data(word_type, name_region)

    run_permutation_test(ERP1, ERP2, name_region)


