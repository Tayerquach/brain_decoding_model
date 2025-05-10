import argparse
import os
import pickle
import numpy as np
from utils.plot_helpers import plot_decoding_acc_tbyt, str2bool
from utils.techniques import classification_decoding_kfold
from utils.eeg_helpers import get_channel_name_ids
from utils.analysis_helpers import prepare_data_word_class
from utils.config import INTERVALS, T_MAX, T_MIN, best_clusters, CHANNEL_NAMES
import matplotlib.pyplot as plt


if __name__ == '__main__': 
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Word class analysis - Decoding')

    # Add parameters to the parser
    parser.add_argument('-category', type=str, help='Specify the word type e.g, NOUN, VERB, ADJ, ADV, PRON, AUX, ADP, DET, content, function')
    parser.add_argument('-region', type=str, default='best', help='Specify the brain region')
    parser.add_argument('-permutation', type=str2bool, nargs='?', const=True, default=False, help='Conduct cluster-based permutation test or not')
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

    # Check for missing arguments
    missing_args = [arg for arg in ['category', 'region', 'permutation', 'clusterp', 'n_iter'] if getattr(args, arg) is None]

    if missing_args:
        missing_list = ', '.join(f'--{arg}' for arg in missing_args)
        raise ValueError(f"Missing required argument(s): {missing_list}")

    if name_region not in ['left_hemisphere', 'right_hemisphere', 'midlines', 'all', 'best']:
        raise ValueError("Invalid brain region. Please specify a valid brain region")
    
    allowed_category =  ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'AUX', 'ADP', 'DET', 'content', 'function']
    if word_type not in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'AUX', 'ADP', 'DET', 'content', 'function']:
        raise ValueError(f"Invalid argument: '{word_type}'. Expected one of: {allowed_category}")
    
    EEG_data, labels = prepare_data_word_class(word_type)

    if name_region == 'best':
        region_channels = best_clusters[f'{word_type}_selected_chans']
        indices = [i for i, value in enumerate(CHANNEL_NAMES) if value in(region_channels)]
    else:
        indices, region_channels =  get_channel_name_ids(name_region)
    EEG_data_region = EEG_data[:,:, indices, :] * 1e6

    # Save the arrays to a pickle file
    output_data_folder = f'decoding_data/word_class/{word_type}'
    # Check if the folder exists, if not create it
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)
    file_path = output_data_folder + f'/word_class_{name_region}_accuracies.pkl'
    # Check if the file does not exist
    if not os.path.exists(file_path):
        # Decoding EEG data and validating results using K-fold cross validation
        accuracies = classification_decoding_kfold(EEG_data_region, labels, n_class=2, time_win=1, time_step=1, n_folds=5, n_repeats=100, normalization=True)
        # Save accuracy
        with open(output_data_folder + f'/word_class_{name_region}_accuracies.pkl', 'wb') as file:
            pickle.dump(accuracies, file)

    else:
        with open(output_data_folder + f'/word_class_{name_region}_accuracies.pkl', 'rb') as file:
            accuracies = pickle.load(file)


    # Create the region name
    region = name_region.replace('_', ' ')
    region = region.title()

    # Save figure
    output_result_folder = f"photo/{word_type}/classification_decoding/{name_region}"
    # Check if the folder exists, if not create it
    if not os.path.exists(output_result_folder):
        os.makedirs(output_result_folder)

    # Plot decoding accuracies time by time
    fig = plt.figure(figsize=(14, 8))
    title = f'EEG Decoding between high and low cloze for {word_type.capitalize()} in {region}'
    plot_decoding_acc_tbyt(accuracies, start_time=T_MIN, end_time=T_MAX, time_interval=INTERVALS, chance=0.5, p=p_value, 
                           cluster_permutation=cluster_permutation, clusterp=clusterp, iter=n_iter,
                           stats_time=[T_MIN, T_MAX], title=None, color='g', xlim=[-0.1, 0.5], ylim=[0.4, 0.7], avgshow=True)
    plt.show()
    fig.savefig(output_result_folder + f'/decoding_eeg_in_{name_region}_for_{word_type}.jpg')

    print(f"\nData was saved in {output_data_folder}")
    print(f"\nResults were saved in {output_result_folder}")