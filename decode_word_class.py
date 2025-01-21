import argparse
import os
import pickle
import numpy as np
from utils.plot_helpers import plot_decoding_acc_tbyt
from utils.techniques import classification_decoding_kfold
from utils.eeg_helpers import get_channel_name_ids
from utils.analysis_helpers import prepare_data_word_class
from utils.config import INTERVALS, T_MAX, T_MIN
import matplotlib.pyplot as plt


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

    if name_region not in ['left_hemisphere', 'right_hemisphere', 'midlines', 'all']:
        raise ValueError("Invalid brain region. Please specify a valid brain region")
    
    if word_type not in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'AUX', 'ADP', 'DET', 'content', 'function']:
        raise ValueError("Invalid word type. Please specify a valid word category")
    
    EEG_data, labels = prepare_data_word_class(word_type)
    indices, region_channels =  get_channel_name_ids(name_region)
    EEG_data_region = EEG_data[:,:, indices, :] * 1e6

    # Decoding EEG data and validating results using K-fold cross validation
    accuracies = classification_decoding_kfold(EEG_data_region, labels, n_class=2, time_win=1, time_step=1, n_folds=5, n_repeats=100, normalization=True)

    # Save the arrays to a pickle file
    output_folder = f'decoding_data/word_class/{word_type}'
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_folder + f'/word_class_{name_region}_accuracies.pkl', 'wb') as file:
        pickle.dump(accuracies, file)

    # Create the region name
    region = name_region.replace('_', ' ')
    region = region.title()

    # Save figure
    output_folder = f"photo/{word_type}/classification_decoding/{name_region}"
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Plot decoding accuracies time by time
    fig = plt.figure(figsize=(14, 8))
    title = f'EEG Decoding between high and low cloze for {word_type.capitalize()} in {region}'
    plot_decoding_acc_tbyt(accuracies, start_time=t_min, end_time=t_max, time_interval=step, chance=0.5, p=p_value, 
                           cluster_permutation=cluster_permutation, clusterp=clusterp, iter=n_iter,
                           stats_time=[t_min, t_max], title=None, color='g', xlim=[-0.1, 0.5], ylim=[0.4, 0.7], avgshow=True)
    plt.show()
    fig.savefig(output_folder + f'/decoding_eeg_in_{name_region}_for_{word_type}.jpg')

    print("Done!")