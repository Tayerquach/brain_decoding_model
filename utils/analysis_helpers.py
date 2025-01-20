import pickle

import numpy as np


def prepare_data_word_class(word_type):
    '''
    Import data
    '''
    # Read labels to a pickle file
    with open(f'results/{word_type}/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    # Read EEG_data to a different pickle file
    with open(f'results/{word_type}/EEG_data.pkl', 'rb') as f:
        EEG_data = pickle.load(f)

    print("Shape of EEG original data: ", EEG_data.shape)

    return EEG_data, labels

def prepare_data_cloze(cloze_type):
    '''
        Import data
    '''

    # Read labels for content words
    with open(f'results/content/labels.pkl', 'rb') as f:
        labels_content = pickle.load(f)

    # Read EEG_data for content words
    with open(f'results/content/EEG_data.pkl', 'rb') as f:
        EEG_data_content = pickle.load(f)
        
    # Read labels for function words
    with open(f'results/function/labels.pkl', 'rb') as f:
        labels_function = pickle.load(f)

    # Read EEG_data for function words
    with open(f'results/function/EEG_data.pkl', 'rb') as f:
        EEG_data_function = pickle.load(f)

    # Extract type cloze data
    if cloze_type == 'low':
        EEG_data_content = EEG_data_content[:, labels_content[0] == 1, :, :]
        EEG_data_function = EEG_data_function[:, labels_function[0] == 1, :, :]
    elif cloze_type == 'high':
        EEG_data_content = EEG_data_content[:, labels_content[0] == 0, :, :]
        EEG_data_function = EEG_data_function[:, labels_function[0] == 0, :, :]

    EEG_data_new = np.concatenate((EEG_data_content, EEG_data_function), axis=1)
    labels_new = np.concatenate((np.zeros(EEG_data_content.shape[1]), np.ones(EEG_data_function.shape[1], dtype=int)))
    labels_new = np.tile(labels_new, (EEG_data_new.shape[0], 1))

    print("Shape of EEG original data: ", EEG_data_new.shape)

    return EEG_data_new, labels_new