import argparse
from collections import Counter
import os
from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
from utils.eeg_helpers import add_part_of_speech, create_electrodes_data
from utils.config import CONTENT_TYPES, EEG_PATH, FUNCTION_TYPES, HIGH_CLOZE, LOW_CLOZE, N_CHANNELS, N_TOPICS, NUM_TIME_POINTS, PREPROCESSED_FOLDER

import pandas as pd
import mne

from scipy.stats import ks_2samp

def sample_to_match_distribution(baseline, other_p_cloze, other_ids):
    np.random.seed(42)  # For reproducibility
    best_sample = None
    best_ids = None
    best_ks_stat = np.inf
    sample_size = len(baseline)
    
    for _ in tqdm(range(5000), position=0, leave=True):  # Number of iterations
        indices = np.random.choice(len(other_p_cloze), sample_size, replace=False)
        sample = other_p_cloze[indices]
        ids = other_ids[indices]
        
        ks_stat, _ = ks_2samp(baseline, sample)
        
        if ks_stat < best_ks_stat:
            best_ks_stat = ks_stat
            best_sample = sample
            best_ids = ids
            
            if best_ks_stat == 0:
                break
    
    return best_sample, best_ids
def convert_data_to_dict():
    """
    Converts EEG-based reading data into a dictionary: EEG data, metadata, and labels.

    Parameters
    ----------
    data : 

    Returns
    -------
    dict
        The dictionary of data.
    """

    # Read epochs
    n_channels = N_CHANNELS
    n_timepoints = NUM_TIME_POINTS
    data_path = EEG_PATH
    folder = PREPROCESSED_FOLDER
    subject_path = Path(data_path) / folder
    file_name = 'preprocessed_epoch.fif'
    subject_ids = [f for f in os.listdir(subject_path) if not f.startswith('.')]
    excluded_subject = ['QPF42','USQ95']
    # Remove values from excluded_subject that are present in subject_ids
    subject_ids = [sub_id for sub_id in subject_ids if sub_id not in excluded_subject]
    n_topics = N_TOPICS
    # Initialize the dictionary
    subjects_dict = {}

    for subject_id in tqdm(subject_ids):
        df = pd.DataFrame()
        metadata = pd.DataFrame()
        for topic_id in tqdm(range(n_topics)):
            article_path = 'article_' + str(topic_id)
            epochs_path = subject_path / subject_id / article_path / file_name
            epochs = mne.read_epochs(epochs_path)
            metadata = epochs.metadata # Access metadata
            # Add a pos column
            df_pos = add_part_of_speech(topic_id) 
            metadata_df = pd.merge(metadata, df_pos, left_on='word', right_on='word_id', how='left').drop(columns=['word_id'])
            # Reassign the modified metadata to the epochs object
            epochs.metadata = metadata_df
            
            # Filter epochs
            epochs = epochs[epochs.metadata['pos'].isin(CONTENT_TYPES + FUNCTION_TYPES)
                        & epochs.metadata['level'].isin(HIGH_CLOZE + LOW_CLOZE)]
            db, temp = create_electrodes_data(epochs)
            temp['word_prefix'] = temp['word'].str.split('_').str[0]
            df = pd.concat([df, db], ignore_index=True)
            metadata = pd.concat([temp, metadata], ignore_index=True)
        # Convert the DataFrame to a NumPy array
        data_array = df.iloc[:,2:].to_numpy()

        # Reshape the array to (100, 600, n_channels)
        reshaped_array = data_array.reshape(-1, n_timepoints, n_channels)

        # Transpose the array to get the shape (_, n_channels, 600)
        EEG_data = reshaped_array.transpose(0, 2, 1)

        word_list = df.word.unique()

        # Convert the 'word' column to a categorical type with the custom order
        metadata['word'] = pd.Categorical(metadata['word'], categories=word_list, ordered=True)

        # Sort the DataFrame according to the custom order
        metadata = metadata.sort_values('word').reset_index(drop=True)

        word_ids = metadata.WordID.values

        # Get indices where 'WordID' is empty
        empty_indices = metadata[metadata['WordID'] == ''].index.tolist()
        # Create a mask to select all indices except the ones to remove
        mask_EEG = np.ones(EEG_data.shape[0], dtype=bool)
        mask_EEG[empty_indices] = False

        mask_words = np.ones(word_ids.shape[0], dtype=bool)
        mask_words[empty_indices] = False

        # Apply the mask to the array to remove the specified indices along axis 1
        EEG_data = EEG_data[mask_EEG, :, :]
        word_ids = word_ids[mask_words]

        # Remove rows where 'WordID' is an empty string
        metadata = metadata[metadata['WordID'] != '']

        # Reset the index
        metadata = metadata.reset_index(drop=True)

        '''
            Labels
        '''

        words_labels =  metadata['WordID'].values
        content_word_ids = metadata[metadata['pos'].isin(CONTENT_TYPES)]['WordID'].to_list()
        function_word_ids = metadata[metadata['pos'].isin(FUNCTION_TYPES)]['WordID'].to_list()

        # Create labels based on the presence of words in content_words and function_words
        labels = np.array([0 if word in content_word_ids else 1 if word in function_word_ids else -1 for word in words_labels])


        # Check mismatch in Cloze level and Part of Speech
        # We will remove words having different cloze levels or different part of speecj
        # Initialize the dictionary
        prefix_to_indices = {}

        # Group by 'word_prefix'
        grouped = metadata.groupby('word_prefix')

        # Iterate over the groups and collect indices
        for word_prefix, group in grouped:
            prefix_to_indices[word_prefix] = group.index.tolist()


        # Check mismatch in cloze (or Part of Speech)
        removed_indices = []
        for key, value in prefix_to_indices.items():
            x = labels[value] 
            all_same = np.all(x == x[0])
            if all_same == False:
                # If labels 0 and 1 equals => remove all indices, otherwise remove the indices having the least frequent value
                words_indices = prefix_to_indices[key]
                # Find the least frequent value
                array = labels[words_indices]
                counter = Counter(array)
                if counter[1] == counter[0]:
                    removed_indices_temp = words_indices
                else:   
                    least_frequent_value = min(counter, key=counter.get)
                    least_frequent_indices = np.where(array == least_frequent_value)[0]
                    removed_indices_temp = [words_indices[i] for i in least_frequent_indices]
                removed_indices += removed_indices_temp

        # Create a mask to select all indices except the ones to remove
        mask_EEG = np.ones(EEG_data.shape[0], dtype=bool)
        mask_EEG[removed_indices] = False

        mask_labels = np.ones(labels.shape[0], dtype=bool)
        mask_labels[removed_indices] = False

        # Apply the mask to the array to remove the specified indices along axis 1
        EEG_data_filter = EEG_data[mask_EEG, :, :]
        labels_filter = labels[mask_labels]
        # Remove rows with specified indices
        metadata_filter = metadata.drop(removed_indices)
        metadata_filter = metadata_filter.reset_index(drop=True)
        
        # Loop through each subject_id and add the data
        subjects_dict[subject_id] = {
            'EEG_data': EEG_data_filter,
            'label': labels_filter,
            'metadata': metadata_filter
        }

    return subjects_dict

def prepare_data(subjects_dict, subject_ids, word_type):
    if word_type == 'content':
        temp = CONTENT_TYPES
    elif word_type == 'function':
        temp = FUNCTION_TYPES
    elif word_type in (CONTENT_TYPES + FUNCTION_TYPES):
        # NOUN, VERB, ADJECTIVE, ADVERB, PRONOUN, ETC.
        temp = [word_type] 
    else:
        raise TypeError("Please enter a proper word type!")
    
    counts_cloze_all = []
    for subject_id in subject_ids:
        metadata = subjects_dict[subject_id]['metadata']
        metadata = metadata[metadata['pos'].isin(temp)]
        counts_cloze = Counter(metadata['level'].values)
        counts_cloze_all.append(dict(counts_cloze)) 

    # Initialize variables to store the minimum values and corresponding subject_ids
    min_high_value = float('inf')
    min_high_subject = None
    min_low_value = float('inf')
    min_low_subject = None

    # Iterate through the data to find the minimum high and low values and their corresponding subject_ids
    for i, entry in enumerate(counts_cloze_all):
        if 'high' in entry.keys() and entry['high'] < min_high_value:
            min_high_value = entry['high']
            min_high_subject = subject_ids[i]
        
        if 'low' in entry.keys() and entry['low'] < min_low_value:
            min_low_value = entry['low']
            min_low_subject = subject_ids[i]

    # Print the results
    print(f"Subject with the least high value: {min_high_subject} (high: {min_high_value})")
    print(f"Subject with the least low value: {min_low_subject} (low: {min_low_value})")
    
    # Create baseline data
    metadata_content_high = subjects_dict[min_high_subject]['metadata']
    metadata_content_high = metadata_content_high[(metadata_content_high['pos'].isin(temp)) & (metadata_content_high['level'] == 'high')]

    metadata_content_low = subjects_dict[min_low_subject]['metadata']
    metadata_content_low = metadata_content_low[(metadata_content_low['pos'].isin(temp)) & (metadata_content_low['level'] == 'low')]

    baseline_condition1 = metadata_content_high['p_cloze'].values
    baseline_condition2 = metadata_content_low['p_cloze'].values

    all_EEG_subs = []
    for subject_id in tqdm(subject_ids, position=0, leave=True):
        print(subject_id)
        # Get EEG_data
        EEG_data = subjects_dict[subject_id]['EEG_data']
        # Get metadata
        metadata = subjects_dict[subject_id]['metadata']
        # if word_type is not None:
        # For condition 1
        metadata_con1 = metadata[(metadata['pos'].isin(temp)) & (metadata['level'] == 'high')]
        # For condition2
        metadata_con2  = metadata[(metadata['pos'].isin(temp)) & (metadata['level'] == 'low')]
        
        # Create sample and ids for condition1
        sample_con1 = metadata_con1['p_cloze'].values
        ids_con1 = metadata_con1['WordID'].values
        # Create sample and ids for condition2
        sample_con2 = metadata_con2['p_cloze'].values
        ids_con2 = metadata_con2['WordID'].values
        
        # Get the sample based on distribution
        best_sample_con1, best_ids_con1 = sample_to_match_distribution(baseline_condition1, sample_con1, ids_con1)
        best_sample_con2, best_ids_con2 = sample_to_match_distribution(baseline_condition2, sample_con2, ids_con2)
        
        ## Get EEG data and labels
        indices_con1 = metadata[metadata['WordID'].isin(best_ids_con1)].index.tolist()
        indices_con2 = metadata[metadata['WordID'].isin(best_ids_con2)].index.tolist()
        EEG_data_con1 = EEG_data[indices_con1, :, :]
        EEG_data_con2  = EEG_data[indices_con2, :, :]
        new_EEG_data = np.concatenate((EEG_data_con1, EEG_data_con2), axis=0)
        all_EEG_subs.append(new_EEG_data)
    
    EEG_data = np.stack(all_EEG_subs, axis=0)
    labels = np.concatenate((np.zeros(len(baseline_condition1), dtype=int), np.ones(len(baseline_condition2), dtype=int)))
    labels = np.tile(labels, (EEG_data.shape[0], 1))
        
    return EEG_data, labels

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Data Preparation')

    # Add parameters to the parser
    parser.add_argument('-word_type', type=str, help='Specify the categories')

     # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    word_type = args.word_type

    # Convert data to dictionary
    subjects_dict = convert_data_to_dict()

    # Get the subject_ids
    subject_ids = list(subjects_dict.keys())
    # Prepare data
    EEG_data, labels = prepare_data(subjects_dict, subject_ids, word_type)

    # Save the data
    saved_path = 'results'
    output_folder = Path(saved_path) / word_type
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save labels
    with open(output_folder / 'labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

    # Save EEG_data
    with open(output_folder / 'EEG_data.pkl', 'wb') as f:
        pickle.dump(EEG_data, f)

    print("Data saved successfully!")

