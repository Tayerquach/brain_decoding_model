import pickle
import numpy as np
import pandas as pd
from utils.config import CHANNEL_NAMES, INTERVALS, N_CHANNELS, N_TOPICS, T_MAX, T_MIN, LEFT_HEMISPHERE, RIGHT_HEMISPHERE, MIDLINES
from ipa.preprocessing.tokenizers.spacy_tokenizer import SpacyTokenizer  # For tokenizing text using Spacy

def add_part_of_speech(topic_id):
    # Initialize metadata as an empty dictionary
    metadata_dict = {
        'word_id': [],
        'pos': []
    }

    # Part of Speech
    spacy_tokenizer = SpacyTokenizer(language="en", return_pos_tags=True, return_lemmas=True)
    
    with open(f'data/article/article_{topic_id}.pkl', 'rb') as f:
        word_stimulus = pickle.load(f)
    stimuli = [w + f'_{topic_id}_{i}'for i, w in enumerate(word_stimulus)]

        # Append word_id to metadata_dict
    metadata_dict['word_id'].extend(stimuli)
    
    # Part of Speech
    sentences = np.load(f'data/article/sentences_article_{topic_id}.npy')
    word_pos = []
    sentences = [str(element) for element in sentences]
    for sen in sentences:
        tokenized = spacy_tokenizer(sen)
        for word in tokenized:
            if (word.pos not in ['PUNCT']) & (word.text not in ["\'s"]):
                word_pos.append(word.pos)
    # Append pos tags to metadata_dict
    metadata_dict['pos'].extend(word_pos)

    # Convert metadata_dict to a DataFrame after the loop
    metadata_df = pd.DataFrame(metadata_dict)
    
    return metadata_df

def create_electrodes_data(epochs):
    """
    This code will change epochs data to a dataframe.
    Parameters
    ----------
    epochs: Epochs object

    Returns
    -------
    merged_df: EEG dataframe
    metadata: dataframe
    """
    tmin = T_MIN
    tmax = T_MAX
    step = INTERVALS
    n_channels = N_CHANNELS

    data = epochs.get_data()
    # Data started from 0 (-.2) to 1200 (1.0)
    start = int((tmin - epochs.tmin) * 1000)
    end = int((tmax - epochs.tmin) * 1000)
    custom_data = data[:,:,start:end]
    metadata = epochs.metadata
    words = metadata['word'].values
    channel_names = epochs.ch_names
    # Create times from -0.1 to 0.5
    times = np.arange(tmin, tmax, step)
    
    db = pd.DataFrame({
        'word': words
    })
    # Create a DataFrame from the list
    times_df = pd.DataFrame({'times': times})

    # Perform a cross merge
    db = pd.merge(db.assign(key=1), times_df.assign(key=1), on='key').drop('key', axis=1)

    #Add values for each electrode
    # Swap the dimensions
    swapped_data = np.swapaxes(custom_data, 0, 1)
    # Reshape the array
    reshaped_data = swapped_data.reshape(n_channels, -1)
    # Transpose the reshaped data to have 32 columns
    transposed_data = reshaped_data.T
    # Convert to DataFrame
    d_channels = pd.DataFrame(transposed_data, columns=channel_names)
    # Merge based on index
    merged_df = pd.merge(db, d_channels, left_index=True, right_index=True)
    
    return merged_df, metadata

def get_channel_name_ids(name):
    # Define mappings for regions
    region_map = {
        "all": CHANNEL_NAMES,
        "left_hemisphere": LEFT_HEMISPHERE,
        "right_hemisphere": RIGHT_HEMISPHERE,
        "midlines": MIDLINES,
    }

    # Handle special cases
    if name == "all":
        indices = list(range(len(CHANNEL_NAMES)))
        region_channel_names = CHANNEL_NAMES.copy()
    else:
        # Get the target region or default to the specific channel name
        target_region = region_map.get(name, [name])
        indices = [i for i, value in enumerate(CHANNEL_NAMES) if value in target_region]
        region_channel_names = [CHANNEL_NAMES[i] for i in indices]

    return indices, region_channel_names