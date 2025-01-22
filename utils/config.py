N_CHANNELS = 32
T_MIN = -0.1
T_MAX = 0.5
INTERVALS = 0.001
NUM_TIME_POINTS = int((T_MAX - T_MIN) / INTERVALS)

EEG_PATH = 'data/EEG_data'
PREPROCESSED_FOLDER = 'preprocessed'
N_TOPICS = 5

CONTENT_TYPES = ['NOUN','VERB','ADJ','ADV']
FUNCTION_TYPES = ['PRON','AUX','DET','ADP']

HIGH_CLOZE = ['high']
LOW_CLOZE = ['low']

CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'T7', 'CP5', 'FC6', 'T8', 'CP6', 'FT9', 'FT10', 'FC1', 'FC2', 
'C3', 'Cz', 'C4', 'P3', 'P7', 'CP1', 'CP2', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'TP9', 'TP10']
LEFT_HEMISPHERE = ['Fp1', 'F7', 'F3', 'FC5', 'T7', 'CP5', 'FT9', 'FC1', 'C3', 'P3', 'P7', 'CP1', 'O1', 'TP9']
RIGHT_HEMISPHERE = ['Fp2', 'F4', 'F8', 'FC6', 'T8', 'CP6', 'FT10', 'FC2', 'C4', 'CP2', 'P4', 'P8', 'O2', 'TP10']
MIDLINES = ['Fz', 'Cz', 'Pz', 'Oz']
best_clusters = {
    "content_selected_chans": ['P3', 'P7', 'CP1', 'Fp1', 'F7', 'F3', 'FC5', 'T7', 'CP5', 'Cz', 'Pz', 'Oz', 'Fz', 'T8', 'CP6'],
    "function_selected_chans": ['Fp1', 'F7', 'F3', 'FC5', 'T7', 'CP5', 'Cz', 'Pz', 'Oz', 'Fp2', 'F4'],
    "NOUN_selected_chans": ['P3', 'P7', 'CP1', 'O1', 'Cz', 'Pz', 'Oz', 'Fz', 'C4', 'CP2', 'P4'],
    "VERB_selected_chans": ['P3', 'P7', 'CP1', 'O1', 'FC5', 'T7', 'CP5', 'Cz', 'Pz', 'Oz', 'T8', 'CP6'],
    "DET_selected_chans": ['FT9', 'FC1', 'C3', 'P3', 'P7', 'CP1', 'Fp1', 'F7', 'F3', 'FC5', 'T7', 'CP5', 'Cz', 'Pz', 'FT10', 'FC2', 'C4', 'CP2', 'Fp2', 'F4', 'F8'],
    "PRON_selected_chans": ['Fp1', 'F7', 'F3', 'FC5', 'T7', 'CP5', 'FT9', 'FC1', 'C3', 'P3', 'P7', 'CP1', 'O1', 'TP9', 'Fz', 'Cz', 'Pz', 'Oz', 'T8', 'CP6'],
}