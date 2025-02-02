
import argparse
import pickle
import numpy as np
from utils.analysis_helpers import prepare_data_word_class
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
from utils.config import CHANNEL_NAMES, INTERVALS, T_MIN, best_clusters
from utils.plot_helpers import str2bool

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

def prepare_data_diff(category, optimal=False):
    '''
        Import data
    '''

    EEG_data, labels = prepare_data_word_class(category)
    if optimal:   
        best_channels = best_clusters[f'{category}_selected_chans']
        best_indices = [i for i, value in enumerate(CHANNEL_NAMES) if value in(best_channels)]
    else:
        best_indices = [i for i, value in enumerate(CHANNEL_NAMES)]

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

    return ERP_diff_category_region_window

def prepare_data_for_cohendz(category, time_window, optimal):
    '''
        Import data
    '''

    ERP_diff_category_region_window = prepare_data_diff(category, optimal)        
    print("ERP_diff_category_region_window shape: ", ERP_diff_category_region_window.shape)
    

    # Content accuracies
    with open(f'decoding_data/word_class/{category}/word_class_all_accuracies.pkl', 'rb') as f:
        content_accuracies = pickle.load(f)
    # N400 window - Content words
    content_accuracies_window = content_accuracies[:,time_window[0]:time_window[1]]
    print("SVM-based decoding accuracies at N400 window shape: ", content_accuracies_window.shape)

    return ERP_diff_category_region_window, content_accuracies_window

def calculate_cohen_dz(data, chance):
    
    # Step 1: Compute the mean difference
    mean_diff = np.mean(data, axis=0) - chance
    
    # Step 2: Calculate the standard deviation
    std_diff = np.std(data, axis=0, ddof=1)
    
    # Step 3: Compute Cohen's dz
    dz = abs(mean_diff / std_diff)
    
    #Note: Using rpTtest to get the same result
    return dz

def run_bootstrap(data, N, state=42, confidence_level=0.95):
    # Set the random seed
    np.random.seed(state)
    
    # Bootstrap to calculate the 95% confidence intervals
    n_bootstraps = N  # Number of bootstrap samples
    
    # Bootstrap function to calculate the mean
    def bootstrap_mean(data):
        return np.mean(data)
    
    # Perform bootstrapping
    bootstrap_res = bootstrap(
        (data,),
        bootstrap_mean,
        confidence_level=confidence_level,
        n_resamples=n_bootstraps,
        method='percentile'
    )

    # Extract the bootstrap results
    bootstrap_mean_dz = np.mean(data)  # Mean of the original data
    # bootstrap_lower_ci = bootstrap_res.confidence_interval.low
    # bootstrap_upper_ci = bootstrap_res.confidence_interval.high
    sem_dz = np.std(bootstrap_res.bootstrap_distribution, ddof=1)  # Standard error
    
    return bootstrap_mean_dz, sem_dz

def calculate_dz_for_viz(ERP_diff, accuracies, chance_diff=0, chance_decoding=0.5, N=10000):
    # For ERPs
    n_subs = ERP_diff.shape[0]
    N = 10000
    cohen_dz_erp = []
    for sub in range(n_subs):
        # Step 1: Compute the difference wave for each subject
        difference_wave = ERP_diff[sub]

        # Calculate Cohen's dz
        dz  = calculate_cohen_dz(difference_wave, chance=chance_diff)
        cohen_dz_erp.append(dz)

    # Bootstrap to calculate the 95% confidence intervals
    bootstrap_mean_dz_erp, sem_dz_erp  = run_bootstrap(cohen_dz_erp, N)
    
    # For decoding technique
    cohen_dz_decode = []
    for sub in range(n_subs):
        # Step 1: Get the accuracies for each subject
        accuracy  = accuracies[sub]

        # Calculate Cohen's dz
        dz  = calculate_cohen_dz(accuracy, chance=chance_decoding)
        cohen_dz_decode.append(dz)

    # Bootstrap to calculate the 95% confidence intervals
    bootstrap_mean_dz_decode, sem_dz_decode   = run_bootstrap(cohen_dz_decode, N)
    
    return bootstrap_mean_dz_erp, sem_dz_erp, bootstrap_mean_dz_decode, sem_dz_decode

def viz_cohen_dz(bootstrap_mean_dz_erp, sem_dz_erp, bootstrap_mean_dz_decode, sem_dz_decode, ax, labels=['ERP', 'SVM']):
    # Means and SEMs
    means = [bootstrap_mean_dz_erp, bootstrap_mean_dz_decode]
    sems = [sem_dz_erp, sem_dz_decode]

    # Bar positions
    x = np.arange(len(labels))

    # Plot bars with error bars
    bars = ax.bar(x, means, yerr=sems, capsize=10, color=['crimson', 'mediumseagreen'])

    # Adding labels and title
    ax.set_ylabel("Cohen's dz", fontsize=25)
    # ax.set_title("Effect Size Comparison (High vs. Low Cloze)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Set y-axis limit similar to your previous examples
    ax.set_ylim(0, 5)

    # Customize the y-axis to match the sample plot style
    ax.yticks = [0, 1, 2, 3, 4, 5]

    # Increase the font size and weight for better readability
    ax.tick_params(axis='y', which='major', labelsize=20, width=2)
    ax.tick_params(axis='x', which='major', labelsize=24, width=2)


    # Remove the top and right spines (the box around the plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optionally, you can also thicken the left and bottom spines
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Word class analysis')

    # Add parameters to the parser
    parser.add_argument('-category', type=str, help='Specify the word group')
    parser.add_argument('-optimal', type=str2bool, nargs='?', const=True, default=False, help='Get optimal electrodes for ERP analysis')
    parser.add_argument('-start_window', type=int, help='Specify the beginning of time window')
    parser.add_argument('-end_window', type=int, help='Specify the end of time window')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parameter values
    word_type = args.category
    optimal = args.optimal
    start_window = args.start_window 
    end_window = args.end_window

    # Load data
    time_window = get_time_window(start_window, end_window)
    ERP_diff_category_region_window, content_accuracies_window = prepare_data_for_cohendz(word_type, time_window, optimal)
    
    # Cohen's dz calculation
    bootstrap_mean_dz_group_erp, sem_dz_group_erp, bootstrap_mean_dz_group_svm, sem_dz_group_svm = calculate_dz_for_viz(ERP_diff_category_region_window, content_accuracies_window)

    # Visualise Cohen's dz
    fig, ax = plt.subplots(figsize=(6, 8))
    viz_cohen_dz(bootstrap_mean_dz_group_erp, sem_dz_group_erp, bootstrap_mean_dz_group_svm, sem_dz_group_svm, ax)
    fig.savefig(f'photo/{word_type}/{optimal}_cohen_dz_analysis.png', dpi=500, bbox_inches='tight')

    print("Running Successfully!")
    print(f"Result is saved in photo/{word_type}/{optimal}_cohen_dz_analysis.png")
    
