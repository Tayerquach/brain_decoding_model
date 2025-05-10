from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from mne.stats import permutation_cluster_1samp_test, fdr_correction
import numpy as np
from scipy.stats import ttest_rel



"""
    ERPs ANALYSIS TECHNIQUES: STATISTICS
"""
def run_erps_analysis(erp1, erp2, cluster_permutation=False, clusterp=0.05, iter=1000, state=42):
    """
    T-test

    ====
    Statistical technique for ERPs analysis

    Parameters
    ----------
    erp1: a matrix with the shape [n_subs, n_times]
    corresponding to all subjects' ERPs under condition 1
    erp2: a matrix with the shape [n_subs, n_times]
    corresponding to all subjects' ERPs under condition 2

    Returns
    ----------
    p_vals: p values of each time point
    avg1: mean of ERP condition 1  
    err1: corresponding variation (condition 1)
    avg2: mean of ERP condition 2 
    err2: corresponding variation (condition 2)
    """

    # Set the random seed
    np.random.seed(state)

    n_subjects = np.shape(erp1)[0]
    # averaging the ERPs
    avg1 = np.average(erp1, axis=0)
    avg2 = np.average(erp2, axis=0)
    # calcualte the SEM for each time-point
    err1 = np.std(erp1, axis=0, ddof=0)/np.sqrt(n_subjects)
    err2 = np.std(erp2, axis=0, ddof=0)/np.sqrt(n_subjects)

    if cluster_permutation:
        # Calculate the difference between the two conditions
        data_diff = erp1 - erp2

        # Perform the cluster-based permutation test
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            data_diff, n_permutations=iter, threshold=clusterp, tail=0, n_jobs=1, seed=state)
        
        # Apply FDR correction to the cluster p-values
        _, p_vals_fdr = fdr_correction(cluster_p_values, alpha=0.05)

        # Create an array to store p-values for each time point
        p_vals = np.ones(data_diff.shape[1])

        # Assign p-values to each time point based on cluster results
        for cluster, p_val in zip(clusters, cluster_p_values):
            if p_val < clusterp:
                p_vals[cluster] = p_val
    
    else:
        # statistical analysis
        t_vals, p_vals = ttest_rel(erp1, erp2, axis=0)
        # FDR-correction
        # rejects, p_fdr_corrected = fdr_correction(p_vals, alpha=p_threshold)


    return p_vals, avg1, err1, avg2, err2

"""
    EEG DECODING TECHNIQUES
"""
def smooth_1d(x, n=5):

    """
    smoothing for 1-D results

    Parameters
    ----------
    x : array
        The results.
        The shape of x should be [n_sub, n_ts]. n_subs, n_ts represent the number of subjects and the number of
        time-points.
    n : int. Default is 5.
        The smoothing step is n.

    Returns
    -------
    x_smooth : array
        The results after smoothing.
        The shape of x_smooth should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and the number of
        time-points.
    """

    nsubs, nts = np.shape(x)

    x_smooth = np.zeros([nsubs, nts])

    ts1 = int(n / 2)
    ts2 = n - ts1

    for t in range(nts):

        if t >= ts1 and t < (nts - ts1):
            x_smooth[:, t] = np.average(x[:, t - ts1:t + ts2], axis=1)
        elif t < ts1:
            x_smooth[:, t] = np.average(x[:, :t + ts2], axis=1)
        else:
            x_smooth[:, t] = np.average(x[:, t - ts1:], axis=1)

    return x_smooth

def classification_decoding_kfold(data, labels, n_class, time_win=1, time_step=1, n_avg=None, n_folds=5, n_repeats=2, normalization=False, smooth=True, state=42):
    """
    Conduct time-by-time decoding for EEG-like data (cross validation)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
        
    labels : array
        The labels of each trial.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.

    n_avg : int. Default is None.
        The number of trials used to average.
        
    n_class : int. Default is 2.
        The number of categories for classification.
        
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
        
    time_step : int. Default is 5.
        The time step size for each time of decoding.
        
    n_folds : int. Default is 5. 
    The number of folds for validation. k should be at least 2.
    
    n_repeats : int. Default is 2.
        The times for iteration.
        
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
        
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.

        
    state: int. Default is 42
        Setting a random state ensures that the random processes yield the same results every time the code is run.
    
    Returns
    -------
    accuracies : array
        The time-by-time decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts-time_win)/time_step)+1].
    """
    
    # Set the random seed
    np.random.seed(state)
    
    # Check number of subjects
    if np.shape(data)[0] != np.shape(labels)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"
    
    # Check number of epochs
    if np.shape(data)[1] != np.shape(labels)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"
    
    n_subs, n_trials, n_chans, n_tpoints = np.shape(data)
    ncategories = np.zeros([n_subs], dtype=int)
    
    # Check how many categories we need to classify
    for sub in range(n_subs):
        sublabels_set = set(labels[sub].tolist())
        ncategories[sub] = len(sublabels_set)
        
    # Check if the number of categories in each subject are the same
    all_categories_same = np.all(ncategories == ncategories[0])
    if not all_categories_same:
        print("\nThe number each subject's classes in labels are not the same!\n")
        return "Invalid input!"
    
    # Check whether the number of categories is equal to the number of decoding categories
    if n_class != ncategories[0]:

        print(f"\nThe number of categories for decoding ({n_class}) doesn't match ncategories in labels (" + str(ncategories[0]) + ")!\n")

        return "Invalid input!"  
    
    # Categories in labels: E.g., two classes: [0. 1]
    categories = list(sublabels_set)

    # Customise time points based on window slide and steps
    new_n_tpoints = int((n_tpoints-time_win)/time_step) + 1
    
        
    # Customise EEG data after change time points
    avgt_data = np.zeros([n_subs, n_trials, n_chans, new_n_tpoints])

    for t in range(new_n_tpoints):
        avgt_data[:, :, :, t] = np.average(data[:, :, :, t * time_step:t * time_step + time_win], axis=3)
    
    # Initialise decoding accuracies with each time point x sub -> 1 accuracy
    acc = np.zeros([n_subs, new_n_tpoints])

    # The number of times running classifier
    # total = n_subs * n_repeats * new_n_tpoints * n_folds
    
    # Run Clasification for each subject
    # with tqdm(total=total, desc='Processing', leave=True, position=0, bar_format='{l_bar}{bar}| {percentage:3.0f}% / 100% [ETA: {remaining}]') as pbar:
    for sub in tqdm(range(n_subs), position=0, leave=True):

        # Count number of labels of each class. For example, ns = array([50, 100]) and categories = [0, 1], meaning that 50 trials labelling as 0, and 100 trails labelling as 1
        ns = np.zeros([n_class], dtype=int)

        for i in range(n_trials):
            for j in range(n_class):
                if labels[sub, i] == categories[j]:
                    ns[j] = ns[j] + 1
                    
        # Customise number of trials for training.
        if n_avg is None:
            if np.min(ns) < 15:
                minn = np.min(ns)
                n_avg = 1
            else:
                minn = 15 
                n_avg = int(np.min(ns) / 15)
        else:
            minn = int(np.min(ns) / n_avg)        

        subacc = np.zeros([n_repeats, new_n_tpoints, n_folds])

        # All subjects' data is training n_repeats times
        for i in tqdm(range(n_repeats), position=0, leave=True):

            # Initialise EEG data at the ith run
            datai = np.zeros([n_class, minn * n_avg, n_chans, new_n_tpoints])
            # Initialise labels at the ith run
            labelsi = np.zeros([n_class, minn], dtype=int)

            for j in range(n_class):
                labelsi[j] = j

            # Shuffle index
            randomindex = np.random.permutation(np.array(range(n_trials)))

            m = np.zeros([n_class], dtype=int)

            # Configure data and labels according to suffled indices
            for j in range(n_trials):
                for k in range(n_class):
                    if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * n_avg:
                        datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                        m[k] = m[k] + 1

            avg_datai = np.zeros([n_class, minn, n_chans, new_n_tpoints])

            for j in range(minn):
                avg_datai[:, j] = np.average(datai[:, j * n_avg:j * n_avg + n_avg], axis=1)

            x = np.reshape(avg_datai, [n_class * minn, n_chans, new_n_tpoints])
            y = np.reshape(labelsi, [n_class * minn])

            # Now run SVM using K-fold cross validation for each time point
            for t in range(new_n_tpoints):

                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=state) #Using random_state for consistent results
                xt = x[:, :, t]

                fold_index = 0
                for train_index, test_index in kf.split(xt, y):

                    x_train = xt[train_index]
                    x_test = xt[test_index]

                    if normalization is True:
                        scaler = StandardScaler()
                        x_train = scaler.fit_transform(x_train)
                        x_test = scaler.transform(x_test)

                    svm = SVC(kernel='rbf', tol=1e-4, probability=False)
                    svm.fit(x_train, y[train_index])
                    subacc[i, t, fold_index] = svm.score(x_test, y[test_index])

                    # # Simulate work being done
                    # time.sleep(0.01)
                    # pbar.update(1)    

                    if sub == (n_subs - 1) and i == (n_repeats - 1) and t == (new_n_tpoints - 1) and fold_index == (
                            n_folds - 1):
                        print("\nDecoding finished!\n")

                    fold_index = fold_index + 1

        acc[sub] = np.average(subacc, axis=(0, 2))
        
    if smooth is False:

        return acc

    if smooth is True:

        smooth_acc = smooth_1d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_1d(acc, n=smooth)

        return smooth_acc
    
