import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp, ttest_rel


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

    print(f"Shape of EEG original data: {EEG_data.shape} \n")

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

def get_cluster_index_1d_1sided(m):

    """
    Get 1-D & 1-sided cluster-index information from a vector

    Parameters
    ----------
    m : array
        A significant vector.
        The values in m should be 0 or 1, which represent not significant point or significant point, respectively.

    Returns
    -------
    index_v : array
        The cluster-index vector.
    index_n : int
        The number of clusters.
    """

    x = np.shape(m)[0]
    b = np.zeros([x+2])
    b[1:x+1] = m

    index_v = np.zeros([x])

    index_n = 0
    for i in range(x):
        if b[i+1] == 1 and b[i] == 0 and b[i+2] == 1:
            index_n = index_n + 1
        if b[i+1] == 1:
            if b[i] != 0 or b[i+2] != 0:
                index_v[i] = index_n

    return index_v, index_n

def clusterbased_permutation_1d_1samp_1sided(results, level=0, p_threshold=0.05, clusterp_threshold=0.05, n_threshold=2,
                                             iter=1000, state=42):

    """
    1-sample & 1-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x]. n_subs represents the number of subjects.
    level : float. Default is 0.
        An expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    clusterp_threshold : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    n_threshold : int. Default is 2.
        The threshold of number of values in one cluster (number of values per cluster > n_threshold).
    iter : int. Default is 1000.
        The times for iteration.

    Returns
    -------
    ps : float
        The permutation test resultz, p-values.
        The shape of ps is [x]. The values in ps should be 0 or 1, which represent not significant point or significant
        point after cluster-based permutation test, respectively.
    """
    # Set the random seed
    np.random.seed(state)

    nsubs, x = np.shape(results)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        ts[t], p = ttest_1samp(results[:, t], level, alternative='greater')
        if p < p_threshold and ts[t] > 0:
            ps[t] = 1
        else:
            ps[t] = 0

    cluster_index, cluster_n = get_cluster_index_1d_1sided(ps)

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t in range(x):
                if cluster_index[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in tqdm(range(iter)):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t in range(x):
                    if cluster_index[t] == j + 1:
                        v = np.hstack((results[:, t], chance))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)

            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index[t] == i + 1:
                        ps[t] = 0

    newps = np.zeros([x + 2])
    newps[1:x + 1] = ps

    for i in range(x):
        if newps[i + 1] == 1 and newps[i] != 1:
            index = 0
            while newps[i + 1 + index] == 1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

    ps = newps[1:x + 1]

    return ps