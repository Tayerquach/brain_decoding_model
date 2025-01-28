from decimal import Decimal
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from ptitprince import PtitPrince as pt
import warnings
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from utils.analysis_helpers import clusterbased_permutation_1d_1samp_1sided
warnings.filterwarnings('ignore')

### Define a function to visualize joint ERPs for two conditions together - plot_erp_2cons_results()
def plot_erp_2cons_results(p_vals, avg1, err1, avg2, err2, times, con_labels=['Condition1', 'Condition2'], ylim=[-6, 6], p_threshold=0.05, labelpad=0, cluster_permutation=False):
    """
    Visualize joint ERPs for two conditions together

    Parameters
    ----------
    times: an array with the shape [n_times]
    corresponding to the time-points and the range of x-axis

    con_labels: a list or array with the labels of two conditions,
    default ['Condition 1', 'Condition 2']

    ylim: the lims of y-axis, default [-10, 10]
    p_threshold : a float value, default is 0.05,
    representing the threshold of p-value

    labelpad : a int value, spacing in points from the axes, default 0

    Returns
    -------
    """

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines['bottom'].set_position(('data', 0))

    # highlight the significant time-windows
    tstep = (times[-1]-times[0])/len(times)
    for i, p_val in enumerate(p_vals):
        if p_val < p_threshold:
            plt.fill_between([times[i], times[i]+tstep], [ylim[1]], [ylim[0]],
    facecolor='gray', alpha=0.1)
    # plot the ERPs with statistical results
    plt.fill_between(times, avg1+err1, avg1-err1, alpha=0.2, color='red')
    plt.fill_between(times, avg2+err2, avg2-err2, alpha=0.2, color='green')


    plt.plot(times, avg1, alpha=0.9, color='red')
    plt.plot(times, avg2, alpha=0.9, color='green')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(ylim[0], ylim[1])
    plt.ylabel(r'Amplitude in $\mu$V', fontsize=25, labelpad=labelpad)
    plt.xlabel('Time (ms)', fontsize=25)
    # Adjust the ylabel position
    ax = plt.gca()
    ax.yaxis.set_label_coords(0.1, 0.75)
    # plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.5), fontsize=14)
    # Add legend manually with custom positions
    legend1 = plt.Line2D([0], [0], color='red', label=con_labels[0])
    legend2 = plt.Line2D([0], [0], color='green', label=con_labels[1])
    if cluster_permutation:
        legend3 = mlines.Line2D([0], [0], color='lightgray', linewidth=8, label=f'cluster p-value < {p_threshold}')
    else:
        legend3 = mlines.Line2D([0], [0], color='lightgray', linewidth=8, label=f'p-value < {p_threshold}')

    ax.legend(handles=[legend1, legend2, legend3], loc='upper right', bbox_to_anchor=(1.05, 1.25), fontsize=16)

"""
    Decoding Visualisation
"""
def plot_decoding_acc_tbyt(acc, start_time=0, end_time=1, time_interval=0.01, chance=0.5, p=0.05, cluster_permutation=False,
                           clusterp=0.05, iter=1000, stats_time=[0, 1], color='green', xlim=[0, 1], ylim=[0.4, 0.8],
                           xlabel='Time (s)', ylabel='Decoding Accuracy', figsize=[13, 8], x0=0, labelpad=0,
                           ticksize=16, fontsize=25, title=None, title_fontsize=25, sigshow=True, avgshow=True):
    """
    Plot the time-by-time decoding accuracies

    Parameters
    ----------
    acc : array
        The decoding accuracies.
        The size of acc should be [n_subs, n_tpoints]. n_subs, n_tpoints represent the number of subjects and number of
        time-points, respectively.

    start_time : int or float. Default is 0.
        The start time.
        
    end_time : int or float. Default is 1.
        The end time.
        
    time_interval : float. Default is 0.01.
        The time interval between two time samples.
        
    smooth : bool True or False. Default is True.
        Smooth the results or not.
        
    chance : float. Default is 0.5.
        The chance level.
        
    p : float. Default is 0.05.
        The threshold of p-values.
        
    cluster_permutation : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
        
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
        
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        Time period for statistical analysis.
        
    color : matplotlib color or None. Default is 'green'.
        The color for the curve.
        
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (time) view lims.
        
    ylim : array or list [ymin, ymax]. Default is [0.4, 0.8].
        The y-axis (decoding accuracy) view lims.
        
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
        
    ylabel : string. Default is 'Representational Similarity'.
        The label of y-axis.
        
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
        
    x0 : float. Default is 0.
        The Y-axis is at x=x0.
        
    labelpad : int or float. Default is 0.
        Distance of ylabel from the y-axis.
        
    ticksize : int or float. Default is 12.
        The size of the ticks.
        
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
        
    markersize : int or float. Default is 2.
        The size of significant marker.
        
    title : string-array. Default is None.
        The title of the figure.
        
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    
    sigshow : boolen True or False. Default is False.
        Show the significant windows results or not.

    avgshow : boolen True or False. Default is True.
        Show the averaging decoding accuracies or not.
    """

    # Check accuracy's format
    if len(np.shape(acc)) != 2:
        print("Your accuracy does not have the right format!")
        return "Invalid input!"
    
    # Get number of subjects and time points
    n_subs, n_tpoints = np.shape(acc)

    # Calculate time step
    t_step = float(Decimal((end_time - start_time) / n_tpoints).quantize(Decimal(str(time_interval))))

    if t_step != time_interval:
        return "Invalid input!"
    
    # Calculate the start and end time points
    delta1 = (stats_time[0] - start_time) / t_step - int((stats_time[0] - start_time) / t_step)
    delta2 = (stats_time[1] - start_time) / t_step - int((stats_time[1] - start_time) / t_step)

    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / t_step)
    else:
        stats_time1 = int((stats_time[0] - start_time) / t_step) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / t_step)
    else:
        stats_time2 = int((stats_time[1] - start_time) / t_step) + 1
        
    # Calculate mean and err for accuracy
    avg = np.average(acc, axis=0)
    
    err = np.zeros([n_tpoints])

    for t in range(n_tpoints):
        err[t] = np.std(acc[:, t], ddof=1) / np.sqrt(n_subs)
        
    # Running cluster-based permutation test
    if cluster_permutation == True:
        label = f'cluster p-value < {clusterp}'

        ps_stats = clusterbased_permutation_1d_1samp_1sided(acc[:, stats_time1:stats_time2], level=chance,
                                                            p_threshold=p, clusterp_threshold=clusterp, iter=iter)
        ps = np.zeros([n_tpoints])
        ps[stats_time1:stats_time2] = ps_stats

    else:
        label = f'p-value < {p}'
        ps = np.zeros([n_tpoints])
        for t in range(n_tpoints):
            if t >= stats_time1 and t< stats_time2:
                ps[t] = ttest_1samp(acc[:, t], chance, alternative="greater")[1]
                if ps[t] < p:
                    ps[t] = 1
                else:
                    ps[t] = 0
    
    if sigshow == True:
        print('\nSignificant time-windows:')
        # Get significant time windows
        for t in range(n_tpoints):
            if t == 0 and ps[t] == 1:
                print(str(int(start_time * 1000)) + 'ms to ', end='')
            if t > 0 and ps[t] == 1 and ps[t - 1] == 0:
                print(str(int((start_time + t * t_step) * 1000)) + 'ms to ', end='')
            if t < n_tpoints - 1 and ps[t] == 1 and ps[t + 1] == 0:
                print(str(int((start_time + (t + 1) * t_step) * 1000)) + 'ms')
            if t == n_tpoints - 1 and ps[t] == 1:
                print(str(int(stats_time[1] * 1000)) + 'ms')
    
    yminlim = ylim[0]
    ymaxlim = ylim[1]
    
    for t in range(n_tpoints):
        if ps[t] == 1:
            # plt.plot(t * t_step + start_time + 0.5 * t_step, chance + 0.001, 's', color='r', alpha=1,
            #          markersize=markersize, zorder=2)
            xi = [t * t_step + start_time, t * t_step + t_step + start_time]
            ymin = [chance]
            ymax = [avg[t] - err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor='red', alpha=0.4)
            
    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", chance))
    x = np.arange(start_time + 0.5 * t_step, end_time + 0.5 * t_step, t_step)
    if avgshow is True:
        plt.plot(x, avg, color=color, alpha=0.95, zorder=1)
    plt.fill_between(x, avg+err, avg-err, facecolor=color, alpha=0.35, zorder=1)
    plt.ylim(yminlim, ymaxlim)
    plt.xlim(xlim[0], xlim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=labelpad)
    # Adjust the ylabel position
    ax.yaxis.set_label_coords(0.05, 0.75)
    # legend1 = plt.Line2D([0], [0], color='red', label=label)
    legend2 = mlines.Line2D([0], [0], color=color, linewidth=8, label='Accuracies', alpha=0.35)
    legend3 = mlines.Line2D([0], [0], color='red', linewidth=8, label=label, alpha=0.4)

    ax.legend(handles=[legend2, legend3], loc='upper right', bbox_to_anchor=(1.05, 1.15), fontsize=16)

    plt.title(title, fontsize=title_fontsize, pad=20)
    plt.show()

def plot_raincloud_diff(df, group):
    output_folder = f'photo/raincloud/{group}'
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Rainclouds with FacetGrid according to conditions separately (including three regions) 
    g = sns.FacetGrid(df, col = "Condition", height = 6)
    g = g.map_dataframe(pt.RainCloud, x = "Region", y = "Diff_amp", data = df, bw = 0.2, width_box = .2, point_size = 1,
                    width_viol = 0.5, orient = "v", pointplot = True)

    # Adjust the space between the graphs
    g.fig.subplots_adjust(top=0.85, wspace=0.4) # Increase the horizontal space between plots
    g.set_ylabels('Difference amplitude ($\mu$V)', fontsize=14)

    # Set titles with a larger font size
    g.set_titles(col_template='{col_name}', size=16)  # Adjust size as needed

    # Adjust the font size of the tick labels and axis labels
    for ax in g.axes.flat:
        ax.yaxis.set_tick_params(labelleft=True)
        ax.tick_params(axis='x', labelsize=12)      # Adjust x-axis tick label font size
        ax.tick_params(axis='y', labelsize=12)      # Adjust y-axis tick label font size
    g.fig.suptitle("N400 (300 - 500 ms)",  fontsize=22)
    plt.savefig(f'{output_folder}/conditions_difference_amplitude_distribution.png', bbox_inches='tight')

    # Rainclouds with FacetGrid according to conditions separately (including three regions) TOGETHER
    # Now with the group as hue
    pal = "Set2"
    sigma = .2
    ort = "v"
    dx = "Region"; dy = "Diff_amp"; dhue = "Condition"
    f, ax = plt.subplots(figsize=(12, 6))

    ax=pt.RainCloud(x = dx, y = dy, hue=dhue, data = df, palette = pal, bw = sigma, width_viol = .5, point_size = 1, width_box = .2,
                    ax = ax, orient = ort , alpha = .65, dodge = True, pointplot = True, move = 0)
    ax.set_ylabel('Absolute Difference amplitude ($\mu$V)', fontsize=20)
    # Hide x-axis labels
    ax.set_xlabel("", fontsize=14)  # Removes the x-axis label
    ax.tick_params(axis='x', labelsize=20)      # Adjust x-axis tick label font size
    ax.tick_params(axis='y', labelsize=14)      # Adjust y-axis tick label font size
    # Adjust the legend size and position inside the plot
    # Get the handles and labels from the current legend
    handles, labels = ax.get_legend_handles_labels()

    # Filter out duplicates: Keep only the first occurrence of each label
    filtered_handles = []
    filtered_labels = []
    seen_labels = set()

    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            filtered_handles.append(handle)
            filtered_labels.append(label)
            seen_labels.add(label)
        
        # Stop after adding two unique conditions
        if len(filtered_labels) == 2:
            break

    # Create the custom legend with only two unique conditions
    ax.legend(filtered_handles, filtered_labels, fontsize=20, loc='upper right', bbox_to_anchor=(1.01, 1.35), frameon=True)
    plt.title("N400 (300 - 500 ms)",  fontsize=22)
    plt.savefig(f'{output_folder}/together_regions_difference_amplitude_distribution.png', dpi=500, bbox_inches='tight')

    # Rainclouds with FacetGrid according to regions separately (including different conditions) SEPARATELY
    g = sns.FacetGrid(df, col = "Region", height = 6)
    g = g.map_dataframe(pt.RainCloud, x = "Condition", y = "Diff_amp", data = df, bw = 0.2, width_box = .2, point_size = 1,
                    width_viol = 0.5, orient = "v", pointplot = True)

    # Adjust the space between the graphs
    g.fig.subplots_adjust(top=0.85, wspace=0.2) # Increase the horizontal space between plots
    g.set_ylabels('Absolute Difference amplitude ($\mu$V)', fontsize=20)

    # Set titles with a larger font size
    g.set_titles(col_template='{col_name}', size=20)  # Adjust size as needed

    # Adjust the font size of the tick labels and axis labels
    for ax in g.axes.flat:
        ax.yaxis.set_tick_params(labelleft=True)
        ax.tick_params(axis='x', labelsize=20)      # Adjust x-axis tick label font size
        ax.tick_params(axis='y', labelsize=14)      # Adjust y-axis tick label font size
    g.fig.suptitle("N400 (300 - 500 ms)",  fontsize=22)
    plt.savefig(f'{output_folder}/separate_regions_difference_amplitude_distribution.png', dpi=500, bbox_inches='tight')

    

def plot_raincloud_decoding(df, group):
    
    output_folder = f'photo/raincloud/{group}'
    # Check if the folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Rainclouds with FacetGrid according to conditions
    g = sns.FacetGrid(df, col = "Condition", height = 6)
    g = g.map_dataframe(pt.RainCloud, x = "Region", y = "Accuracy", data = df, bw = 0.2, width_box = .2, point_size = 1,
                    width_viol = 0.5, orient = "v", pointplot = True)

    # Adjust the space between the graphs
    g.fig.subplots_adjust(top=0.85, wspace=0.2) # Increase the horizontal space between plots
    g.set_ylabels('Decoding Accuracy', fontsize=20)

    # Set titles with a larger font size
    g.set_titles(col_template='{col_name}', size=20)  # Adjust size as needed

    # Adjust the font size of the tick labels and axis labels
    for ax in g.axes.flat:
        ax.yaxis.set_tick_params(labelleft=True)
        ax.tick_params(axis='x', labelsize=17)      # Adjust x-axis tick label font size
        ax.tick_params(axis='y', labelsize=14)      # Adjust y-axis tick label font size
    g.fig.suptitle("N400 (300 - 500 ms)",  fontsize=22)
    plt.savefig(f'{output_folder}/separate_conditions_decoding_accuracies_distribution.png', dpi=400, bbox_inches='tight')

    # Together
    # Rainclouds with FacetGrid according to conditions separately (including three regions) TOGETHER
    # Now with the group as hue
    pal = "Set2"
    sigma = .2
    ort = "v"
    dx = "Region"; dy = "Accuracy"; dhue = "Condition"
    f, ax = plt.subplots(figsize=(12, 6))

    ax=pt.RainCloud(x = dx, y = dy, hue=dhue, data = df, palette = pal, bw = sigma, width_viol = .5, point_size = 1, width_box = .2,
                    ax = ax, orient = ort , alpha = .65, dodge = True, pointplot = True, move = 0)
    ax.set_ylabel('Decoding Accuracy', fontsize=20)
    # Hide x-axis labels
    ax.set_xlabel("", fontsize=14)  # Removes the x-axis label
    ax.tick_params(axis='x', labelsize=20)      # Adjust x-axis tick label font size
    ax.tick_params(axis='y', labelsize=14)      # Adjust y-axis tick label font size
    # Adjust the legend size and position inside the plot
    # Get the handles and labels from the current legend
    handles, labels = ax.get_legend_handles_labels()

    # Filter out duplicates: Keep only the first occurrence of each label
    filtered_handles = []
    filtered_labels = []
    seen_labels = set()

    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            filtered_handles.append(handle)
            filtered_labels.append(label)
            seen_labels.add(label)
        
        # Stop after adding two unique conditions
        if len(filtered_labels) == 2:
            break

    # Create the custom legend with only two unique conditions
    ax.legend(filtered_handles, filtered_labels, fontsize=20, loc='upper right', bbox_to_anchor=(1.01, 1.35), frameon=True)
    plt.title("N400 (300 - 500 ms)",  fontsize=22)
    plt.savefig(f'{output_folder}/together_regions_decoding_accuracy_distribution.png', dpi=500, bbox_inches='tight')


    # Rainclouds with FacetGrid according to regions
    g = sns.FacetGrid(df, col = "Region", height = 6)
    g = g.map_dataframe(pt.RainCloud, x = "Condition", y = "Accuracy", data = df, bw = 0.2, width_box = .2, point_size = 1,
                    width_viol = 0.5, orient = "v", pointplot = True)

    # Adjust the space between the graphs
    g.fig.subplots_adjust(top=0.85, wspace=0.2) # Increase the horizontal space between plots
    g.set_ylabels('Decoding Accuracy', fontsize=20)

    # Set titles with a larger font size
    g.set_titles(col_template='{col_name}', size=20)  # Adjust size as needed

    # Adjust the font size of the tick labels and axis labels
    for ax in g.axes.flat:
        ax.yaxis.set_tick_params(labelleft=True)
        ax.tick_params(axis='x', labelsize=20)      # Adjust x-axis tick label font size
        ax.tick_params(axis='y', labelsize=14)      # Adjust y-axis tick label font size
    g.fig.suptitle("N400 (300 - 500 ms)",  fontsize=22)
    plt.savefig(f'{output_folder}/regions_decoding_accuracies_distribution.png', dpi=500, bbox_inches='tight')