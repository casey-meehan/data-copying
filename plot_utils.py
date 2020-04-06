#This module provides plotting functions useful for visualizing the 
#experimental results from the paper. 

#import modules 
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

#set plotting defaults 
def plot_model_sweep(x_values, traces, trace_names, xlabel, ylabel, title,
        fname, ref_value = None, log_lh = None):
    """Produces most of plots in the paper excluding the precision recall 
    plot. 

    Inputs 
       x_values: (1 X num_sigmas) np array of the x-axis values. In the
            case of the KDE tests, this will be an array of sigma values. 
            NOTE: >=10 sigmas expected.  

       traces: list of (num_sigmas X num_trials) np arrays, each 
            representing one trace to be plotted. Each trace will be 
            averaged across trials, and plotted with a 1-std dev error
            buffer. 

       trace_names: list of strings, each naming its corresponding 
            trace in 'traces'. Appears next to each trace in the 
            legend. 

       xlabel: (string) label of x-axis values 

       ylabel: (string) label of y-axis values 

       title: (string) chart title

       ref_value: scalar value indicating where on the y-axis to add
            a horizontal black dotted line (as a reference value). If
            =None, is ignored. 

       log_lh: (1 X num_sigmas) trace indicating the log likelihood 
            (or log ELBO) corresponding to each sigma. If None, is 
            ignored. 

       fname: (string) name of .png to be saved in the ./images/ 
            directory 
    """
    #check for number of elements: 
    if len(x_values) < 10: 
        raise ValueError('Needs >=10 values to plot. x_values has only \
        {0:n} values'.format(len(x_values)))
    #set plot format 
    mpl.rcdefaults()
    plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
    plt.rc('legend', fontsize=30)    # legend fontsize
    plt.rc('figure', titlesize=30)  # fontsize of the figure title
    plt.rc('legend', frameon = False) #don't add a box around legends
    plt.rc('lines', linewidth = 4) #make lines thick enough to see 
    plt.rc('axes', titlesize = 40) #make titles big enough to read
    
    f, ax1 = plt.subplots(1,1)
    f.set_size_inches((12,6))
    ax1.set_xscale('log')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)

    #plot log likelihood 
    if log_lh is not None: 
        #get MLE sigma idx
        opt_sig_idx = np.argmax(log_lh)
        #shift up such that only 10 values are >0
        log_lh = log_lh - np.sort(log_lh)[::-1][9]
        ax2 = ax1.twinx()
        ax2.get_yaxis().set_visible(False) #only show proportionality
        ax2.plot(x_values, log_lh, color='gray', label = '$\propto \log$Lik.')
        ax2.legend(loc = 'lower right') 
        ax2.set_yscale('log') #this is needed to make the maximum pronounced 

    #plot each of the traces 
    for trace, name in zip(traces, trace_names): 
        trace_mean = trace.mean(axis = 1)
        trace_std = trace.std(axis = 1)
        #plot the mean of all trials
        ax1.plot(x_values, trace_mean, label = name) 
        #plot 1-std error buffer
        ax1.fill_between(x_values, trace_mean - trace_std, trace_mean 
                         + trace_std, alpha = 0.2) 
        #plot red dot at optimal sigma value 
        if log_lh is not None: 
            ax1.plot(x_values[opt_sig_idx], trace_mean[opt_sig_idx], 'ro')

    #plot the black dotted reference line 
    ax1.plot(x_values, np.ones(len(x_values))*ref_value, '--', color = 'black') 


    #set legend
    ax1.legend()

    #get rid of every other tick 
    tick_mod = 2
    for i, tick in enumerate(ax1.xaxis.get_ticklabels()): 
        if i % tick_mod != 0: 
            tick.set_visible(False)

    #save & show figure 
    plt.savefig('./images/' + fname, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

