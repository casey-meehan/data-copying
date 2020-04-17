#This module provides plotting functions useful for visualizing the 
#experimental results from the paper. 

#import modules 
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

def set_plot_defaults(): 
    """Sets up default plotting values for our figures
    """
    mpl.rcdefaults()
    plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
    plt.rc('legend', fontsize=30)    # legend fontsize
    plt.rc('figure', titlesize=30)  # fontsize of the figure title
    plt.rc('legend', frameon = False) #don't add a box around legends
    plt.rc('lines', linewidth = 4) #make lines thick enough to see 
    plt.rc('axes', titlesize = 40) #make titles big enough to read


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

    set_plot_defaults()
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
        ax2.set_yscale('log') #this is needed to make the maximum pronounced 
        ax2.plot(x_values, log_lh, color='gray', label = '$\propto \log$Lik.')
        ax2.legend(loc = 'lower right') 

    #plot the black dotted reference line 
    if ref_value is not None: 
        ax1.plot(x_values, np.ones(len(x_values))*ref_value, '--', color = 'black') 

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
#            ax1.plot([], [], color = 'gray') 
            ax1.plot(x_values[opt_sig_idx], trace_mean[opt_sig_idx], 'ro')



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

def plot_PR_curve(sigmas, precisions, recalls, fname, opt_sigma, zoom = True):
    """plots precision / recall curves for KDE test. Assumes the 
    last sigma value is the MLE sigma 

    Inputs: 
        sigmas: [1 X num_sigmas] np array of KDE sigma values

        precisions: [num_sigmas X num_trials X angular_res] np array of 
            precision values

        recalls: [num_sigmas X num_trials X angular_res] np array of 
            recall values

        fname: (string) file name for png stored in './images' directory 
            (add .png to end) 

        opt_sigma: (scalar) the MLE sigma value

        zoom: (boolean) whether to zoom on the 'corner' of the PR curve 
            (near 1,1) 
    """
    set_plot_defaults()

    prec_curves_ave = precisions.mean(axis = 1)
    prec_curves_std = precisions.std(axis = 1)
    reca_curves_ave = recalls.mean(axis = 1)
    reca_curves_std = recalls.std(axis = 1)

    #plot each of the PR curves 
    for i in range(len(sigmas)): 
        pca = prec_curves_ave[i]
        pcs = prec_curves_std[i]
        rca = reca_curves_ave[i]
        rcs = reca_curves_std[i]

        #plot the MLE sigma with a dotted red line 
        if(sigmas[i] == opt_sigma):
            plt.plot(pca, rca, 'r--',label = 'MLE $\sigma$' )
        else: 
            plt.plot(pca, rca, label = '{0:0.2e}'.format(sigmas[i]))

        plt.fill_between(pca, rca - rcs, rca + rcs, alpha = 0.05)
        plt.fill_betweenx(rca, pca - pcs, pca + pcs, alpha = 0.05)

    plt.gcf().set_size_inches((12,6))

    #set axis ticks: 
    tick_mod = 2
    for i, tick in enumerate(plt.gca().xaxis.get_ticklabels()): 
        if i % tick_mod != 0: 
            tick.set_visible(False)

    if zoom: 
        plt.xlim(0.90, 0.96)
        plt.ylim(0.90, 0.935)

    plt.legend(frameon = False, loc = 'lower left', ncol = 2)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision & Recall')
    plt.savefig('./images/' + fname, bbox_inches = 'tight', pad_inches = 0)

def plot_VAE(d_vals, traces, trace_names, xlabel, ylabel, title,
        fname, ref_value = None, ELBO = None):
    """Produces all plots associated with VAE tests

    Inputs 
       d_vals: (1 X num VAEs) np array of the x-axis values: the VAE
            latent dimension size from small to large. 

       traces: list of (num VAEs X num_trials) np arrays, each 
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

       ELBO: (1 X num VAEs) trace indicating the VAE ELBO measured
            on held-out validation set for each of the VAEs. 
            If None, is ignored. 

       fname: (string) name of .png to be saved in the ./images/ 
            directory 
    """
#    #check for number of elements: 
#    if len(x_values) < 10: 
#        raise ValueError('Needs >=10 values to plot. x_values has only \
#        {0:n} values'.format(len(x_values)))

    set_plot_defaults()
    f, ax1 = plt.subplots(1,1)
    f.set_size_inches((12,6))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    #flip x-axis to go from complex (ovefit) to small (underfit) 
    ax1.set_xlim(1.05 * d_vals.max(), 0.95 * d_vals.min())

    #plot log likelihood 
    if ELBO is not None: 
        #get max ELBO d_val idx
        opt_dval_idx = np.argmax(ELBO)
        #shift up to make all values > 0
        ELBO = ELBO - ELBO.min() + 1
        ax2 = ax1.twinx()
        ax2.get_yaxis().set_visible(False) #only show proportionality
        ax2.set_yscale('log') #this is needed to make the maximum pronounced 
        ax2.plot(d_vals, ELBO, color='gray', label = '$\propto$ ELBO')
        ax2.legend(loc = 'lower right') 

    #plot the black dotted reference line 
    if ref_value is not None: 
        ax1.plot(d_vals, np.ones(len(d_vals))*ref_value, '--', color = 'black') 

    #plot each of the traces 
    for trace, name in zip(traces, trace_names): 
        trace_mean = trace.mean(axis = 1)
        trace_std = trace.std(axis = 1)
        #plot the mean of all trials
        ax1.plot(d_vals, trace_mean, label = name) 
        #plot 1-std error buffer
        ax1.fill_between(d_vals, trace_mean - trace_std, trace_mean 
                         + trace_std, alpha = 0.2) 
        #plot red dot at optimal sigma value 
        if ELBO is not None: 
#            ax1.plot([], [], color = 'gray') 
            ax1.plot(d_vals[opt_dval_idx], trace_mean[opt_dval_idx], 'ro')



    #set legend
    ax1.legend()

    #get rid of every other tick 
#    tick_mod = 2
#    for i, tick in enumerate(ax1.xaxis.get_ticklabels()): 
#        if i % tick_mod != 0: 
#            tick.set_visible(False)

    #save & show figure 
    plt.savefig('./images/' + fname, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

