#This module includes code for the baselines tested in the paper: 
#frechet statistic, binning based eval, precision/recall, and most notably 
#the two-sample nearest neighbor statistic (see Figure 2 of the paper) 

import numpy as np
from sklearn.neighbors import NearestNeighbors as NN 
from sklearn.neighbors import KernelDensity
from scipy.linalg import sqrtm
from scipy.stats import norm
from sklearn.cluster import KMeans

##########################
#Two-Sample NN Statistics#
##########################

def NN_test(T, Qm): 
    """run NN two-sample test on training and generated samples. As specified in the 
    related work section of the paper, we use the method of Xu et al. (2018) wherein 
    both the leave-one-out (LOO) accuracy of the training (T) and generated samples (Qm) 
    are reported, instead of averaged into a single statistic. 

    As noted in the paper, the T and Qm must be of the same size in this test. 

    Inputs: 
        T: (m x d) np array of m training samples, each of dimension d

        Qm: (m x d) np array of m training samples, each of dimension d

    Outputs: 
        T_LOO_acc: the LOO accuracy score for the training samples 
            (how often is a training sample nearest another training sample) 

        Qm_LOO_acc: the LOO accuracy score for the generated samples 
            (how often is a generated sample nearest another generated sample) 
    """

    #Assert that T and Qm are the same size 
    l = T.shape[0]
    m = Qm.shape[0]

    if l != m: 
        raise ValueError("T ({0:n} samples) and Qm ({1:n} samples) must be of the same size for this test".format(l,m))

    #label training samples 1 and generated samples 0
    T_labels = np.ones(m) 
    Qm_labels = np.zeros(m)

    #make a concatenated dataset of both training and generated samples (to find nearest neighbors) 
    T_Qm = np.concatenate((T, Qm), axis = 0)
    T_Qm_labels = np.concatenate((T_labels, Qm_labels), axis = 0)

    #get nearest neighbors  
    NN_clf = NN(n_neighbors = 2).fit(T_Qm)
    preds_T = NN_clf.kneighbors(T, n_neighbors = 2, return_distance = False)[:,1] #index of T nearest neighbs. 
    preds_T = T_Qm_labels[preds_T] #label of T nearest neighbs
    preds_Qm = NN_clf.kneighbors(Qm, n_neighbors = 2, return_distance = False)[:,1] #index of Qm nearest neighbs
    preds_Qm = T_Qm_labels[preds_Qm] #label of Qm nearest neighbs

    #get accuracy NN accuracy for T and Qm. Ideal is ~0.5 for each 
    T_LOO_acc = np.mean(preds_T == 1)
    Qm_LOO_acc = np.mean(preds_Qm == 0)
    return T_LOO_acc, Qm_LOO_acc



###################
#Frechet Statistic#
###################

def frechet_test(T, Qm):
    """Runs the frechet distance test (Heusel et al.) 
    between training and generated samples. 
    In effect, this test fits an MLE gaussian on both T and Qm, 
    and checks the Frechet distance between these two distributions.
    See Appendix Section 7.4 for Frechet distance formula. 

    Inputs: 
        T: (l x d) np array of l training samples, each of dimension d

        Qm: (m x d) np array of m training samples, each of dimension d

    Outputs:
        frechet_dist: scalar measure of frechet distance between samples
    """

    T_cov = np.cov(T, rowvar=False)
    T_mean = T.mean(axis = 0)
    Qm_cov = np.cov(Qm, rowvar=False)
    Qm_mean = Qm.mean(axis = 0)

    covmean = sqrtm(T_cov.dot(Qm_cov))
    frechet_dist = np.linalg.norm(T_mean - Qm_mean) + np.trace(T_cov) + np.trace(Qm_cov) - 2 * np.trace(covmean)
    return frechet_dist 


####################
#Binning-Based Eval#
####################
def binning_test(Qm, Qm_cells, T, T_cells): 
    """Runs the binning test (Richardson & Weiss, 2018). 
    This checks the null hypothesis that the probability 
    mass allocated to each cell (bin) is equal for the 
    generative distribution (Q) and true distribution (P). 
    See Appendix 7.4 in our paper for more details.

    Note: there should be >0 training (T) samples in each of the cells. 

    Inputs: 
        Qm: (m X d) np array representing generated sample of 
            length n (with dimension d) 

        Qm_cells: (1 X m) np array of integers indicating which of the 
            k cells each sample belongs to

        T: (l X d) np array representing training sample of 
            length l (with dimension d)

        T_cells: (1 X l) np array of integers indicating which of the 
            k cells each sample belongs to

    Outputs: 
        NDB_over: Fraction of cells with greater than 0.05 significance 
            level (overrepresented by Q). 

        NDB_under: Fraction of cells with less than the 0.05 significance 
            level (underrepresented by Q). 
    """
    l = T.shape[0]
    m = Qm.shape[0]

    #check that each cell has >0 training samples 
    labels, cts = np.unique(T_cells, return_counts = True) 
    k = len(labels) #number of unique cells
    if np.product(cts) == 0: 
        raise ValueError("One of the cells in this partition has 0 training samples. Consider reducing the number of cells in the partition.")

    #get probability mass in each cell for T
    cts = cts[labels.astype(int)] #put in order of cell label 
    T_of_pi = cts / l

    #get probability mass in each cell for Q 
    labels, cts = np.unique(Qm_cells, return_counts = True) 
    cts = cts[labels.astype(int)] #put in order of cell label 
    Qm_of_pi = cts / m 

    #compute number NDB over and NDB under 
    p_hat = (l * T_of_pi + m * Qm_of_pi) / (l + m) #normalizing constant 
    Zpi = (Qm_of_pi - T_of_pi) / ( np.sqrt( p_hat*(1 - p_hat) * ((1/l) + (1/m)) ) )
    sig_vals_over = norm.sf(Zpi[Zpi > 0])
    NDB_over = np.sum(sig_vals_over < 0.05) / k
    sig_vals_under = norm.sf(np.abs(Zpi[Zpi < 0])) 
    NDB_under = np.sum(sig_vals_under < 0.05) / k
    return NDB_over, NDB_under




#######################
#Precision/Recall Test#
#######################

def precision_recall(T, Qm, angular_res = 100, n_clusters = 5, n_clusterings = 10): 
    """Performs Precision & Recall test (Sajadi et. al) on T/Qm. 
    This test estimates the precision and recall of the generated distribution (Q)
    on the true distribution (P) from the generated samples (Qm) and training 
    samples (T). See Appendix 7.4 for more details. 
    Inputs: 
        - T: (l X d) numpy array containing the training sample of dim d 

        - Qm: (m X d) numpy array containing the generated sample of dim d 

        - angular_res: number of points on PR curve

        - n_clusters: number of cells to discretize the instance space into.
            higher values are more computationally intensitve but also 
            provide more accurate estimates. 

        - n_clusterings: number of kmeans clusterings to average precision
            and recall statistics over 

    Outputs: 
        -precision: angular_res-length np array of precision values 

        -recall: angular_res-length np array of recall values 
    """
    #get size of samples 
    l = T.shape[0]
    m = Qm.shape[0]

    #get lambda values: used in computing each point of PR curve 
    lams = np.tan((np.pi / 2) * (np.arange(angular_res) + 1)/(angular_res + 1))

    #precision and recall values 
    alphas = np.zeros((n_clusterings, angular_res))
    betas = np.zeros((n_clusterings, angular_res))

    for n in range(n_clusterings): 
        #Fit a kmeans classifier on the joint set of training and generated pts
        km_clf = KMeans(n_clusters = n_clusters, n_init = 1).fit(np.concatenate((T, Qm), axis = 0))

        #Get fraction of points in each cell 
        T_cells  = km_clf.predict(T)
        labels, cts = np.unique(T_cells, return_counts = True) 
        cts = cts[labels] #put in order of cell label 
        T_of_pi = cts / l

        Qm_cells  = km_clf.predict(Qm)
        labels, cts = np.unique(Qm_cells, return_counts = True) 
        cts = cts[labels] #put in order of cell label 
        Qm_of_pi = cts / m

        #compute PR curve 
        lam_T_of_pi = lams[:,None] * T_of_pi #angular_res X k matrix of \lamda * T_of_pi for all \lambda, \pi
        Qms = np.ones((angular_res, 1)) * Qm_of_pi #angular_res X k matrix repeating Qm_of_pi in each row 
        alpha = np.concatenate((lam_T_of_pi[:,:,None], Qms[:,:,None]), axis = 2).min(axis = 2).sum(axis = 1)

        Ts = np.ones((angular_res, 1)) * T_of_pi #angular_res X k matrix repeating T_of_pi in each row 
        Qm_of_pi_over_lam = Qm_of_pi / lams[:,None] #angular_res X k matrix of Qm_of_pi / \lambda for all \lambda, \pi
        beta = np.concatenate((Qm_of_pi_over_lam[:,:,None], Ts[:,:,None]), axis = 2).min(axis = 2).sum(axis = 1)

        #store PR curve of each Kmeans trial 
        alphas[n, :] = alpha
        betas[n,:] = beta

    #average all kmeans trials 
    precision = alphas.mean(axis = 0)
    recall = betas.mean(axis = 0)
    return precision, recall

#########################
#Generalization Gap Test#
#########################

def gen_gap(Pn, T, Q): 
    """Runs the generalization gap test. This test 
    simply checks the difference between the likelihood 
    assigned to the training set versus that assigned to 
    a held out test set. 

    Inputs: 
        Pn: (n X d) np array containing the held out test sample 
            of dimension d

        T: (l X d) np array containing the training sample of 
            dimension d

        Q: trained model of type scipy.neighbors.KernelDensity 

    Outputs: 
        log_lik_gap: scalar representing the difference of the log
            likelihoods of Pn and T
    """
    return Q.score(T) - Q.score(Pn) 
