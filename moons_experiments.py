#This module runs the experiments as seen in Figures 2 and 3 
# in the paper. We perform a logarithmic sweep of bandwidth 
#(\sigma) values from 10^-3 to 10^1 of a KDE model (Q) trained
#on the 2-dimensional `moons' dataset. We run our own 
#C_T test in addition to each of the baselines at each \sigma 
#10 times (resampling Qm) and plot how the tests' outcomes change 
#on average with \sigma along with 1-sigma error bars 


print('Importing modules...')
#import all relevant tests
import baselines as bln
import data_copying_tests as dct 

#import utilities 
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from tqdm import tqdm 

print('Setting up testbench...')
#experimental parameters
l = 2000 #number of training samples (T)
m = 1000 #number of generated samples (Qm) 
n = 1000 #number of test samples (Pn)
num_trials = 10
num_sigmas = 75
k = 5 #number of clusters in C_T test

#Choose sigmas 
sigmas = np.logspace(start = -3, stop = 1, base = 10, num = num_sigmas)

#Get training set 
T, _ = make_moons(n_samples = l, noise = 0.2) 
#Get test set
Pn, _ = make_moons(n_samples = n, noise = 0.2) 

#Get held out validation set on which to compute log likelihood 
V, _ = make_moons(n_samples = n, noise = 0.2) 

log_lh = np.zeros(len(sigmas))
for i in range(len(sigmas)): 
    Q = KernelDensity(bandwidth = sigmas[i])
    Q.fit(X = T)
    log_lh[i] = Q.score(V)

#get the MLE sigma
opt_sigma = sigmas[np.argmax(log_lh)]

#allocate space for each of the test statistics 

#Two sample NN Test: 
T_LOO_acc = np.zeros((num_sigmas, num_trials))
Qm_LOO_acc = np.zeros((num_sigmas, num_trials))

#Frechet Statistic; 
frech_dist = np.zeros((num_sigmas, num_trials))

#Binning-Based Eval 
NDB_over = np.zeros((num_sigmas, num_trials))
NDB_under = np.zeros((num_sigmas, num_trials))

#Precision & Recall
angular_res = 100
precision = np.zeros((num_sigmas, num_trials, angular_res))
recall = np.zeros((num_sigmas, num_trials, angular_res))

#Generalization Gap 
gg = np.zeros((num_sigmas, num_trials))

#C_T Statistic 
ct = np.zeros((num_sigmas, num_trials))
KM_clf = KMeans(n_clusters = k).fit(T) #instance space partition
T_labels = KM_clf.predict(T) #cell labels for T
Pn_labels = KM_clf.predict(Pn) #cell labels for Pn 

#for each sigma...
print('Iterating through {0:n} sigma values'.format(num_sigmas))
for sig_idx in tqdm(range(num_sigmas)): 
    sigma = sigmas[sig_idx]
    #fit KDE model
    Q = KernelDensity(bandwidth = sigma, kernel = 'gaussian', metric = 'euclidean').fit(T) 
    #for each trial...
    for trial_idx in range(num_trials): 
        Qm = Q.sample(n_samples = m) 
        Qm_labels = KM_clf.predict(Qm) #cell labels for Qm

        #Two Sample NN Test 
        T_tilde = T[np.random.choice(np.arange(l), size = m, replace = False)]
        T_LOO_acc[sig_idx, trial_idx], Qm_LOO_acc[sig_idx, trial_idx] = bln.NN_test(T_tilde,Qm)

        #Frechet Statistic
        frech_dist[sig_idx, trial_idx] = bln.frechet_test(T,Qm)

        #Binning-Based Eval
        NDB_over[sig_idx, trial_idx], NDB_under[sig_idx, trial_idx] = bln.binning_test(Qm, Qm_labels, T, T_labels) 

        #Precision & Recall
        precision[sig_idx, trial_idx, :], recall[sig_idx, trial_idx, :] = bln.precision_recall(T,Qm)

        #Generalization Gap
        gg[sig_idx, trial_idx] = bln.gen_gap(Pn, T, Q)

        #C_T Statistic (note: tau is set to only accept cells with >20 samples)
        ct[sig_idx, trial_idx] = dct.C_T(Pn, Pn_labels, Qm, Qm_labels, T, T_labels, tau = 20 / len(Qm))

print('Completed tests on {0:n} sigmas, {1:n} trials'.format(num_sigmas, num_trials))
