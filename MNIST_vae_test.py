#This module produces Figure 4 (a)(b)(c) as seen in the paper. Specifically, 
#reproduces the VAE tests on the MNIST dataset. This involves compressing 
#MNIST images to a lower-dimensional latent space in which the distance 
#between images is symantically meaningful. 

import numpy as np
import torch
from torch.nn import functional as F
from pathlib import Path
import sys
from tqdm import tqdm 
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity as KDE

import MNIST_utils
import data_copying_tests as dct 
import baselines as bln 
import plot_utils as plu
sys.path.append('./MNIST_VAE_models')
from VAE_model import VAE

#ELBO helper functions

def ELBO(recon_x, x, mu, logvar):
    """Compute loss for a batch of images

    Inputs: 
        - recon_x: batch of reconstructed images 

        - x: batch of original images 

        - mu: batch of latent mean values 

        - logvar: batch of latent logvariances 
    Outputs: 
        -BCE + KLD: ELBO loss = likelihood term of deconstructed image + KL prior loss 
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def get_ELBO(samples, vae_model): 
    """returns average ELBO over samples for vae_model

    Inputs: 
        -samples: (n_samps X 28 X 28) np array of samples to get average 
            ELBO of

        -vae_model: VAE with which to compute ELBO
    """
    imgs = torch.Tensor(samples).unsqueeze(1).cuda()

    n_samples = len(samples)

    elbos = np.array([])

    #fetch samples in batches of 128, and add average ELBO
    for i in range(int(n_samples / 128)): 
        data = imgs[i*128 : (i+1)*128].view(128, 1, 28, 28)
        recon_batch, mu, logvar = vae_model(data)
        elbos = np.append(elbos, ELBO(recon_batch, data, mu, logvar).item() / 128)

    #finish the rest of the samples
    if n_samples < 128: 
        i = 0
    else: 
        i+=1

    rem = n_samples - i*128
    data = imgs[i*128: ].view(rem, 1, 28, 28)
    recon_batch, mu, logvar = vae_model(data)

    elbos = np.append(elbos, ELBO(recon_batch, data, mu, logvar).item() / rem)

    return elbos.mean()

#####################
#Get and embed MNIST#
#####################

print('Loading MNIST dataset...') 
T, Val, Pn = MNIST_utils.get_mnist_data()
l = len(T)
n = len(Pn)

print('Embedding train/test/val images...') 
train_z_64 = MNIST_utils.pae_codes(T)
val_z_64 = MNIST_utils.pae_codes(Val)
test_z_64 = MNIST_utils.pae_codes(Pn)

#get instance space partition
print('Getting instance space partition $\Pi$...')
n_clusters = 50
KM_clf = KMeans(n_clusters).fit(train_z_64)


#########################################
#Get ELBO of validation set for each VAE#
#########################################

#set up GPU
device = torch.device("cuda")

#model sizes (latent dimensions) to test: 
d_vals = np.linspace(5, 100, 20, dtype = int)
n_vaes = len(d_vals)

#make saved_data directory if needed
Path('./saved_data').mkdir(parents = True, exist_ok = True)

#collect ELBOs if not already there 
ELBO_data = Path('./saved_data/VAE_MNIST_ELBOs.npy') 
if ELBO_data.is_file(): 
    print('Using saved ELBO data in \'./saved_data/\' dir')
    VAE_dvals_ELBO = np.load('./saved_data/VAE_MNIST_ELBOs.npy') 
else: 
    print('ELBO data not found. Measuring VAE ELBOs of {0:n} \
latent dim. values and saving it in \'./saved_data\' dir.'.format(n_vaes)) 
    VAE_dvals_ELBO = np.array([])
    for d in tqdm(d_vals): 
        #load model
        vae = VAE(d, l=3).eval().to(device)
        vae.load_state_dict(torch.load('./MNIST_VAE_models/trained_weights/VAE_d' 
            + str(d) + '.pkl', map_location = device))
        #get validation set elbo
        VAE_dvals_ELBO = np.append(VAE_dvals_ELBO, get_ELBO(Val, vae))

    np.save('./saved_data/VAE_MNIST_ELBOs.npy', VAE_dvals_ELBO)

#####################################################
#Run tests at each sigma value for several trials# 
#####################################################
n_trials = 10

#get train and test cell labels 
Pn_labels = KM_clf.predict(test_z_64)
T_labels = KM_clf.predict(train_z_64)

#allocate space for statistics 
#C_T test
ct_data = Path('./saved_data/VAE_MNIST_C_Ts.npy')
if ct_data.is_file(): 
    print('Using saved C_T test data in \'./saved_data/\' dir')
    do_ct_test = False
    C_Ts = np.load('./saved_data/VAE_MNIST_C_Ts.npy')
else: 
    print('No C_T test data found, will run C_T tests') 
    do_ct_test = True
    C_Ts = np.zeros((n_vaes, n_trials))

#Two sample NN test
T_LOO_acc_data = Path('./saved_data/VAE_MNIST_T_LOO_acc.npy')
Qm_LOO_acc_data = Path('./saved_data/VAE_MNIST_Qm_LOO_acc.npy')
if T_LOO_acc_data.is_file() and Qm_LOO_acc_data.is_file(): 
    print('Using saved NN test data in \'./saved_data/\' dir')
    do_NN_test = False
    T_LOO_acc = np.load('./saved_data/VAE_MNIST_T_LOO_acc.npy')
    Qm_LOO_acc = np.load('./saved_data/VAE_MNIST_Qm_LOO_acc.npy')
else: 
    print('No NN test data found, will run NN tests') 
    do_NN_test = True
    T_LOO_acc = np.zeros((n_vaes, n_trials))
    Qm_LOO_acc = np.zeros((n_vaes, n_trials))

#Generalization gap test
gg_data = Path('./saved_data/VAE_MNIST_gg.npy')
if gg_data.is_file(): 
    print('Using saved gen gap test data in \'./saved_data/\' dir')
    do_gg_test = False
    gg = np.load('./saved_data/VAE_MNIST_gg.npy')
else: 
    print('No gen gap test data found, will run gen gap tests') 
    do_gg_test = True
    gg = np.zeros((n_vaes, 1))

#for each sigma value...
if do_ct_test or do_NN_test or do_gg_test: 
    print('Gathering C_T, 2-sample NN, and/or gen. gap for {0:n} vae models , {1:n} trials \
    each'.format(n_vaes, n_trials))

    for vae_idx in tqdm(range(n_vaes)): 
        #load model
        d = d_vals[vae_idx]
        vae = VAE(d, l=3).eval().to(device)
        vae.load_state_dict(torch.load('./MNIST_VAE_models/trained_weights/VAE_d' 
            + str(d) + '.pkl', map_location = device))

        if do_gg_test: 
            #gather gen. gap test statistic 
            gg[vae_idx] = get_ELBO(T, vae) - get_ELBO(Pn, vae) 

        #for each trial...
        for trial_idx in range(n_trials): 
            #generate the same number of samples as the test data (10k), m = n
            m = len(test_z_64)
            zs = torch.randn(m, d).to(device) #std normal input to decoder
            Qm = (vae.decode(zs).view(m, 28, 28).to('cpu').detach().numpy() - 0.5) * 2

            #embed generated images with separate MNIST encoder with perceptual loss:
            Qm_z_64 = MNIST_utils.pae_codes(Qm)
            Qm_labels = KM_clf.predict(Qm_z_64)

            #gather C_T statistic 
            if do_ct_test: 
                C_Ts[vae_idx, trial_idx] = dct.C_T(test_z_64, Pn_labels, Qm_z_64, Qm_labels,
                    train_z_64, T_labels, tau = 20 / len(Qm)) 

            #gather NN test statistics 
            if do_NN_test:
                #first subsample T to be the same size as m
                T_tilde = train_z_64[np.random.choice(np.arange(l), size = n, replace = False)]
                #run test setting n_LOO = 1000 to speed things up (minimal effect on outcome)
                T_LOO_acc[vae_idx, trial_idx], Qm_LOO_acc[vae_idx, trial_idx] \
                    = bln.NN_test(T_tilde,Qm_z_64, n_LOO = 1000)

##############
#Save Results#
##############
if do_ct_test: 
    print('saving C_T test data in \'./saved_data\' dir...')
    np.save('./saved_data/VAE_MNIST_C_Ts.npy', C_Ts)
if do_NN_test: 
    print('saving two sample NN test data in \'./saved_data\' dir...')
    np.save('./saved_data/VAE_MNIST_T_LOO_acc.npy', T_LOO_acc)
    np.save('./saved_data/VAE_MNIST_Qm_LOO_acc.npy', Qm_LOO_acc)
if do_gg_test: 
    print('saving gen gap test data in \'./saved_data\' dir...')
    np.save('./saved_data/VAE_MNIST_gg.npy', gg)

##############
#Plot Results#
##############

print('plotting and saving results in \'./images\' dir...')

NN_test_dict = {
    'd_vals': d_vals,
    'traces': [T_LOO_acc, Qm_LOO_acc, (T_LOO_acc + Qm_LOO_acc)/2], 
    'trace_names': ['$T$ acc', '$Q_m$ acc', 'Mean acc'], 
    'xlabel': '$\sigma$', 
    'ylabel': 'Accuracy', 
    'title': 'MNIST VAE: Two Sample NN Statistic', 
    'ref_value': 0.5, 
    'ELBO': VAE_dvals_ELBO, 
    'fname': 'NN_test_MNIST_vae.png'
}

plu.plot_VAE(**NN_test_dict)


ct_test_dict = {
    'd_vals': d_vals,
    'traces': [C_Ts],
    'trace_names': ['$C_T(P_n, Q_m)$'],
    'xlabel': '$\sigma$', 
    'ylabel': '$C_T(P_n, Q_m)',
    'title': 'MNIST VAE: $C_T$ vs. KDE $\sigma$',
    'ref_value': 0.0, 
    'ELBO': VAE_dvals_ELBO, 
    'fname': 'C_T_test_MNIST_vae.png'
}

plu.plot_VAE(**ct_test_dict)

gg_test_dict = {
    'd_vals': d_vals,
    'traces': [gg],
    'trace_names': ['ELBO$(T)$ - ELBO$(P_n)$'], 
    'xlabel': '$\sigma$', 
    'ylabel': 'generalization gap', 
    'title': 'MNIST VAE: Gen. Gap vs. KDE $\sigma$',
    'ref_value': 0.0, 
    'ELBO': VAE_dvals_ELBO, 
    'fname': 'gen_gap_test_MNIST_vae.png'
}

plu.plot_VAE(**gg_test_dict)
