#This module produces Figure 3 (d)(e)(f) as seen in the paper. Specifically, 
#reproduces the KDE tests on the MNIST dataset. This involves compressing 
#MNIST images to a lower-dimensional latent space in which the distance 
#between images is symantically meaningful. 

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from pathlib import Path
import sys
from tqdm import tqdm 
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity as KDE

import data_copying_tests as dct 
import baselines as bln 
sys.path.append('./MNIST_autoencoder/')
from autoencoder import percep_autoencoder as pae


#functions for going from image to embedding and back 
def pae_codes(samples, pae_model): 
    """returns latent codes from perceptual AE"""

    imgs = torch.Tensor(samples).unsqueeze(1).cuda()
    n_samples = len(samples)
    pae_dim = pae_model.encode(imgs[0:1]).view(1, -1).cpu().data.numpy().size
    codes = np.zeros((n_samples, pae_dim))

    #make samples in batches of 128
    for i in range(int(n_samples / 128)): 
        codes[i*128 : (i+1)*128] = pae_model.encode(imgs[i*128 : (i+1)*128]).view(128, -1).cpu().data.numpy()

    #finish the rest of the samples
    if n_samples < 128: 
        i = 0
    else: 
        i+=1

    rem = n_samples - i*128
    codes[i*128 : ] = pae_model.encode(imgs[i*128 : ]).view(rem, -1).cpu().data.numpy()
    return codes 

def pae_imgs(samples, pae_model): 
    """returns imgs from latent samples using perceptual AE"""

    codes = torch.Tensor(samples).unsqueeze(1).cuda()
    n_samples, pae_dim = samples.shape
    imgs = np.zeros((n_samples, 28, 28))

    #make samples in batches of 128
    for i in range(int(n_samples / 128)): 
        imgs[i*128 : (i+1)*128] = pae_model.decode(codes[i*128 : (i+1)*128]).data.cpu().numpy().squeeze()

    #finish the rest of the samples
    if n_samples < 128: 
        i = 0
    else: 
        i+=1

    imgs[i*128 : ] = pae_model.decode(codes[i*128 : ]).data.cpu().numpy().squeeze()
    return imgs


###################
#Gather MNIST data# 
###################

print('Loading MNIST data...') 
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../mnist_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

#load full dataset into memory to simplify making splits 
train_data = []
train_targs = []
for batch_idx, (data, target) in enumerate(train_loader):
    train_data.append(data.squeeze().view(len(data), -1).numpy())
    train_targs.append(target.numpy())

train_data = np.concatenate(train_data, axis = 0)
train_data = train_data.reshape(len(train_data), 28, 28)
train_targs = np.concatenate(train_targs, axis = 0)

test_data = [data.squeeze().view(len(data), -1).numpy() for batch_idx, (data, target) in enumerate(test_loader)]
test_data = np.concatenate(test_data, axis = 0)
test_data = test_data.reshape(len(test_data), 28, 28)

#Make validation split for tuning sigma 
n_samps = len(test_data)
val_data = train_data[:n_samps]
val_targs = train_targs[:n_samps]
train_data = train_data[n_samps:]
train_targs = train_targs[n_samps:]

l = len(train_data)
n = len(test_data)

print('Train Data Size:', l)
print('Val Data Size:', n)
print('Test Data Size:', len(test_data))


################################
#Get latent code for each split#
################################


#set up 64-dim autoencoder
print('Loading autoencoder...') 
pae_64 = pae(d=4).cuda().eval()
pae_64.load_state_dict(torch.load(
    './MNIST_autoencoder/trained_autoencoder_weights.pth', map_location = 'cuda'))

print('Embedding train/test/val images...') 
train_z_64 = pae_codes(train_data, pae_64)
val_z_64 = pae_codes(val_data, pae_64)
test_z_64 = pae_codes(test_data, pae_64)

#get instance space partition
print('Getting instance space partition $\Pi$...')
n_clusters = 50
KM_clf = KMeans(n_clusters).fit(train_z_64)

#sigmas to test 
n_sigmas = 50
sigmas = np.logspace(start = -3, stop = 0, num = n_sigmas)

######################################################
#Get KDE likelihood at each sigma (on validation set)# 
######################################################

##make saved_data directory if needed
#Path('./saved_data').mkdir(parents = True, exist_ok = True)
##collect likelihoods if not already there 
#likelihood_data = Path('./saved_data/KDE_MNIST_likelihoods.npy') 
#if likelihood_data.is_file(): 
#    print('Using saved likelihood data in \'./saved_data/\' dir')
#    likelihoods = np.load('./saved_data/KDE_MNIST_likelihoods.npy') 
#else: 
#    print('Likelihood data not found. Measuring KDE likelihoods at {0:n} \
#sigma values and saving it in \'./saved_data\' dir. \
#This may take a while...'.format(n_sigmas)) 
#    likelihoods = []
#    for sigma in tqdm(sigmas): 
#        kde_sigma = KDE(sigma).fit(train_z_64)
#        likelihood = kde_sigma.score(val_z_64)
#        likelihoods.append(likelihood)
#    np.save('./saved_data/KDE_MNIST_likelihoods.npy', likelihoods)

#####################################################
#Run C_T test at each sigma value for several trials# 
#####################################################
n_trials = 5

#get train and test cell labels 
Pn_labels = KM_clf.predict(test_z_64)
T_labels = KM_clf.predict(train_z_64)

#allocate space for statistics 
#C_T test
C_Ts = np.zeros((n_sigmas, n_trials))
#Two sample NN test
T_LOO_acc = np.zeros((n_sigmas, n_trials))
Qm_LOO_acc = np.zeros((n_sigmas, n_trials))
#Generalization gap test
#this test takes quite a while (requires computing likelihood with high dim model) 
run_gg_tests = False 
gg = np.zeros((n_sigmas, 1))

#for each sigma value...
print('Gathering C_T, 2-sample NN, and gen. gap stats at {0:n} sigma values, {1:n} trials \
each'.format(n_sigmas, n_trials))

for sig_idx in tqdm(range(n_sigmas)): 
    #train Q
    sigma = sigmas[sig_idx]
    Q = KDE(bandwidth = sigma, kernel = 'gaussian', metric = 'euclidean').fit(train_z_64)

    if run_gg_tests: 
        #gather gen. gap test statistic 
        gg[sig_idx] = bln.gen_gap(test_z_64, train_z_64, Q)

    #for each trial...
    for trial_idx in range(n_trials): 
        #generate the same number of samples as the test data (10k), m = n
        Qm = Q.sample(n_samples = n) 
        Qm_labels = KM_clf.predict(Qm)

        #gather C_T statistic 
        C_Ts[sig_idx, trial_idx] = dct.C_T(test_z_64, Pn_labels, Qm, Qm_labels,
            train_z_64, T_labels, tau = 20 / len(Qm)) 

        #gather NN test statistics 
        #first subsample T to be the same size as m
        T_tilde = train_z_64[np.random.choice(np.arange(l), size = n, replace = False)]
        #run test setting n_LOO = 1000 to speed things up (minimal effect on outcome)
        T_LOO_acc[sig_idx, trial_idx], Qm_LOO_acc[sig_idx, trial_idx] = bln.NN_test(T_tilde,Qm, n_LOO = 1000)

