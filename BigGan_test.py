#This module repeats the tests in Figure 4 (d) (e) of the paper: 
#C_T(P_n, Q_m) and Two Sample Nearest neighbor both on the BigGan
#generative model (Q)

#Turn on to generate images in ./BigGan/generated_images/
#ALl Qm's for all classes / trials / truncation levels are generated before testing
GENERATE_IMAGES = False 
IMAGENET_PTH = "/home/dobis_pr/UCSD_local/imgnet" 

#import BigGan utilities 
import sys
sys.path.append('./BigGan/')
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names,
                                         truncated_noise_sample, get_imagenet_mapping)
import BigGan_utils as bgu
import baselines as bln 
import data_copying_tests as dct 
import plot_utils as plu

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 

import os
from pathlib import Path

#########################
#####GENERATE SAMPLES####
#########################

#load pretrained model 
model = BigGAN.from_pretrained('biggan-deep-256').to('cuda')

#truncation levels to sweep
truncations = np.linspace(0.1, 1, 10)
#truncations = np.linspace(0.1, 1, 10)[::5]
n_trials = 5
#n_trials = 2 
classes = ['coffee', 'soap bubble', 'schooner']
n_samples = 2000
batch_size = 16
direc = os.path.abspath('./BigGan/generated_images') + '/'

if GENERATE_IMAGES: 
    #make saved_data directory if needed
    Path('./BigGan/generated_images').mkdir(parents = True, exist_ok = True)


    print('Generating {0:n} samples for each of 3 classes, {1:n} trials each'.format(n_samples, n_trials))

    for name in classes: 
        print('Generating images for class \'{0:s}\''.format(name))
        for trunc in tqdm(truncations): 
            for i in range(n_trials): 
                #generate samples
                samps,_,_ = bgu.generate_samples(model, name = name, truncation = trunc, 
                                             n_samples = n_samples, batch_size = batch_size)
                #save samples 
                fname = direc + name + '_p' + str(int(10*trunc)) + '_t' + str(int(i)) + '.npy'
                fname = fname.replace(' ', '_')
                np.save(fname, samps)


#################
####RUN TESTS####
#################

#Load images from imagenet 
data_transform = transforms.Compose([ transforms.Resize((256, 256)), transforms.ToTensor()])
train_set = ImageFolder(root = IMAGENET_PTH + "/ILSVRC2012_img_train/", transform = data_transform)
val_set = ImageFolder(root = IMAGENET_PTH + "/ILSVRC2012_img_val/", transform = data_transform)

#get mapping tools
imgnet_2_class = get_imagenet_mapping() #is a dict (key is imgnet id)
class_2_imgnet = np.array(list(imgnet_2_class.keys())) #is an array (index is class)

#parameters
n_clusters = 3

#make saved_data directory if needed
Path('./saved_data').mkdir(parents = True, exist_ok = True)

#list for storing each class' results
C_T_classes = []
T_LOO_acc_classes = []
Qm_LOO_acc_classes = []

for name in classes:
    print('Testing class \'{0:s}\'...'.format(name))

    #get one hot encoding of the class
    class_vec = one_hot_from_names([name], batch_size=1)
    img_class = np.argmax(class_vec, axis = 1)
    imgnet_id = class_2_imgnet[img_class][0]

    #Make a loader for this class
    class_id = train_set.class_to_idx['n0' + str(imgnet_id)]
    class_idx = np.arange(len(train_set.targets))[np.array(train_set.targets) == class_id]
    class_sampler = SubsetRandomSampler(class_idx)
    class_loader = DataLoader(train_set, batch_size = 64, shuffle = False, sampler = class_sampler)

    #now val loader
    class_idx_val = np.arange(len(val_set.targets))[np.array(val_set.targets) == class_id]
    class_sampler_val = SubsetRandomSampler(class_idx_val)
    class_loader_val = DataLoader(val_set, batch_size = 64, shuffle = False, sampler = class_sampler_val) 

    #Load train and test data
    n_train_samps = len(class_idx)
    print('n {0:s} training samples: {1:n}'.format(name, n_train_samps))
    train_data = bgu.get_imgnet_samps(n_train_samps, class_loader)

    n_val_samps = len(class_idx_val)
    print('n {0:s} validation samples: {1:n}'.format(name, n_val_samps))
    test_data = bgu.get_imgnet_samps(n_val_samps, class_loader_val)

    #preprocess train and test sets 
    print('Embedding train/test samples into inception space...')
    train_incep = bgu.incep_space(train_data, normalize_input = True) 
    test_incep = bgu.incep_space(test_data, normalize_input = True)

    #Compress to 64d 
    print('Making PCA projection...')
    pca_xf = PCA(n_components=64).fit(train_incep)
    train_pca = pca_xf.transform(train_incep)
    test_pca = pca_xf.transform(test_incep)

    #make Kmeans classifier 
    km_clf = KMeans(n_clusters = n_clusters).fit(train_pca)

    #store statistic data for this class
    C_T = np.zeros((len(truncations), n_trials))
    T_LOO_acc = np.zeros((len(truncations), n_trials))
    Qm_LOO_acc = np.zeros((len(truncations), n_trials))

    print('processing each truncation level and each trial')
    for trunc in tqdm(range(len(truncations))): 
        t = truncations[trunc]
        for trial in range(n_trials): 
            #get generated sample 
            gen_data = np.load(direc + name.replace(' ','_') +  \
                 '_p' + str(int(t*10))  + '_t' + str(int(trial)) + '.npy')

            #Put into inception space 
            gen_incep = bgu.incep_space(gen_data, normalize_input = False) 

            #PCA projection (our Qm)
            gen_pca = pca_xf.transform(gen_incep)

            #Perform C_T test (get per-cell data and aggregate data)
            T_labels = km_clf.predict(train_pca)
            Pn_labels = km_clf.predict(test_pca)
            Qm_labels = km_clf.predict(gen_pca)
            C_T[trunc, trial] = dct.C_T(test_pca, Pn_labels, gen_pca, 
                Qm_labels, train_pca, T_labels, tau = 20 / len(gen_pca))
            
            #Do two sample NN test
            #Unlike other tests, we have more generated samples than training samples 
            #here, there is a dirth of training/test data, so it is unburdensome to 
            #produce an equal number of generated samples (m = |T|). In other tests
            #as with MNIST (|T| = 50k), producing 50k generated samples every time 
            #we want to run this test is excessive, so we were forced to subsample
            #the training set. This is a more apples-to-apples comparison since
            #C_T does not require the training and generated samples to be of
            #equal size. 
            T_LOO_acc[trunc, trial], Qm_LOO_acc[trunc, trial] = \
                bln.NN_test(train_pca, gen_pca[:len(train_pca)], n_LOO = 1000)

    C_T_classes.append(C_T)
    T_LOO_acc_classes.append(T_LOO_acc)
    Qm_LOO_acc_classes.append(Qm_LOO_acc)

print('Finished running tests! Saving and plotting results')

C_T_classes = np.array(C_T_classes)
T_LOO_acc_classes = np.array(T_LOO_acc_classes)
Qm_LOO_acc_classes = np.array(Qm_LOO_acc_classes)

##############
#Save Results#
##############
np.save('./saved_data/IMGNET_biggan_C_Ts.npy', C_T_classes)
np.save('./saved_data/IMGNET_biggan_T_LOO_acc.npy', T_LOO_acc_classes)
np.save('./saved_data/IMGNET_biggan_Qm_LOO_acc.npy', Qm_LOO_acc_classes)

#Uncomment if results already saved and comment out everythin above
#except imports. It's a bit hacky, but gets the job done... 

#classes = ['coffee', 'soap bubble', 'schooner']
#truncations = np.linspace(0.1, 1, 10)
#
###############
##Load Results
###############
#T_LOO_acc_classes = np.load('./saved_data/IMGNET_biggan_T_LOO_acc.npy')
#Qm_LOO_acc_classes = np.load('./saved_data/IMGNET_biggan_Qm_LOO_acc.npy')
#C_T_classes = np.load('./saved_data/IMGNET_biggan_C_Ts.npy')

##############
#Plot Results#
##############
traces = np.concatenate((T_LOO_acc_classes, Qm_LOO_acc_classes, \
                            (T_LOO_acc_classes + Qm_LOO_acc_classes)/2), axis = 0)

trace_types = ['$T$ acc', '$Q_m$ acc', 'Mean acc']
trace_names0 = [classes[0] + trace_type for trace_type in trace_types]
trace_names1 = [classes[1] + trace_type for trace_type in trace_types]
trace_names2 = [classes[2] + trace_type for trace_type in trace_types]
trace_names = trace_names0 + trace_names1 + trace_names2 

#The plots in the paper have a slightly different display to make viewing
#the different trace types easier (e.g. markers, not log scale), 
#but the data is the same. 

#aslo note that the truncation levels passed to the BigGan model 
#need to be multiplied by 2. 

NN_test_dict = {
    'x_values': 2*truncations,
    'traces': traces,
    'trace_names': trace_names,
    'xlabel': 'Truncation Threshold', 
    'ylabel': 'Accuracy', 
    'title': 'ImageNet: NN Acc vs. GAN Input Var.',
    'ref_value': 0.5, 
    'fname': 'NN_test_ImageNet_GAN.png'
}

plu.plot_model_sweep(**NN_test_dict)



ct_test_dict = {
    'x_values': 2*truncations,
    'traces': C_T_classes,
    'trace_names': classes,
    'xlabel': 'Truncation Threshold',
    'ylabel': '$C_T(P_n, Q_m)',
    'title': 'ImageNet: $C_T$ vs. GAN Input Var.',
    'ref_value': 0.0, 
    'fname': 'C_T_test_ImageNet_GAN.png'
}

plu.plot_model_sweep(**ct_test_dict)

