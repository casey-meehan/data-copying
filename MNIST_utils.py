import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

import sys 
sys.path.append('./MNIST_autoencoder/')
import os
from autoencoder import percep_autoencoder as pae

#functions for going from image to embedding and back 
def pae_codes(samples):#, pae_model): 
    """returns latent codes from perceptual AE"""

    #load up autoencoder
    sys.stdout = open(os.devnull, 'w') #silence torch's default print statements
    pae_model = pae(d=4).cuda().eval()
    pae_model.load_state_dict(torch.load(
    './MNIST_autoencoder/trained_autoencoder_weights.pth', map_location = 'cuda'))
    sys.stdout = sys.__stdout__ #reinstate printing...

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

    #load up autoencoder
    pae_model = pae(d=4).cuda().eval()
    pae_model.load_state_dict(torch.load(
    './MNIST_autoencoder/trained_autoencoder_weights.pth', map_location = 'cuda'))

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

def get_mnist_data(val_data_size = 10000):
    """Produces a train / test / validation split of the MNIST data

    Inputs: 
        val_data_size: (int) size of validation dataset (taken 
            from training set)

    Outputs: 
        train_data: (num_training X 28 X 28) np array of training 
            samples where num_training = 60,000 - val_data_size

        val_data: (val_data_size X 28 X 28) np array of validation
            data samples. 

        test_data: (10000 X 28 X 28) np array of the standard MNIST
            test set
    """
    if val_data_size > 25000: 
        raise ValueError('the validataion data size, {0:n}, is much too large \
        keep in mind there are only 60k training examples to draw from'.format(val_data_size))

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
    for batch_idx, (data, target) in enumerate(train_loader):
        train_data.append(data.squeeze().view(len(data), -1).numpy())

    train_data = np.concatenate(train_data, axis = 0)
    train_data = train_data.reshape(len(train_data), 28, 28)

    test_data = [data.squeeze().view(len(data), -1).numpy() for batch_idx, (data, target) in enumerate(test_loader)]
    test_data = np.concatenate(test_data, axis = 0)
    test_data = test_data.reshape(len(test_data), 28, 28)

    #Make validation split 
    val_data = train_data[:val_data_size]
    train_data = train_data[val_data_size:]


    print('Train Data Size:', len(train_data))
    print('Val Data Size:', len(val_data))
    print('Test Data Size:', len(test_data))

    return train_data, val_data, test_data


