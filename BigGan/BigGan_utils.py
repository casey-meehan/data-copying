#These are a variety of utility functions I've put together to make
#working with the BigGan model easier. 

import sys
sys.path.append('./BigGan/')
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names,
                                         truncated_noise_sample, get_imagenet_mapping)
from inception import InceptionV3
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from tqdm import tqdm 

sys.path.append('../')
import baselines as bln
import data_copying_tests as dct

def generate_samples(model, name = 'coffee', truncation = 0.5, n_samples = 2000, batch_size = 16): 
    """Generate Biggan samples of a certain class. n_samples must be divisible by batch_size
    Inputs: 
        - model: biggan model 
        - name: name of class 
        - truncation: noise truncation value (between 0 and 1)
        - n_samples: number of images to generate 
        - batch_size: samples per batch 
    Outputs: 
        - samples: (n_samples, 3, 256, 256) numpy array of images 
        - imgnet_ids: ID of class to be foung in imgnet 
        - img_classes: one hot vector of biggan img class (of 1000)
    """
    # Prepare an input
    class_vec = one_hot_from_names([name], batch_size=1)

    if not np.all(np.sum(class_vec, axis = 1) == 1): 
        print('One of these types has multiple classes in it')
        return 
    else: 
        imgnet_2_class = get_imagenet_mapping() #is a dict (key is imgnet id)
        class_2_imgnet = np.array(list(imgnet_2_class.keys())) #is an array (index is class)
        img_classes = np.argmax(class_vec, axis = 1)
        imgnet_ids = class_2_imgnet[img_classes]

    #generate samples 
    outputs = []
    model.to('cuda')

    for i in range(int(n_samples / batch_size)): 

        class_vector = one_hot_from_names([name], batch_size=batch_size)
        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)

        noise_vector = torch.from_numpy(noise_vector)
        class_vector = (torch.from_numpy(class_vector))


        #put everything on cuda
        noise_vector = noise_vector.to('cuda')
        class_vector = class_vector.to('cuda')


        with torch.no_grad():
            output = model(noise_vector, class_vector, truncation)
            outputs.append(output.cpu().numpy())

    outputs = np.concatenate(outputs)

    return outputs, imgnet_ids, img_classes

def incep_space(imgs, normalize_input):
    """takes array of [N, W, H] images and returns array of [N, 2048] in inception space
    DONT NORMALIZE GENERATED IMGS
    DO NORMALIZE IMGNET IMGS
    """
    maploc = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(maploc)
    iv3 = InceptionV3(output_blocks = [3], normalize_input=normalize_input)
    iv3.to(device)
    
    n_samples = imgs.shape[0]
    inceps = np.zeros((n_samples, 2048)) #for output block 3
    w = imgs.shape[1]
    h = imgs.shape[2]

    #make samples in batches of 128
    with torch.no_grad(): 
        for i in range(int(n_samples / 128)): 
            img_batch = torch.Tensor(imgs[i*128 : (i+1)*128]).cuda()
            inceps[i*128 : (i+1)*128] = iv3(img_batch)[0].squeeze().cpu().numpy()

    #finish the rest of the samples
    if n_samples < 128: 
        i = 0
    else: 
        i+=1

    img_batch = torch.Tensor(imgs[i*128 :]).cuda()
    inceps[i*128 :] = iv3(img_batch)[0].squeeze().cpu().numpy()
    return inceps

def get_imgnet_samps(n_samps, loader): 
    """gets n_samps from imgnet loader returns as np array [N, C, H, W]"""
    btch_sz = loader.batch_size
    samps = np.zeros((n_samps, 3, 256, 256))
    n_batches = np.floor(n_samps / btch_sz)
    for i, samps_targs in enumerate(loader):  
        if (i+1 <= n_batches): 
            samps[i*btch_sz : (i+1)*btch_sz] = samps_targs[0].numpy()
        else: 
            samps[i*btch_sz:] = samps_targs[0][0:n_samps - i*btch_sz].numpy()
            break
    return samps


