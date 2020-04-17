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

def sweep_biggan_tests(train_data, test_data, truncations, n_trials = 5, name = 'coffee', n_clusters = 3): 
    """sweeps hypothesis tests for samples (gathered beforehand) of 5 trials and 10 noise levels
    Inputs: 
        - train_data: imgnet train data for this class (n_imgs, 3, 256, 256)
        - test_data: imgnet test data for this class (n_imgs, 3, 256, 256)
        - truncations: (1 X n truncations) np array of truncation levels to test 
        - name: name of class to search for in imgnet and class id's 
        - n_clusters: number of clusters to form on instance space 
    Outputs: 
        - stats: statistics gathered at each iteration (list of 10 lists of 5 dicts with all stats for sample)
    """
    pth = os.path.abspath('./generated_imgs') + '/'
    #preprocess train and test sets 
    print('Embedding train/test samples into inception space...')
    train_incep = incep_space(train_data, normalize_input = True) 
    test_incep = incep_space(test_data, normalize_input = True)

    #Compress to 64d 
    print('Making PCA projection...')
    pca_xf = PCA(n_components=64).fit(train_incep)
    train_pca = pca_xf.transform(train_incep)
    test_pca = pca_xf.transform(test_incep)

    #make Kmeans classifier 
    km_clf = KMeans(n_clusters = n_clusters).fit(train_pca)

    stats = []

    print('processing each noise val and each trial')
    for t in tqdm(truncations): 
        stats_t = []
        for trial in range(n_trials): 
            #get generated sample 
            gen_data = np.load(pth + name.replace(' ','_') +  \
                '_p' + str(int(t*10))  + '_t' + str(int(trial)) + '.npy')

            #Put into inception space 
            gen_incep = incep_space(gen_data, normalize_input = False) 

            #PCA project
            gen_pca = pca_xf.transform(gen_incep)

            #Perform C_T test (get per-cell data and aggregate data)
            u, z, p = cell_MW_test(train_pca, test_pca, gen_pca, km_clf, isplot = False)
            ct, sv = cell_M_tilde(train_pca, test_pca, gen_pca, km_clf, tau = 20 / len(gen_pca))

            #Perform NN test
            train_acc, gen_acc = NN_test(train_pca, gen_pca[:len(train_pca)], n_LOO = 1000)

            sample_stats = {
                'C_T': ct, #averaged C_T(P_n, Q_m) score 
                'Z_U': z, #cell-by-cell man-whitney z-scores 
                'NDB_ov': NDB_ov, #cell-by-cell overrepresentation scores 
                'NDB_un': NDB_un, #cell-by-cell overrepresentation scores 
                'NN_train_acc': train_acc, #Nearest Neighbor test
                'NN_gen_acc': gen_acc #Nearest Neighbor test
            }

            stats_nv.append(sample_stats)

        stats.append(stats_nv)
        
    return stats
