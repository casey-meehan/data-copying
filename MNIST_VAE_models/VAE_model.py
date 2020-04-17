#!/usr/bin/env python3
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import sys
sys.path.append('../gan_test/PerceptualSimilarity')
import numpy as np

class VAE(nn.Module):
    def __init__(self, d = 50, l = 3):
        """builds VAE
        Inputs: 
            - d: dimension of latent space 
            - l: number of layers 
        """
        super(VAE, self).__init__()
        
        #Build VAE here 
        self.encoder, self.decoder, self.encoder_mean, self.encoder_lv = self.build_VAE(d, l)
        
        
    def build_VAE(self, d, l): 
        """builds VAE with specified latent dimension and number of layers 
        Inputs: 
            -d: latent dimension 
            -l: number of layers 
        """
        encoder_layers = []
        decoder_layers = []
        alpha = 3 / l 

        for lyr in range(l)[::-1]:
            lyr += 1
            dim_a = int(np.ceil(2**(alpha*(lyr+1))))
            dim_b = int(np.ceil(2**(alpha*lyr)))
            if lyr == l: 
                encoder_layers.append(nn.Linear(784, d * dim_b))
                encoder_layers.append(nn.ReLU())
                decoder_layers.insert(0, nn.Linear(d * dim_b, 784))
                decoder_layers.insert(0, nn.ReLU())
            else: 
                encoder_layers.append(nn.Linear(d * dim_a, d * dim_b))
                encoder_layers.append(nn.ReLU())
                decoder_layers.insert(0, nn.Linear(d * dim_b, d * dim_a))
                decoder_layers.insert(0, nn.ReLU())
        decoder_layers.insert(0, nn.Linear(d, d*int(np.ceil(2**(alpha))) ))

        encoder = nn.Sequential(*encoder_layers)
        decoder = nn.Sequential(*decoder_layers)

        encoder_mean = nn.Linear(d*int(np.ceil(2**(alpha))), d)
        encoder_lv = nn.Linear(d*int(np.ceil(2**(alpha))), d)
        return encoder, decoder, encoder_mean, encoder_lv
        

    def encode(self, x):
        """take an image, and return latent space mean + log variance
        Inputs: 
            -images, x, flattened to 784
        Outputs: 
            -means in latent dimension
            -logvariances in latent dimension 
        """
        h1 = self.encoder(x)
        return self.encoder_mean(h1), self.encoder_lv(h1)

    def reparameterize(self, mu, logvar):
        """Sample in latent space according to mean and logvariance
        Inputs: 
            -mu: batch of means
            -logvar: batch of logvariances
        Outputs: 
            -samples: batch of latent samples 
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """Decode latent space samples
        Inputs: 
            -z: batch of latent samples 
        Outputs: 
            -x_recon: batch of reconstructed images 
        """
        raw_out = self.decoder(z)
        return torch.sigmoid(raw_out)

    def forward(self, x):
        """Do full encode and decode of images
        Inputs: 
            - x: batch of images 
        Outputs: 
            - recon_x: batch of reconstructed images
            - mu: batch of latent mean values 
            - logvar: batch of latent logvariances 
        """
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar