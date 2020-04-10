import torch
import torchvision
from torch import nn
from torch.nn import functional as F
#import sys
#sys.path.append('./perceptual_similarity')
#from models import dist_model as dm
#import dist_model as dm
import perceptual_similarity.dist_model as dm

class percep_autoencoder(nn.Module):
    def __init__(self, lam = 0.1, d = 4):
        super(percep_autoencoder, self).__init__()
        self.d = d
        self.lam = lam 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, d*2, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.BatchNorm2d(d*2),
            nn.Conv2d(d*2, d*2, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.BatchNorm2d(d*2),
            nn.Conv2d(d*2, d*4, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.BatchNorm2d(d*4),
            nn.Conv2d(d*4, d*4, 3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2)
        )

        #then flatten features 
        self.bottle = nn.Sequential(
            nn.Linear(d*4*7*7, d*4*7*7),
            nn.ReLU(), 
            nn.Dropout(p = 0.5), 
            nn.Linear(d*4*7*7, d*4*4)
        )

        self.unbottle = nn.Sequential(
            nn.Linear(d*4*4, d*4*7*7),
            nn.ReLU(), 
            nn.Dropout(p = 0.5), 
            nn.Linear(d*4*7*7, d*4*7*7)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear'),

            nn.BatchNorm2d(d*4),
            nn.Conv2d(d*4, d*4, 3, stride=1, padding=1), 
            nn.ReLU(),

            nn.BatchNorm2d(d*4),
            nn.Conv2d(d*4, d*2, 3, stride=1, padding=1), 
            nn.ReLU(),

            nn.Upsample(scale_factor = 2, mode='bilinear'),

            nn.BatchNorm2d(d*2),
            nn.Conv2d(d*2, d*2, 3, stride=1, padding=1), 
            nn.ReLU(),

            nn.BatchNorm2d(d*2),
            nn.Conv2d(d*2, 1, 3, stride=1, padding=1), 
            nn.Tanh()
        )

        #Build perceptual loss vgg model 
        self.vgg_percep = dm.DistModel()
        #(load vgg network weights -- pathing is relative to top level) 
        self.vgg_percep.initialize(model='net-lin', net='vgg', 
            model_path = './MNIST_autoencoder/perceptual_similarity/weights/vgg.pth', use_gpu=True, spatial=False)

    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)
        return x_, z

    def encode(self, x): 
        batch_size = x.shape[0]
        enc = self.encoder(x)
        z = self.bottle(enc.view(batch_size, -1))
        return z

    def decode(self, z): 
        batch_size = z.shape[0]
        dec = self.unbottle(z)
        x_ = self.decoder(dec.view(batch_size, self.d*4, 7, 7))
        return x_

    def perceptual_loss(self, im0, im1, z): 
        """computes loss as perceptual distance between image x0 and x1
        and adds squared loss of the latent z norm"""
        batch_size = im0.shape[0]
        im0 = im0.expand(batch_size, 3, 28, 28)
        im1 = im1.expand(batch_size, 3, 28, 28)
        percep_dist = self.vgg_percep.forward_pair(im0, im1)
        z_norms = (z.view(batch_size, -1)**2).sum(dim=1)
        latent_penalty =  self.lam * (F.relu(z_norms - 1))
        return (percep_dist + latent_penalty).sum()
        


