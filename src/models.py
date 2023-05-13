import random
import numpy as np

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler

def mish(x):
    return (x*torch.tanh(F.softplus(x)))

class Autoencoder(nn.Module):
    
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder_1 = nn.Linear(input_size, 106)
        self.encoder_2 = nn.Linear(106, 56)
        self.encoder_3 = nn.Linear(56, latent_size)

        self.decoder_1 = nn.Linear(latent_size, 56)
        self.decoder_2 = nn.Linear(56, 106)
        self.decoder_3 = nn.Linear(106, input_size)


    def forward(self, x):
        # Encoding
        x = mish(self.encoder_1(x))
        x = mish(self.encoder_2(x))
        x = mish(self.encoder_3(x))

        #Decoding
        x = mish(self.decoder_1(x))
        x = mish(self.decoder_2(x))
        return mish(self.decoder_3(x))



class QuantumAutoencoder(nn.Module):
    
    def __init__(self, input_size, latent_size, quantum_classical):
        super(QuantumAutoencoder, self).__init__()
        self.encoder_1 = nn.Linear(input_size, 106)
        self.encoder_2 = nn.Linear(106, 56)
        self.encoder_3 = nn.Linear(56, latent_size)
        
        self.hybrid = quantum_classical(latent_size)
        
        self.decoder_1 = nn.Linear(latent_size, 56)
        self.decoder_2 = nn.Linear(56, 106)
        self.decoder_3 = nn.Linear(106, input_size)

    def forward(self, x):
        # Encoding
        x = mish(self.encoder_1(x))
        x = mish(self.encoder_2(x))
        x = mish(self.encoder_3(x))
        
        x_out = []
        for _, x_in in enumerate(x):
            x_out.append(self.hybrid(x_in))
        x = torch.cat(x_out, dim=0)
        
        #Decoding
        x = mish(self.decoder_1(x))
        x = mish(self.decoder_2(x))
        return mish(self.decoder_3(x))