from torch import nn
import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from data import convert_data

class ResNetLayer(nn.Module):
    def __init__(self,filters, filter_size = 3):
        super(ResNetLayer, self ).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels=filters, out_channels=filters,  kernel_size=filter_size, stride=1, padding="same", padding_mode="reflect"),
            nn.InstanceNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=filter_size, stride=1, padding="same", padding_mode="reflect"),
            nn.InstanceNorm2d(filters)
        )
        self.relu = nn.ReLU()

    def forward(self, X):
        residual = X
        result = self.resnet(X)
        return self.relu(result + residual)





class GeneratorV3(nn.Module):
    def __init__(self, length, dictionary_size):
        super(GeneratorV3, self).__init__()
        self.filters = 32
        self.filter_size = 3
        self.length = length
        self.dictionary_size = dictionary_size
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.filters, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(self.filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=self.filter_size, stride=2, padding="valid"),
            nn.InstanceNorm2d(self.filters*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters*2, out_channels=self.filters * 4, kernel_size=self.filter_size, stride=2, padding="valid"),
            nn.InstanceNorm2d(self.filters * 4),
            nn.ReLU()
        )
        self.resnet_layer = nn.Sequential(
            ResNetLayer(self.filters * 4),
            ResNetLayer(self.filters * 4),
            ResNetLayer(self.filters * 4),
            ResNetLayer(self.filters * 4),
            ResNetLayer(self.filters * 4),
            ResNetLayer(self.filters * 4),
            ResNetLayer(self.filters * 4),
            ResNetLayer(self.filters * 4),
            ResNetLayer(self.filters * 4),
        )

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.filters*4, out_channels=self.filters * 2, kernel_size=self.filter_size, stride=2),
            nn.InstanceNorm2d(self.filters * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.filters * 2, out_channels=self.filters, kernel_size=self.filter_size, stride=2, output_padding=(1,1)),
            nn.InstanceNorm2d(self.filters),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.filters, out_channels=self.filters, kernel_size=self.filter_size, stride=2),
            nn.InstanceNorm2d(self.filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters, out_channels=1, kernel_size=self.filter_size, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Linear(self.length, self.dictionary_size)
        )
        self.softmax = nn.Softmax(dim=3)

    def forward(self, X, logits=True):
        reconstruction = self.deconv_layer(self.resnet_layer(self.conv_layer(X)))
        if not logits:
            reconstruction = self.softmax(reconstruction)
        return reconstruction



class DiscriminatorV3(nn.Module):
    def __init__(self,  length, dictionary_size):
        super(DiscriminatorV3, self).__init__()
        self.length = length
        self.dictionary_size = dictionary_size
        self.filter_size = 4
        self.filters = 64
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.filters, kernel_size=self.filter_size, stride=2, padding=1),
            nn.InstanceNorm2d(self.filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2,  kernel_size=self.filter_size, stride=2, padding=1),
            nn.InstanceNorm2d(self.filters*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters*2, out_channels=self.filters*4, kernel_size=self.filter_size, stride=2),
            nn.InstanceNorm2d(self.filters*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters*4, out_channels=self.filters*8,  kernel_size=self.filter_size, stride=2),
            nn.InstanceNorm2d(self.filters*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters*8, out_channels=1, kernel_size=self.filter_size, stride=1)
        )
        self.linear = nn.Linear(self.filters*8 * 6 * 3, 1)

    def forward(self, X):
        return self.conv_layer(X).view(-1,1)



if __name__  == "__main__":
    data = convert_data(fixed_len=100)
    sample = torch.tensor(data[:1])
    sample = sample.view(-1, 1, 100, 27).float()
    lin = nn.Linear(27,100)
    g = DiscriminatorV3(100,27)
    out = g(lin(sample))
    print(out.size())

