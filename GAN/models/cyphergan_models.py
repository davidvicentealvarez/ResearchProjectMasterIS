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



class ResNetLayer1D(nn.Module):
    def __init__(self,filters, filter_size = 3):
        super(ResNetLayer1D, self ).__init__()
        self.resnet = nn.Sequential(
            nn.Conv1d(in_channels=filters, out_channels=filters,  kernel_size=filter_size, stride=1, padding="same"),
            nn.InstanceNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=filter_size, stride=1, padding="same"),
            nn.InstanceNorm1d(filters)
        )
        self.relu = nn.ReLU()

    def forward(self, X):
        residual = X
        result = self.resnet(X)
        return self.relu(result + residual)


class GeneratorV4(nn.Module):
    def __init__(self, length, dictionary_size):
        super(GeneratorV4, self).__init__()
        self.length = length
        self.dictionary_size = dictionary_size
        self.filter_size = 1
        self.filters = 32
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=self.dictionary_size, out_channels=self.filters, kernel_size=self.filter_size, stride=1, padding="same"),
            nn.InstanceNorm1d(self.filters),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters, out_channels=self.filters * 2, kernel_size=self.filter_size, stride=1, padding="same"),
            nn.InstanceNorm1d(self.filters * 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters * 2, out_channels=self.filters * 4, kernel_size=self.filter_size, stride=1, padding="same"),
            nn.InstanceNorm1d(self.filters * 4),
            nn.ReLU(),
            ResNetLayer1D(self.filters * 4, filter_size=1),
            ResNetLayer1D(self.filters * 4, filter_size=1),
            ResNetLayer1D(self.filters * 4, filter_size=1),
            ResNetLayer1D(self.filters * 4, filter_size=1),
            ResNetLayer1D(self.filters * 4, filter_size=1),
            nn.Conv1d(in_channels=self.filters * 4, out_channels=dictionary_size, kernel_size=self.filter_size, stride=1, padding="same"),
        )
        self.softmax= nn.Softmax(1)

    def forward(self, X, logits=True):
        new_x = self.net(X.view(-1, self.length, self.dictionary_size).transpose(1,2))
        if not logits:
            new_x = self.softmax(new_x)
        return new_x.transpose(1,2).view(-1, 1, self.length, self.dictionary_size)


class DiscriminatorV4(nn.Module):
    def __init__(self, length, dictionary_size):
        super(DiscriminatorV4, self).__init__()
        self.length = length
        self.dictionary_size = dictionary_size
        self.filter_size = 15
        self.filters = 32
        self.droprate = 0.5
        self.net = nn.Sequential(
            nn.Dropout(self.droprate),
            nn.Conv1d(in_channels=self.dictionary_size, out_channels=self.filters, kernel_size=1, stride=2, padding="valid"),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=1, stride=2, padding="valid"),
            nn.InstanceNorm1d(self.filters*2),
            # nn.LayerNorm(25),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters * 2, out_channels=self.filters * 4, kernel_size=1, stride=2, padding="valid"),
            nn.InstanceNorm1d(self.filters * 4),
            # nn.LayerNorm(13),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters * 4, out_channels=self.filters * 8, kernel_size=1, stride=2, padding="valid"),
            nn.InstanceNorm1d(self.filters * 8),
            # nn.LayerNorm(7),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters * 8, out_channels=self.filters * 16, kernel_size=1, stride=2, padding="valid"),
            nn.InstanceNorm1d(self.filters * 2),
            # nn.LayerNorm(4),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters * 16, out_channels=self.filters * 32, kernel_size=1, stride=2, padding="valid"),
            nn.InstanceNorm1d(self.filters * 32),
            # nn.LayerNorm(2),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters * 32, out_channels=1, kernel_size=2, stride=1,padding="valid"),
        )

    def forward(self, X):
        new_x = self.net(X.view(-1, self.length, self.dictionary_size).transpose(1, 2))
        return new_x.view(-1,1)



class GeneratorV3(nn.Module):
    def __init__(self, length, dictionary_size):
        super(GeneratorV3, self).__init__()
        self.filters = 32
        self.filter_size = 3
        self.length = length
        self.dictionary_size = dictionary_size
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.filters, kernel_size=7, stride=1, padding="same", padding_mode="reflect"),
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
            nn.ConvTranspose2d(in_channels=self.filters * 2, out_channels=self.filters, kernel_size=self.filter_size, stride=2, output_padding=(1,0)),
            nn.InstanceNorm2d(self.filters),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.filters, out_channels=self.filters, kernel_size=self.filter_size, stride=2),
            nn.InstanceNorm2d(self.filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters, out_channels=1, kernel_size=self.filter_size, stride=2, padding="valid"),
            nn.ReLU(),
            # nn.Linear(self.length, self.dictionary_size)
        )
        self.softmax = nn.Softmax(dim=3)

    def forward(self, X, logits=True):
        reconstruction = self.deconv_layer(self.resnet_layer(self.conv_layer(X)))
        if not logits:
            reconstruction = self.softmax(reconstruction)
        return reconstruction



class DiscriminatorV3(nn.Module):
    def __init__(self,  length, dictionary_size, full_size = False):
        super(DiscriminatorV3, self).__init__()
        self.full_size = full_size
        self.length = length
        self.dictionary_size = dictionary_size
        self.filter_size = 4
        self.filters = 64
        self.conv_layer_100 = nn.Sequential(
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

        self.conv_layer_27 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.filters, kernel_size=(self.filter_size * 8, self.filter_size), stride=2, padding="valid"),
            nn.InstanceNorm2d(self.filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters, out_channels=self.filters * 2, kernel_size=(self.filter_size * 4, self.filter_size), stride=1, padding="valid"),
            nn.InstanceNorm2d(self.filters * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters * 2, out_channels=self.filters * 4, kernel_size=(self.filter_size * 2, self.filter_size), stride=1, padding="valid"),
            nn.InstanceNorm2d(self.filters * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters * 4, out_channels=self.filters * 8,  kernel_size=(self.filter_size * 2, self.filter_size), stride=1, padding="valid"),
            nn.InstanceNorm2d(self.filters * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters * 8, out_channels=1, kernel_size=(6, 3), stride=1, padding="valid"),
        )

    def forward(self, X):
        if self.full_size:
            return self.conv_layer_100(X).view(-1, 1)
        else :
            return self.conv_layer_27(X).view(-1, 1)



if __name__  == "__main__":
    data = convert_data(fixed_len=100)
    sample = torch.tensor(data[:1])
    sample = sample.view(-1, 1, 100, 27).float()
    lin = nn.Linear(27,100)
    g = DiscriminatorV3(100,27)
    out = g(sample)
    print(out.size())


