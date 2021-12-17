from torch import nn
from data import convert_data
import torch


class GeneratorV2(nn.Module):
    def __init__(self, length, dictionary_size):
        super().__init__()
        self.dict_size = dictionary_size
        self.length = length
        # initial size  [1, length, dictionary_size]
        self.reduction_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, dictionary_size), stride=1),
            nn.ReLU()
        )

        self.convultion_layer = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=1, stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(1, dictionary_size), stride=1),
            nn.Softmax(dim=3)
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(self.length*32,(self.length * self.dict_size)//4),
            nn.LeakyReLU(),
            nn.Linear((self.length * self.dict_size) // 4, (self.length * self.dict_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.length * self.dict_size) // 2, self.length * self.dict_size),
            nn.ReLU()
        )
        self.final_layer = nn.Softmax(dim=3)

    def forward(self, X, logits=True):
        # tensor_1d = self.reduction_layer(X).view(-1, 4, self.length)
        # new_rep = self.convultion_layer(tensor_1d).view(-1, 4, self.length, 1)
        # return self.deconv_layer(new_rep)

        tensor_1d = self.reduction_layer(X).view(-1, 4, self.length)
        new_rep = self.convultion_layer(tensor_1d).view(-1, 1, self.length *32)
        reconstruction = self.linear_layer(new_rep).view(-1, 1, self.length, self.dict_size)
        if not logits:
            reconstruction = self.final_layer(reconstruction)
        return reconstruction


class DiscriminatorV2(nn.Module):
    def __init__(self, length, dictionary_size):
        super().__init__()
        self.dict_size = dictionary_size
        self.length = length
        # initial size  [1, length, dictionary_size]
        self.reduction_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, dictionary_size), stride=1),
            nn.ReLU()
        )

        self.convultion_layer = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.mlp_layer = nn.Sequential(
            nn.Linear(4 * self.length, 2048),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, X):
        tensor_1d = self.reduction_layer(X).view(-1, 4, self.length)
        new_rep = self.convultion_layer(tensor_1d).view(-1, 1, 4 * self.length)
        return self.mlp_layer(new_rep).view(-1,1)
