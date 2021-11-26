from torch import nn
import torch
import numpy as np


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
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
        )

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(1, dictionary_size), stride=1),
            nn.Softmax(dim=3)
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(self.length,(self.length * self.dict_size)//4),
            nn.LeakyReLU(),
            nn.Linear((self.length * self.dict_size) // 4, (self.length * self.dict_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.length * self.dict_size) // 2, self.length * self.dict_size),
        )
        self.final_layer = nn.Softmax(dim=3)

    def forward(self, X):
        # tensor_1d = self.reduction_layer(X).view(-1, 4, self.length)
        # new_rep = self.convultion_layer(tensor_1d).view(-1, 4, self.length, 1)
        # return self.deconv_layer(new_rep)

        tensor_1d = self.reduction_layer(X).view(-1, 4, self.length)
        new_rep = self.convultion_layer(tensor_1d).view(-1,self.length)
        reconstruction = self.linear_layer(new_rep).view(-1, 1, self.length, self.dict_size)
        return self.final_layer(reconstruction)


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
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        tensor_1d = self.reduction_layer(X).view(-1, 4, self.length)
        new_rep = self.convultion_layer(tensor_1d).view(-1, 4 * self.length)
        return self.mlp_layer(new_rep)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # initial size  [1, 216, 27]
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 1), stride=1),
            # size [4, 214, 27]
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=1),
            # size [4, 214, 25]
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding="valid"),
            # size [8, 212, 23]
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=(2, 1), padding="valid"),
            # size [16, 105, 21]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding="valid"),
            # size [16, 51, 19]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 3), stride=(2, 1), padding="valid"),
            # size = [32, 24, 17]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 3), stride=(2, 1), padding="valid"),
            # size = [ 64, 10, 15]
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 3), stride=(2, 1),
                               output_padding=(1, 0)),
            # size = [32, 24, 17]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 3), stride=(2, 1)),
            # size [16, 51, 19]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(5, 3), stride=(2, 1)),
            # size [16, 105, 21]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=(2, 1), output_padding=(1, 0)),
            #  size [16, 105, 21]
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1),
            # size [4, 214, 25]
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=1),
            # size [4, 214, 27]
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(3, 1), stride=1),
            nn.LeakyReLU(),
            # size [ 1, 216, 27]
            nn.Softmax(dim=3)
        )

    def forward(self, X):
        # return self.encoder(X)
        return self.decoder(self.encoder(X))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # initial size  [1, 216, 27]
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 1), stride=1),
            # size [4, 214, 27]
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 3), stride=1),
            # size [4, 214, 25]
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding="valid"),
            # size [8, 212, 23]
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=(2, 1), padding="valid"),
            # size [16, 105, 21]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding="valid"),
            # size [16, 51, 19]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding="valid"),
            # size = [16, 24, 17]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding="valid"),
            # size = [ 16, 10, 15]
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * 10 * 15, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        encoded = self.encoder(X)
        return self.mlp(encoded.view((-1, 16 * 10 * 15)))


if __name__ == "__main__":
    data = np.load("brown_corpus.npy")
    sample = torch.tensor(data[0])
    sample= sample.view(1, 1, 216, 27)
    indices = torch.argmax(sample, dim=3)
    print(indices)
    # sample = sample.float()
    emb = nn.Embedding(27, 216)
    out = emb(indices)
    torch.matmul(out, sample)
    # d = GeneratorV2(216, 27)
    # output = d(sample)
    # print(output.size())
