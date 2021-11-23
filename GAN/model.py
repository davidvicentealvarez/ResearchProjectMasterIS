from torch import nn
import torch
import numpy as np


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            #initial size  [1, 216, 27]
            nn.Conv2d(in_channels=1, out_channels=4 , kernel_size=(3,1), stride=1),
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
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=(2,1), padding="valid"),
            # size [16, 105, 21]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 3), stride=(2,1), padding="valid"),
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
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 3), stride=(2, 1), output_padding=(1,0)),
            # size = [32, 24, 17]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(5, 3), stride=(2, 1)),
            # size [16, 51, 19]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(5, 3), stride=(2,1)),
            # size [16, 105, 21]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=(2,1), output_padding=(1,0)),
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
    input = torch.randn(10,1,216,27)
    g = PointGenerator()
    output = g(input)
    # data = np.load("brown_corpus.npy")
    # sample = torch.tensor(data[0:128])
    # sample = sample.view(128, 1, 216, 27)
    # sample = sample.float().to("cuda")
    # d = Discriminator().to("cuda")
    # output = d(sample)
    # print(output)
