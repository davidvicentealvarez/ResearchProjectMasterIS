from torch import nn
from torch.nn import functional as F
from data import convert_data
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, length, dictionary_size):
        super().__init__()
        self.length = length
        self.dictionary_size = dictionary_size
        self.encoder = nn.Sequential(
            nn.Linear(self.dictionary_size, self.length),
            nn.ReLU(),
            # initial size  [1, 100, 100]
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=11, stride=1),
            # size [4, 90, 90]
            nn.InstanceNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=11, stride=1),
            # size [4, 80, 80]
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=11, stride=1),
            # size [12, 70, 70]
            nn.InstanceNorm2d(12),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=11, stride=1),
            # size [16, 60, 60]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size [20, 50, 50]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [24, 40, 40]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [ 28, 30, 30]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [ 32, 20, 20]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            # size = [32, 20, 20]
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [28, 30, 30]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size [24, 40, 40]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size [20, 50, 50]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            #  size [16, 60, 60]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=12, kernel_size=11, stride=1),
            # size [4, 70, 70
            nn.InstanceNorm2d(12),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=8, kernel_size=11, stride=1),
            # size [4, 80, 80]
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=11, stride=1),
            nn.LeakyReLU(),
            nn.InstanceNorm2d(4),
            # size [ 1, 90, 90]
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=11, stride=1),
            nn.InstanceNorm2d(1),
            # size [ 1, 100, 100]
            nn.ReLU(),
            nn.Linear(self.length, self.dictionary_size)
        )
        # self.embeding = nn.Linear(self.dictionary_size, self.length)
        # self.embeding_reverse = nn.Linear(self.length, self.dictionary_size)
        self.optional_softmax = nn.Softmax(dim=3)


    def forward(self, X, logits=True):
        # embedded = self.embeding(X)
        # reconstruction = self.decoder(self.encoder(embedded))
        # return self.embeding_reverse(reconstruction)
        translation = self.decoder(self.encoder(X))
        if not logits:
            translation = self.optional_softmax(translation)
        return translation



class Discriminator(nn.Module):
    def __init__(self, length, dictionary_size):
        super().__init__()
        self.length = length
        self.dictionary_size = dictionary_size
        self.encoder = nn.Sequential(
            nn.Linear(self.dictionary_size, self.length),
            nn.ReLU(),
            # initial size  [1, 100, 100]
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=11, stride=1),
            # size [4, 90, 90]
            nn.InstanceNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=11, stride=1),
            # size [4, 80, 80]
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=11, stride=1),
            # size [12, 70, 70]
            nn.InstanceNorm2d(12),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=11, stride=1),
            # size [16, 60, 60]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size [20, 50, 50]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [24, 40, 40]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [ 28, 30, 30]
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [ 32, 20, 20]
            nn.InstanceNorm2d(16),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 *20 *20, 2048),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, X):
        encoded = self.encoder(X)
        endoded_flatten = encoded.view(-1, 1, encoded.size()[1] * encoded.size()[2] * encoded.size()[3])
        return self.mlp(endoded_flatten).view(-1, 1)


if __name__ == "__main__":
    data = convert_data(fixed_len=100)
    sample = torch.from_numpy(data[0])
    sample= sample.view(-1, 1, 100, 27).float()
    emb = Generator(100, 27)
    out = emb(sample, logits=False)
    print(out.size(), out.sum())

    #
    # print(reconstruct_loss.item())
    # fake_crypted_reconstruct_loss = cross_entropy(fake_crypted_reconstruct.view(-1, wandb.config["instance_size"], wandb.config["dictionary_size"]).transpose(1,2), torch.argmax(real_crypted_text, 3).view(-1,wandb.config["instance_size"]))
