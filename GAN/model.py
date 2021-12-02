from torch import nn
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
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=11, stride=1),
            # size [4, 80, 80]
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=11, stride=1),
            # size [12, 70, 70]
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=11, stride=1),
            # size [16, 60, 60]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size [20, 50, 50]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [24, 40, 40]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [ 28, 30, 30]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [ 32, 20, 20]
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            # size = [32, 20, 20]
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [28, 30, 30]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size [24, 40, 40]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size [20, 50, 50]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            #  size [16, 60, 60]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=12, kernel_size=11, stride=1),
            # size [4, 70, 70
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=8, kernel_size=11, stride=1),
            # size [4, 80, 80]
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=11, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            # size [ 1, 90, 90]
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=11, stride=1),
            nn.BatchNorm2d(1),
            # size [ 1, 100, 100]
            nn.ReLU(),
            nn.Linear(self.length, self.dictionary_size),
            nn.ReLU()


        )
        # self.embeding = nn.Linear(self.dictionary_size, self.length)
        # self.embeding_reverse = nn.Linear(self.length, self.dictionary_size)

    def forward(self, X):
        # embedded = self.embeding(X)
        # reconstruction = self.decoder(self.encoder(embedded))
        # return self.embeding_reverse(reconstruction)
        return  self.decoder(self.encoder(X))



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
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=11, stride=1),
            # size [4, 80, 80]
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=11, stride=1),
            # size [12, 70, 70]
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=11, stride=1),
            # size [16, 60, 60]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size [20, 50, 50]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [24, 40, 40]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [ 28, 30, 30]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=11, stride=1),
            # size = [ 32, 20, 20]
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 *20 *20, 2048),
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
            nn.Linear(16, 1),
        )

    def forward(self, X):
        encoded = self.encoder(X)
        endoded_flatten = encoded.view(-1, 1, encoded.size()[1] * encoded.size()[2] * encoded.size()[3])
        return self.mlp(endoded_flatten).view(-1, 1)


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

    def forward(self, X):
        # tensor_1d = self.reduction_layer(X).view(-1, 4, self.length)
        # new_rep = self.convultion_layer(tensor_1d).view(-1, 4, self.length, 1)
        # return self.deconv_layer(new_rep)

        tensor_1d = self.reduction_layer(X).view(-1, 4, self.length)
        new_rep = self.convultion_layer(tensor_1d).view(-1, 1, self.length *32)
        reconstruction = self.linear_layer(new_rep).view(-1, 1, self.length, self.dict_size)
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



if __name__ == "__main__":
    data = convert_data(fixed_len=100)
    sample = torch.tensor(data[3])
    sample= sample.view(-1, 1, 100, 27).float()



    # print(loss.item())
    # fake_crypted_reconstruct_loss = cross_entropy(fake_crypted_reconstruct.view(-1, wandb.config["instance_size"], wandb.config["dictionary_size"]).transpose(1,2), torch.argmax(real_crypted_text, 3).view(-1,wandb.config["instance_size"]))
