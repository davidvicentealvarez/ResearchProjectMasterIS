import torch
import torch.optim as optim
import torch.nn as nn
from torch import nn
import numpy as np
from nltk.corpus import brown





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("brown_corpus.npy")
    sample = torch.tensor(data[0:2])
    sample = sample.float()
    print(sample.size())
    sample = sample.view(2,1,1000,27)
    m = nn.ConvTranspose2d(1, 10, (3,3))
    print(m(sample).size())
