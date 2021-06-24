import torch
from torch import nn
from torch.nn import functional as F


class DenseEncoder(torch.nn.Module):
    def __init__(self, input_shape, latent_shape):
        super(DenseEncoder, self).__init__()
        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        self.f = nn.Flatten()
        self.dense1 = nn.Linear(flattened_size, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.fc = nn.Linear(64, latent_shape)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = self.f(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = F.sigmoid(self.fc(x))
        return x

