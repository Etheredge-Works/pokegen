import torch
from torch import nn
import torch.nn.functional as F


class DenseDecoder(nn.Module):
    def __init__(self, latent_shape, output_shape):
        super(DenseDecoder, self).__init__()
        self.output_shape = output_shape
        self.flattened_size = torch.prod(torch.tensor(output_shape), 0)

        self.dense1 = nn.Linear(latent_shape, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 256)
        self.dense4 = nn.Linear(256, 512)
        self.fc = nn.Linear(512, self.flattened_size)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = torch.sigmoid(self.fc(x))
        x = x.view(-1, *self.output_shape)
        return x

