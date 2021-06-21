import torch
from torch import nn
from torch.nn import functional as F


class Encoder(torch.nn.Module):
    def __init__(self, input_shape, latent_shape):
        super(Encoder, self).__init__()
        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        self.dense1 = torch.nn.Linear(flattened_size, 256)
        self.dense2 = torch.nn.Linear(256, 128)
        self.dense3 = torch.nn.Linear(128, 64)
        self.dense4 = torch.nn.Linear(64, 32)
        self.dense5 = torch.nn.Linear(32, latent_shape)
        self.f = nn.Flatten()
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = self.f(x)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = F.sigmoid(self.dense5(x))

        return x


class Decoder(torch.nn.Module):
    def __init__(self, latent_shape, output_shape):
        super(Decoder, self).__init__()
        self.output_flatten_shape = torch.prod(torch.tensor(output_shape), 0).item()
        self.output_shape = output_shape
        self.dense = torch.nn.Linear(latent_shape, self.output_flatten_shape)
        
        self.dense1 = nn.Linear(latent_shape, 32)
        self.dense2 = nn.Linear(32, 64)
        self.dense3 = nn.Linear(64, 128)
        self.dense4 = nn.Linear(128, 256)
        self.dense5 = nn.Linear(256, self.output_flatten_shape)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = self.dense5(x)

        x = torch.reshape(x, [-1, *self.output_shape])
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_shape, latent_shape):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_shape, latent_shape)
        self.decoder = Decoder(latent_shape, input_shape)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
