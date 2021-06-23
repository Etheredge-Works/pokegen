import torch
from torch import nn
from torch.nn import functional as F


class Encoder(torch.nn.Module):
    def __init__(self, input_shape, latent_shape):
        super(Encoder, self).__init__()
        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        self.conv1 = nn.Conv2d(input_shape[0], 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2)
        #TODO global avg poolself.pool = nn.Av
        self.fc = nn.Linear(64*5*5, latent_shape)
        self.f = nn.Flatten()
        #self.dense = torch.nn.Linear()

    def forward(self, x):
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.f(x)
        x = F.sigmoid(self.fc(x))

        return x


class Decoder(torch.nn.Module):
    def __init__(self, latent_shape, output_shape):
        super(Decoder, self).__init__()
        self.output_flatten_shape = torch.prod(torch.tensor(output_shape), 0).item()
        self.output_shape = output_shape
        # TODO math latent_to_shape form input shape and layers
        
        self.latent_to_shape = (64, 5, 5)

        self.fc = nn.Linear(latent_shape, 64*5*5) # TODO math out better
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2)
        self.conv3 = nn.ConvTranspose2d(16, 8, 3, stride=2)
        self.conv4 = nn.ConvTranspose2d(8, 3, 3, stride=2, output_padding=1)
        #self.conv1 = nn.ConvTranspose2d(64, 32)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, *self.latent_to_shape)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        #x = torch.view(x, [-1, *self.output_shape])
        #x = x.view(-1, *self.output_shape)
        # TODO test speed view
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
