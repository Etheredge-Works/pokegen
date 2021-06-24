import torch
from torch import nn
from torch.nn import functional as F


class ConvDecoder(nn.Module):
    def __init__(self, latent_shape, output_shape):
        super(ConvDecoder, self).__init__()
        self.output_flatten_shape = torch.prod(torch.tensor(output_shape), 0).item()
        self.output_shape = output_shape
        # TODO math latent_to_shape form input shape and layers
        
        self.latent_to_shape = (64, 5, 5)

        self.fc = nn.Linear(latent_shape, 64) # TODO math out better
        self.up = nn.Upsample(scale_factor=5)
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2)
        self.conv3 = nn.ConvTranspose2d(16, 8, 3, stride=2)
        self.conv4 = nn.ConvTranspose2d(8, 3, 3, stride=2, output_padding=1)
        #self.conv1 = nn.ConvTranspose2d(64, 32)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), x.size(1), 1, 1) 
        x = self.up(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.sigmoid(self.conv4(x))

        #x = torch.view(x, [-1, *self.output_shape])
        #x = x.view(-1, *self.output_shape)
        # TODO test speed view
        return x
