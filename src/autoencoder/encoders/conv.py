import torch
from torch import nn
from torch.nn import functional as F


class ConvEncoder(torch.nn.Module):
    def __init__(
        self, 
        input_shape, 
        latent_shape):
        super(ConvEncoder, self).__init__()
        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        self.conv1 = nn.Conv2d(input_shape[0], 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2)
        self.f = nn.Flatten()
        # Pooling here
        self.fc = nn.Linear(64, latent_shape)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # TODO was pooling used just to make it easier to construct model? no mathing dims
        # Pooling
        x = x.mean([2, 3])

        x = self.f(x)
        x = F.relu(self.fc(x))
        #x = F.sigmoid(self.fc(x))

        return x