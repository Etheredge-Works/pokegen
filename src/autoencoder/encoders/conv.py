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
        # TODO enlarge kernel
        self.conv1 = nn.Conv2d(input_shape[0], 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 5, stride=2)
        self.flatten = nn.Flatten()
        # Pooling here
        self.fc = nn.Linear(128, latent_shape)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # TODO was pooling used just to make it easier to construct model? no mathing dims
        # Pooling
        #print(x.size())
        #input()
        x = x.mean([2, 3])
        #print(x.size())
        #input()

        x = self.flatten(x)
        x = self.fc(x)

        return x