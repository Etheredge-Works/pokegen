import torch
from torch import nn
import torch.nn.functional as F
# from utils import nextPowerOf2

# https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/
def nextPowerOf2(n):
    count = 0
 
    # First n in the below
    # condition is for the
    # case where n is 0
    if (n and not(n & (n - 1))):
        return n
     
    while( n != 0):
        n >>= 1
        count += 1
     
    return 1 << count

class DenseDecoder(nn.Module):
    def __init__(
        self, 
        latent_shape, 
        output_shape, 
        activation_regularization_func=lambda _: 0,
        node_count=1024,
    ):
        super(DenseDecoder, self).__init__()

        self.act_reg_func = activation_regularization_func

        self.output_shape = output_shape
        self.flattened_size = torch.prod(torch.tensor(output_shape), 0)

        input_count = latent_shape
        layers = []
        next_count = nextPowerOf2(input_count)
        layers.append(nn.Linear(input_count, next_count))
        input_count = next_count
        print("input_count:", input_count)

        while(input_count < node_count):
            print("input_count:", input_count)
            layers.append(nn.Linear(input_count, input_count*2))
            input_count *= 2

        self.dense = nn.ModuleList(layers) # registers modules

        self.fc = nn.Linear(node_count, self.flattened_size)

        self.activations_total = None

    def forward(self, x):
        for layer in self.dense:
            x = layer(x)
            x = F.leaky_relu(x)
            #F.relu_(x)
            self.activations_total += self.act_reg_func(x)

        x = self.fc(x)
        x = torch.sigmoid(x)
        x = x.view(-1, *self.output_shape)
        return x
