import torch
from autoencoder.encoders import DenseEncoder, ConvEncoder
from autoencoder.decoders import DenseDecoder, ConvDecoder

# TODO cleanup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class AutoEncoder(torch.nn.Module):
    def __init__(
        self, 
        input_shape, 
        latent_size,
        reg_type,
        reg_rate,
        encoder_type,
        decoder_type
):
        super(AutoEncoder, self).__init__()

        self.reg_rate = reg_rate
        if reg_type == 'l1':
            reg_func = lambda x: reg_rate * x.abs().sum()
        elif reg_type == 'l2':
            reg_func = lambda x: reg_rate * torch.sqrt((x**2).sum()) # TODO switch to norm
        else:
            reg_func = lambda _: 0

        self.input_shape = input_shape
        self.latent_size = latent_size

        encoder_constructor = DenseEncoder if encoder_type == 'dense' else ConvEncoder
        decoder_constructor = DenseDecoder if decoder_type == 'dense' else ConvDecoder
        self.encoder = encoder_constructor(input_shape, 
                                           latent_size,
                                           activation_regularization_func=reg_func)
        self.decoder = decoder_constructor(latent_size, 
                                           input_shape,
                                           activation_regularization_func=reg_func)
        self.reset()
        self.raw_latent = None
        self.latent = None
    
    def forward(self, x):
        x = self.encoder(x)
        self.raw_latent = x
        self.latent = x.tolist()
        x = self.decoder(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def generate(self, x):
        return self.decoder(x)

    def criterion(self, y_hat, y):
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y, reduction='mean') + self.encoder.activations_total + self.decoder.activations_total
        #loss = torch.nn.functional.binary_cross_entropy(y_hat, y, reduction='sum') 
        #loss += self.reg_rate * torch.sqrt((self.raw_latent**2).sum())
        #loss = torch.nn.functional.mse_loss(y_hat, y, reduction='sum') 
        return loss
    
    def reset(self):
        self.decoder.activations_total = torch.tensor([0.0]).to(DEVICE)
        self.encoder.activations_total = torch.tensor([0.0]).to(DEVICE)
    
    def epoch_reset(self):
        pass

