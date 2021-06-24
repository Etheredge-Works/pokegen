import torch


class AutoEncoder(torch.nn.Module):
    def __init__(
        self, 
        input_shape, 
        latent_size,
        encoder_constructor,
        decoder_constructor
):
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.encoder = encoder_constructor(input_shape, latent_size)
        self.decoder = decoder_constructor(latent_size, input_shape)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        return self.decoder(x)