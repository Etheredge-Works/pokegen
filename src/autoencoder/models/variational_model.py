import torch

# Reference: https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
class VAE(torch.nn.Module):
    def __init__(
        self, 
        input_shape, 
        latent_size,
        encoder_constructor,
        decoder_constructor
):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.encoder = encoder_constructor(input_shape, latent_size*2)
        self.decoder = decoder_constructor(latent_size, input_shape)

        #self.bce = torch.nn.BCELoss(reduction='sum')
        self.bce = torch.nn.MSELoss()
    
    @staticmethod
    def reparameterize(mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):
        x = self.encoder(x)

        x = x.view(-1, 2, self.latent_size)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]

        z = self.reparameterize(mu, log_var)

        x_hat = self.decoder(z)
        # TODO sigmoid at end?

        return x_hat, mu, log_var

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_pred, *_ = self.forward(x)
        return y_pred

    def generate(self, x):
        x = x.view(-1, 2, self.latent_size)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]

        z = self.reparameterize(mu, log_var)
        return self.decoder(z)
        
    def criterion(self, y_hat, y):
        reconstruction, mu, log_var = y_hat
        #BCE = self.bce(reconstruction, y)  # TODO why did this fail previously?
        BCE = torch.nn.functional.binary_cross_entropy(reconstruction, y, reduction='sum')
        #print(BCE.item())
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        #print(KLD.item())
        #input()
        return BCE + KLD
