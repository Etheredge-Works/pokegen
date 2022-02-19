from multiprocessing.sharedctypes import Value
import torch
from autoencoder.encoders import DenseEncoder, ConvEncoder
from autoencoder.decoders import DenseDecoder, ConvDecoder
import dvclive

# TODO cleanup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reference: https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
class VAE(torch.nn.Module):
    def __init__(
        self, 
        input_shape,
        latent_size,
        reg_type,
        reg_rate,
        encoder_type,
        decoder_type,
        beta=0.0,
        beta_rate=0.01,
        beta_max=1.
    ):
        super(VAE, self).__init__()

        # TODO pull out
        if reg_type == 'l1':
            reg_func = lambda x: reg_rate * x.abs().sum()
        elif reg_type == 'l2':
            reg_func = lambda x: reg_rate * torch.sqrt((x**2).sum())
        else:
            reg_func = lambda _: 0

        self.input_shape = input_shape
        self.latent_size = latent_size

        encoder_constructor = DenseEncoder if encoder_type == 'dense' else ConvEncoder
        decoder_constructor = DenseDecoder if decoder_type == 'dense' else ConvDecoder
        self.encoder = encoder_constructor(input_shape, 
                                           latent_size*2, 
                                           activation_regularization_func=reg_func)
        self.decoder = decoder_constructor(latent_size, 
                                           input_shape,
                                           activation_regularization_func=reg_func)

        #self.bce = torch.nn.BCELoss(reduction='mean')
        self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))
        self.beta = beta
        self.beta_rate = beta_rate
        self.beta_max = beta_max
        self.latent = None
        self.reset()
    
    @staticmethod
    def reparameterize(mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        #std = torch.exp(log_var * 0.5)
        std = torch.exp(0.5*log_var)
        #q = torch.distributions.Normal(mu, std)
        #z = q.rsample()
        eps = torch.randn_like(std)
        z = mu + (eps*std)

        return z, std
    
    # https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        
        # sum over last dim to go from single dim distribution to multi-dim
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        x = self.encoder(x)
        # NOTE must use self.activation_total to keep using pretty summaries 
        #      Since two returns on forward breaks things
        #encoder_activations = self.encoder.activation_total

        mu = x[:, :self.latent_size]
        log_var = x[:, self.latent_size:]
        self.latent = x.tolist()

        z, std = self.reparameterize(mu, log_var)

        x_hat = self.decoder(z)
        #decoder_activations = self.decoder.activation_total
        # TODO sigmoid at end?

        return x_hat, z, mu, log_var, std

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            y_pred, *_ = self.forward(x)
        return y_pred

    def generate(self, x):
        #x = x.view(-1, 2, self.latent_size)
        #mu = x[:, 0, :]
        #log_var = x[:, 1, :]

        #z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(x)
        return x_hat
        
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def criterion(self, y_hat, y):
        x_hat, z, mu, log_var, std = y_hat

        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, y)
        #recon_loss = self.gl(x_hat)
        #recon_loss = self.bce(x_hat, y)

        kl = self.kl_divergence(z, mu, std)
        #kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        #recon_loss = torch.nn.binary_cross_entropy(x_hat, y, size_average=False) / y.size(0)


        # Works!
        # recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, y, reduction='mean')
        # kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1. - log_var))
        # loss = recon_loss + (self.beta * kl_loss)


        #return recon_loss
        #loss = recon_loss + kl
        #loss = recon_loss + (self.beta * kl)
        elbo = ((self.beta*kl) - recon_loss)
        # elbo = ((self.beta*kl) - recon_loss)
        # elbo = kl - recon_loss
        # return elbo.mean()
        loss =  elbo.mean() + self.encoder.activations_total + self.decoder.activations_total

        return loss

    def reset(self):
        self.decoder.activations_total = torch.tensor([0.]).to(DEVICE)
        self.encoder.activations_total = torch.tensor([0.]).to(DEVICE)

    def epoch_reset(self, beta=None):
        # https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder
        #https://arxiv.org/pdf/1511.06349.pdf
        if beta is not None:
            self.beta = beta
        else:
            self.beta = min(self.beta+self.beta_rate, self.beta_max)
        # TODO use mlflow and log both losses
        dvclive.log('beta', self.beta)
