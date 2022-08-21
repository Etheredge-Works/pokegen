
from cProfile import label
from data import sprites

from pl_bolts.models.gans import GAN, DCGAN
from autoencoder.encoders import ConvEncoder
from autoencoder.decoders import ConvDecoder
from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule
import torchmetrics
from utils import LatentDimInterpolator, WandbGenerativeModelImageSampler

import torch 
from torch import Tensor

class CustomDCGAN(DCGAN):
    def __init__(
        self, 
        soften_lower=0.3, 
        soften_upper=0.7, 
        swap_prob=0.01, 
        noise_std=0.01,
        labels_size = 41,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.real_accuracy = torchmetrics.Accuracy()
        self.fake_accuracy = torchmetrics.Accuracy()
        self.swap_prob = swap_prob
        self.soften_lower = soften_lower
        self.soften_upper = soften_upper
        self.noise_std = noise_std
        self.labels_size = labels_size
        self.last_labels = None # for convenience
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, label = batch
        # print(label.shape)

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(real, label)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real, label)

        return result

    def forward(self, noise: Tensor) -> Tensor:
        """Generates an image given input noise.

        Example::

            noise = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    def _get_noise(self, n_samples: int, latent_dim: int) -> Tensor:
        return torch.randn(n_samples, latent_dim-self.labels_size, device=self.device)

    def _get_fake_pred(self, real: Tensor, labels) -> Tensor:
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        # print(f"noise shape: {noise.shape}")
        # print(f"labels shape: {labels.shape}")
        noise = torch.concat([noise, labels], dim=1)
        # self.last_labels = labels
        fake = self(noise)
        fake = fake + torch.randn_like(fake) * self.noise_std
        fake_pred = self.discriminator(fake)

        return fake_pred
    
    def _gen_step(self, real: Tensor, labels) -> Tensor:
        gen_loss = self._get_gen_loss(real, labels)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss
    
    def _disc_step(self, real: Tensor, labels) -> Tensor:
        disc_loss = self._get_disc_loss(real, labels)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _get_gen_loss(self, real: Tensor, labels) -> Tensor:
        # Train with fake
        fake_pred = self._get_fake_pred(real, labels)
        fake_gt = torch.ones_like(fake_pred)
        gen_loss = self.criterion(fake_pred, fake_gt)
        return gen_loss

    def _get_disc_loss(self, real: Tensor, labels) -> Tensor:
        # Train with real
        # print(f"real shape: {real.shape}")
        real = real + torch.randn_like(real) * self.noise_std
        real_pred = self.discriminator(real)
        # print(f"real_pred shape: {real_pred.shape}")
        # real_gt = torch.ones_like(real_pred)

        # real_acc = self.real_accuracy(real_pred.cpu(), torch.ones_like(real_pred).type(torch.long).cpu())
        real_acc = self.real_accuracy(real_pred, torch.ones_like(real_pred, device=real_pred.device).type(torch.long))
        self.log("acc/real", real_acc, on_epoch=True, on_step=False)

        # Train with fake
        fake_pred = self._get_fake_pred(real, labels)
        # fake_gt = torch.zeros_like(fake_pred)
        fake_acc = self.fake_accuracy(fake_pred, torch.zeros_like(fake_pred, device=fake_pred.device).type(torch.long))
        # fake_acc = self.fake_accuracy(fake_pred.cpu(), torch.zeros_like(fake_pred).type(torch.long).cpu())
        self.log("acc/fake", fake_acc, on_epoch=True, on_step=False)

        # Soften labels
        real_gt = (self.soften_upper - 1.0) * torch.rand_like(real_pred) + 1.0
        fake_gt = (0.0 - self.soften_lower) * torch.rand_like(fake_pred) + self.soften_lower

        # Randomly swap labels
        swap_chances = torch.rand_like(real_pred)
        # print(swap_chances.shape)
        # real_gt = torch.where(swap_chances < self.swap_prob, fake_gt, real_gt)
        # fake_gt = torch.where(swap_chances < self.swap_prob, real_gt, fake_gt)

        # Loss
        real_loss = self.criterion(real_pred, real_gt)
        fake_loss = self.criterion(fake_pred, fake_gt)

        self.log("loss/d_real", real_loss, on_epoch=True, on_step=False)
        self.log("loss/d_fake", fake_loss, on_epoch=True, on_step=False)
        disc_loss = real_loss + fake_loss

        dis_acc = (real_acc + fake_acc) / 2 # MAYBE BAD?
        self.log("acc/total", dis_acc, on_epoch=True, on_step=False)

        return disc_loss

class BenGAN(DCGAN):
    def init_generator(self, img_dim):
        generator = ConvDecoder(
            latent_shape=self.hparams.latent_dim, 
            output_shape=img_dim,
            final_activation=torch.tanh,)
        return generator
    
    def init_discriminator(self, img_dim):
        discriminator = ConvEncoder(
            input_shape=img_dim,
            latent_shape=1,
            final_activation=torch.sigmoid,
            dropout_rate=0.1)
        return discriminator


from pl_bolts.models.gans import GAN, DCGAN, SRResNet, SRGAN
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
# from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
from pytorch_lightning.callbacks import StochasticWeightAveraging
import torchvision
from pytorch_lightning import loggers as pl_loggers
from torchvision.datasets import CelebA, LFWPairs, MNIST

if __name__ == "__main__":
    label_dim = 41
    latent_dim = 32 + label_dim
    ROWS = 16
    BATCH_SIZE = 256

    # logger = WandbLogger(project="gan_test")
    # gan = BenGAN(3, 96, 96, latent_dim=latent_dim)
    # gan = DCGAN(image_channels=3, feature_maps_disc=16, feature_maps_gen=16, latent_dim=latent_dim)
    # gan = CustomDCGAN(i6age_channels=3, feature_maps_disc=32, feature_maps_gen=128, latent_dim=latent_dim)
    gan = CustomDCGAN(
        image_channels=3, 
        latent_dim=latent_dim, 
        # beta=0.9,
        # feature_maps_disc=32, 
        # feature_maps_gen=64
        feature_maps_disc=16, 
        feature_maps_gen=64,
        swap_prob=0.05,
        soften_lower=0.0,
        soften_upper=0.9,
        noise_std=0.0,
        )
    # gan = DCGAN(image_channels=3)
    # gan = SRResNet()
    # gan = SRGAN(scale_factor=2)
    # gan = SRResNet(image_channels=3, feature_maps=16, latent_dim=latent_dim)
    print(gan)
    # gan = GAN(3, 96, 96, latent_dim=latent_dim)
    #gan = GAN(1, 28, 28, latent_dim=latent_dim)

    # data = CelebA(
    #     root="./data", 
    #     # root="/data/torch/", 
    #     download=False, 
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    # data = MNIST(
    #     root="./data",
    #     download=True,
    #     transform=torchvision.transforms.Compose([
    #         torchvision.transforms.Resize((64, 64)),
    #         torchvision.transforms.ToTensor(),
    #         # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         torchvision.transforms.Normalize((0.5,), (0.5,)),
    #     ]))

    data = sprites.PokemonDataset("data/sprites")
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{data.__class__.__name__}_gan_logs/")
    wandb_logger = pl_loggers.WandbLogger(
        # project=f"{data.__class__.__name__}_gan",
        # name=None
    )
    # wandb.init(project=f"{data.__class__.__name__}_gan", sync_tensorboard=True)

    trainer = Trainer(
        accelerator="auto",
        devices=-1,
        # precision=16,
        logger=wandb_logger,
        # logger=tb_logger,
        # logger=[tb_logger, wandb_logger],
        callbacks=[
            WandbGenerativeModelImageSampler(
                interpolate_epoch_interval=10, 
                num_samples=BATCH_SIZE, 
                nrow=ROWS, 
                normalize=True, ),
            # TensorboardGenerativeModelImageSampler(num_samples=BATCH_SIZE, nrow=ROWS, normalize=True, ),
            # TensorboardGenerativeModelImageSampler(num_samples=BATCH_SIZE, nrow=ROWS, normalize=True, ),
            LatentDimInterpolator(interpolate_epoch_interval=10, normalize=True),
            # StochasticWeightAveraging(),
        ],
        # strategy="dp",
        max_epochs=2000
        # track_grad_norm=2,
        )

    # trainloadersssss = CIFAR10DataModule(
    #     "/tmp/", 
    #     num_workers=16,
    #     drop_last=True,
    #     train_transforms=torchvision.transforms.Compose([
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.Resize((64, 64)),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]),
    #     pin_memory=True,
    #     batch_size=256)
    #trainer.fit(gan, train_dataloaders=MNISTDataModule(num_workers=0, batch_size=64))

    # from pl_bolts.datamodules import 
    # trainloader = torch.utils.data.DataLoader(
    #     data, 
    #     batch_size=BATCH_SIZE, 
    #     pin_memory=True,
    #     persistent_workers=True,
    #     shuffle=True, 
    #     num_workers=24)


    trainloader, valloader = sprites.get_loader(
            batch_size=BATCH_SIZE,
            workers=16,
            val_ratio=0.0, 
            )

    trainer.fit(gan, train_dataloaders=trainloader)