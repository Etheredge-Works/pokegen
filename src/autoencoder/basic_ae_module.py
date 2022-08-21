from gc import callbacks
import urllib.parse
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F

from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)


class AE(LightningModule):
    """Standard AE.

    Model is available pretrained on different datasets:

    Example::

        # not pretrained
        ae = AE()

        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        "cifar10-resnet18": urllib.parse.urljoin(_HTTPS_AWS_HUB, "ae/ae-cifar10/checkpoints/epoch%3D96.ckpt"),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-3,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + " not present in pretrained weights.")

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, z):
        # Making forward decode so I can use pl_bolts.callbacks.[LatentDimInterpolator,Tensorbo...]
        x_hat = self.decode(z)
        return torch.tanh(x_hat)
    
    def decode(self, z):
        return self.decoder(z)

    def encode_decode(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        x_hat = torch.tanh(x_hat)

        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        x_hat = torch.tanh(x_hat)

        # loss = F.l1_loss(x_hat, x, reduction="mean")
        loss = F.mse_loss(x_hat, x, reduction="mean")
        # loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction="mean")
        # loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction="none")
        # loss = F.mse_loss(x_hat, x, reduction="none")
        # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        # loss = F.mse_loss(x_hat, x, reduction="mean")

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        # NOTE training with cross entropy loss and not sigmoiding on predictions makes funky colors
        # return torch.sigmoid(self(x))
        x_hat = self.encode_decode(x)
        x_hat = torch.tanh(x_hat)
        return x_hat

    def configure_optimizers(self):
        optimizer= torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.2, 
                                                         patience=20, 
                                                         min_lr=5e-5,
                                                         verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default="resnet18", help="resnet18/resnet50")
        parser.add_argument("--first_conv", action="store_true")
        parser.add_argument("--maxpool1", action="store_true")
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets",
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser

from data import sprites
def cli_main(args=None):
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "stl10", "imagenet", "pokemon"])
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "stl10":
        dm_cls = STL10DataModule
    elif script_args.dataset == "imagenet":
        dm_cls = ImagenetDataModule
    elif script_args.dataset == "pokemon":
        # dm_cls, val_loader = sprites.get_loader(
            # batch_size=128,
            # path='data/external/sprites',
            # val_ratio=0
        # )
        dm_cls = sprites.PokeModule
    else:
        raise ValueError(f"undefined dataset {script_args.dataset}")

    parser = AE.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    args.input_height = dm.size()[-1]

    if args.max_steps == -1:
        args.max_steps = None

    model = AE(**vars(args))

    from utils import LatentDimInterpolator
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(
        model, 
        datamodule=dm,
            )
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
