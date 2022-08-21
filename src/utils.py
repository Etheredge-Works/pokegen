from torchvision import transforms
import pathlib
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from sklearn.decomposition import PCA
from pytorch_lightning import Trainer
from torch import Tensor
import torchvision


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


def display(tensor):
    im = transforms.ToPILImage()(tensor)
    im.show()


import plotly.express as px
import pandas as pd
def save_latents(
    latents, 
    latent_size,
    path: Path, 
    step=None,
):
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)
    #df = pd.DataFrame(latents, columns=['x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
    #print(latents)
    #print(len(latents[0]))
    #print(latent_size)
    # print(len(latents))
    latents = np.stack(latents)
    # print(latents[0].shape)

    # print(latents.shape)
    if latents.shape[-1] == 2*latent_size:
        mu = latents[:, :latent_size]
        pca_mu = PCA(n_components=2).fit(mu)
        print(f"pca_mu: {pca_mu.explained_variance_ratio_}")
        pca_mu_results = pca_mu.transform(mu)
        log_var = latents[:, latent_size:]
        std = np.exp(0.5 * log_var)
        pca_log_var = PCA(n_components=2).fit(std)
        print(f"pca_log_var: {pca_log_var.explained_variance_ratio_}")
        pca_log_var_results = pca_log_var.transform(std)
        # df = pd.DataFrame(latents, columns=[str(idx) for idx in range(latent_size*2)])
        df = pd.DataFrame(pca_mu_results, columns=[str(idx) for idx in range(pca_mu_results.shape[1])])

        #fig = px.scatter_3d(df, x="x1", y="y1") #, z="z1")
        fig = px.scatter(df, title="Mu PCA", x="0", y="1") #, z="z1")
        fig.write_image(path/'mu.png')
        # with open(path/"mu_explained_variance", "w") as f:
        #     pca_mu
        #fig2 = px.scatter_3d(df, x="x2", y="y2") #, z="z1")

        df = pd.DataFrame(pca_log_var_results, columns=[str(idx) for idx in range(pca_log_var_results.shape[1])])
        fig2 = px.scatter(df, title="STD PCA", x="0", y="1") #, z="z1")

        fig.write_image(path/'mu.png')

        fig2.write_image(path/'log_var.png')

        # N1 = torch.randn(len(latents)).numpy()
        # N2 = torch.randn(len(latents)).numpy()
        
        z = np.random.normal(mu, std, size=mu.shape)
        z_df = pd.DataFrame(z, columns=[str(idx) for idx in range(latent_size)])
        # for idx in range(latent_size):
            # z_df[str(idx)] = z[:, idx]
        z_df = pd.DataFrame(z, columns=[str(idx) for idx in range(z.shape[1])])
        print(z.shape)
        pca_z = PCA(n_components=2).fit(z)
        # pca_z_results = pca_z.transform(z)
        print(f"pca_z: {pca_z.explained_variance_ratio_}")
        
        #df['N'] = N
        # df['z1'] = (N1 * np.exp(df[str(latent_size)]*0.5)) + df['0']
        # df['z2'] = (N2 * np.exp(df[str(latent_size+1)]*0.5)) + df['1']
        df['z1'] = z[:, 0]
        df['z2'] = z[:, 1]

        figz1 = px.scatter(df, title="Z (2 dims) raw", x="z1", y="z2") #, z="z1")
        figz1.write_image(path/'z.png')
    else:
        df = pd.DataFrame(latents, columns=[str(idx) for idx in range(latent_size)])

        #fig = px.scatter_3d(df, x="x1", y="y1") #, z="z1")
        fig = px.scatter(df, x="0", y="1") #, z="z1")
        fig.write_image(path/'latent.png')


def save_image(
    tensor, 
    path: Path, 
    step=None,
    convert_func=lambda x: x
):
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)

    for idx, item in enumerate(tensor):
        item = convert_func(item)
        im = transforms.ToPILImage()(item)

        if step is not None:
            item_path = path/str(idx)
            item_path.mkdir(exist_ok=True, parents=True)
            if type(step) == int:
                step = f"{step:08d}"
            im.save(item_path/f"{step}.jpg")
        else:
            im.save(path/f"{idx}.jpg")


def make_gif(path):
    path = pathlib.Path(path)
    ims = [Image.open(file).resize((128, 128)) for file in list(sorted(list(path.iterdir())))]
    ims[0].save(
        f"{path}.gif", 
        save_all=True,
        append_images=ims[1:], 
        duration=300,
        loops=0)


def make_gifs(path):
    path = pathlib.Path(path)
    for dir in path.iterdir():
        make_gif(dir)


def reverse_norm(tensor, value_range = None):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if value_range is not None and not isinstance(value_range, tuple):
        raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    norm_range(tensor, value_range)
    return tensor


from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from typing import Optional, Tuple, List
class WandbGenerativeModelImageSampler(Callback):
    """Generates images and logs to tensorboard. Your model must implement the ``forward`` function for generation.

    Requirements::

        # model must have img_dim arg
        model.img_dim = (1, 28, 28)

        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)

    Example::

        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        interpolate_epoch_interval: int = 20,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.interpolate_epoch_interval = interpolate_epoch_interval

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            dim = (self.num_samples, pl_module.hparams.latent_dim)
            z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)
            try:
                z[:, pl_module.hparams.labels_size:] = 0
            except AttributeError:
                pass
            # z[:, pl_module.hparams.labels_size:] = torch.randint(0, 2, pl_module.hparams.labels_size, device=pl_module.device)

            # generate images
            with torch.no_grad():
                pl_module.eval()
                images = pl_module(z)
                pl_module.train()

            if len(images.size()) == 2:
                img_dim = pl_module.img_dim
                images = images.view(self.num_samples, *img_dim)

            grid = torchvision.utils.make_grid(
                tensor=images,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )
            str_title = f"{pl_module.__class__.__name__}_images"
            trainer.logger.log_image(key=str_title, images=[grid], step=trainer.global_step)

class LatentDimInterpolator(Callback):
    """Interpolates the latent space for a model by setting all dims to zero and stepping through the first two
    dims increasing one unit at a time.

    Default interpolates between [-5, 5] (-5, -4, -3, ..., 3, 4, 5)

    Example::

        from pl_bolts.callbacks import LatentDimInterpolator

        Trainer(callbacks=[LatentDimInterpolator()])
    """

    def __init__(
        self,
        interpolate_epoch_interval: int = 20,
        range_start: int = -5,
        range_end: int = 5,
        steps: int = 11,
        num_samples: int = 2,
        normalize: bool = True,
    ):
        """
        Args:
            interpolate_epoch_interval: default 20
            range_start: default -5
            range_end: default 5
            steps: number of step between start and end
            num_samples: default 2
            normalize: default True (change image to (0, 1) range)
        """

        super().__init__()
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.normalize = normalize
        self.steps = steps

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(pl_module, latent_dim=pl_module.hparams.latent_dim)
            images = torch.cat(images, dim=0)

            num_rows = self.steps
            grid = torchvision.utils.make_grid(images, nrow=num_rows, normalize=self.normalize)
            str_title = f"{pl_module.__class__.__name__}_latent_space"
            trainer.logger.log_image(key=str_title, images=[grid], step=trainer.global_step)

    def interpolate_latent_space(self, pl_module: LightningModule, latent_dim: int) -> List[Tensor]:
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                for z2 in np.linspace(self.range_start, self.range_end, self.steps):
                    # set all dims to zero
                    z = torch.zeros(self.num_samples, latent_dim, device=pl_module.device)

                    # set the fist 2 dims to the value
                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    # sample
                    # generate images
                    img = pl_module(z)

                    if len(img.size()) == 2:
                        img = img.view(self.num_samples, *pl_module.img_dim)

                    img = img[0]
                    img = img.unsqueeze(0)
                    images.append(img)

        pl_module.train()
        return images

import torch
from pytorch_lightning.callbacks import BasePredictionWriter, StochasticWeightAveraging, Callback, EarlyStopping
import os
from typing import Any, List
import utils
import torchvision

class EncodeDecodeWriter(Callback):

    def __init__(
        self,
        data: Any,
        write_interval: int = 20,
        num_rows: int = 1,
        normalize: bool = False,
        post_fix: str = "",
    ):
        super().__init__()
        self.write_interval = write_interval
        self.data = data
        self.num_rows = num_rows
        self.normalize = normalize
        self.post_fix = post_fix

    def on_epoch_end(self, trainer: "pl.Trainer", ae_module: "pl.LightningModule") -> None:

        # print("writing batch")
        # torch.save(prediction, os.path.join(
        #     self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

        if (trainer.current_epoch + 1) % self.write_interval == 0:
            ae_module.eval()
            with torch.no_grad():
                results = ae_module.encode_decode(self.data.to(ae_module.device))


            grid = torchvision.utils.make_grid(results, nrow=self.num_rows, normalize=self.normalize)
            str_title = f"{ae_module.__class__.__name__}_encode_decode{self.post_fix}"
            # trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)
            trainer.logger.log_image(key=str_title, images=[grid], step=trainer.global_step)
        # utils.save_image(
        #     results, 
        #     os.path.join(self.output_dir, "preds"), 
        #     # convert_func=sprites.denormalize
        #     )
    # def write_on_epoch_end(
    #     self, trainer, pl_module: 'LightningModule', predictions: List[Any], batch_indices: List[Any]
    # ):