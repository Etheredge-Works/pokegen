from torchvision import transforms
import pathlib
from pathlib import Path
from PIL import Image
import torch
import numpy as np

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
    step=None
):
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)
    #df = pd.DataFrame(latents, columns=['x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
    #print(latents)
    #print(len(latents[0]))
    #print(latent_size)
    if len(latents[0]) == 2*latent_size:
        df = pd.DataFrame(latents, columns=[str(idx) for idx in range(latent_size*2)])

        #fig = px.scatter_3d(df, x="x1", y="y1") #, z="z1")
        fig = px.scatter(df, x="0", y="1") #, z="z1")
        fig.write_image(path/'mu.png')
        #fig2 = px.scatter_3d(df, x="x2", y="y2") #, z="z1")
        fig2 = px.scatter(df, x=str(latent_size), y=str(latent_size+1))
        fig2.write_image(path/'log_var.png')

        N1 = torch.randn(len(latents)).numpy()
        N2 = torch.randn(len(latents)).numpy()
        #df['N'] = N
        df['z1'] = (N1 * np.exp(df[str(latent_size)]*0.5)) + df['0']
        df['z2'] = (N2 * np.exp(df[str(latent_size+1)]*0.5)) + df['1']
        figz1 = px.scatter(df, x="z1", y="z2") #, z="z1")
        figz1.write_image(path/'z.png')
    else:
        df = pd.DataFrame(latents, columns=[str(idx) for idx in range(latent_size)])

        #fig = px.scatter_3d(df, x="x1", y="y1") #, z="z1")
        fig = px.scatter(df, x="0", y="1") #, z="z1")
        fig.write_image(path/'latent.png')


def save_image(
    tensor, 
    path: Path, 
    step=None
):
    path = pathlib.Path(path)
    path.mkdir(exist_ok=True, parents=True)

    for idx, item in enumerate(tensor):
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