from torchvision import transforms
import pathlib
from pathlib import Path
from PIL import Image


def display(tensor):
    im = transforms.ToPILImage()(tensor)
    im.show()


def save(
    tensor, 
    path: Path, 
    step
):
    path = pathlib.Path(path)

    for idx, item in enumerate(tensor):
        im = transforms.ToPILImage()(item)

        item_path = path/str(idx)
        item_path.mkdir(exist_ok=True, parents=True)
        im.save(item_path/f"{step}.jpg")


def make_gif(path):
    path = pathlib.Path(path)
    ims = [Image.open(file).resize((128, 128)) for file in path.iterdir()]
    ims[0].save(
        f"{path}.gif", 
        save_all=True,
        append_images=ims[1:], 
        duration=300,
        loops=0)


def make_gifs(path):
    path = pathlib.Path(path)
    for dir in path.iterdir():
        print(dir)
        make_gif(dir)