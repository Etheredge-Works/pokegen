from torchvision import transforms
import pathlib
from pathlib import Path
from PIL import Image

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


def save(
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