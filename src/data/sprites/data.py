from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
from pathlib import Path

# This is for a similar issue: https://github.com/pytorch/pytorch/issues/57273
import warnings
warnings.filterwarnings("ignore")
# NOTE doesn't work...


class PokemonDataset(Dataset):
    
    normal_sprites_sub_dir = Path("pokemon")
    female_sub_dir = Path("pokemon/female")
    shiny_sprites_sub_dir = Path("pokemon/shiny")
    female_shiny_sprites_sub_dir = Path("pokemon/shiny/female")

    # TODO do I want to use backs?
    # TODO filter question mark
    backs_sub_dir = "pokemon/shiny/female"

    def __init__(self, sprites_path, transform=None):
        self.sprites_path = Path(sprites_path)
        self.transform = transform
        self.files = list(self.sprites_path.glob(str(self.normal_sprites_sub_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(self.female_sub_dir/'*.png')))

        # TODO create shiny flag to make generation shiny.
        # TODO shiny generater?
        # TODO cooler shiny geneator trained with only "good" shinies
        # TODO mega pokemon generator
        self.files += list(self.sprites_path.glob(str(self.shiny_sprites_sub_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(self.female_shiny_sprites_sub_dir/'*.png')))

        self.files = [str(file) for file in self.files]
        self.files = [file for file in self.files if "png" in file]

        # TODO figure out why PIL can't open this file
        self.files = [file for file in self.files if "10180.png" not in file]
        print(len(self.files))


    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(
            self.files[idx],
        ).convert('RGB')
        '''
        # TODO pull out and test conversion
        raw_image = Image.open(self.files[idx])
        #print(raw_image)
        if raw_image.mode == 'L':
            image = raw_image.convert('RGB')
        elif raw_image.mode == 'RGBA' or raw_image.mode == 'P':
            raw_image = raw_image.convert('RGBA')
            # Impose white background
            # TODO test generating png for sprite?
            raw_image.load()
            background = Image.new("RGB", raw_image.size, (255, 255, 255))
            background.paste(raw_image, mask=raw_image.split()[3]) # 3 is alpha channel
            image = background
        '''

        #image  = image.astype(float)

        if self.transform:
            image = self.transform(image)

        #ksample = {
            #k'image': image
        #k}

        # TODO add label to ds
        #return sample
        return image

def get_ds(
	path='data/external/sprites', 
	size=96
):
	return PokemonDataset(
		path,
		transform=transforms.Compose([
			transforms.Resize((size,size)),
			transforms.ToTensor(),
    ]))