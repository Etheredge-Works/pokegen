from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import yaml


class PokemonDataset(Dataset):
    
    normal_sprites_sub_dir = Path("pokemon")
    female_sub_dir = Path("pokemon/female")
    shiny_sprites_sub_dir = Path("pokemon/shiny")
    female_shiny_sprites_sub_dir = Path("pokemon/shiny/female")

    # TODO do I want to use backs?
    # TODO filter question mark
    backs_sub_dir = "pokemon/shiny/female"

    def __init__(
        self, 
        sprites_path, 
        transform=None,
        target_transform=None
    ):
        self.sprites_path = Path(sprites_path)
        self.transform = transform
        self.target_transform = target_transform

        self.files = list(self.sprites_path.glob(str(self.normal_sprites_sub_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(self.female_sub_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(self.shiny_sprites_sub_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(self.female_shiny_sprites_sub_dir/'*.png')))
        # TODO create shiny flag to make generation shiny.
        # TODO shiny generater?
        # TODO cooler shiny geneator trained with only "good" shinies
        # TODO mega pokemon generator

        self.files = [str(file) for file in self.files]
        self.files = [file for file in self.files if "png" in file]

        # TODO figure out why PIL can't open this file
        self.files = [file for file in self.files if "10180.png" not in file]

    
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

        if self.transform:
            transformed_image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(image)

        # TODO consider dict when adding meta information
        #sample = {
            #'image': image
        #}

        # TODO return meta information (mega, evo, types, etc)
        # Return 3 so trainer has more flexibility
        return {
            #'pil_image': transforms.ToTensor()(transforms.Reseize(image),
            'transformed_image': transformed_image, 
            'label': label
        }


# TODO put this inside get loader to not use it in main scope
with open("params.yaml") as f:
    config = yaml.safe_load(f)['pokemon_sprites']


# TODO pull to config file

def get_loader(
    batch_size: int,
    path: str = config['data_dir'],
    resize_shape=config['resize_shape'],
    normalize_mean=config['normalize_mean'],
    normalize_std=config['normalize_std'],
    val_ratio=.1,
    workers=4,
    seed=4,
):
    torch.manual_seed(seed)

    transform = T.Compose([
        #T.RandomRotation(90),
        #T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.Resize(resize_shape),
        #transforms.RandomResizedCrop(resize_shape),
        # TODO vertical flip and rot90
        T.ToTensor(),
        #transforms.Normalize(normalize_mean, normalize_std)
        #transforms.RandomErasing(),
    ])

    target_transform = T.Compose([
        T.Resize(resize_shape),
        T.ToTensor(),
    ])

    ds = PokemonDataset(
        path,
        transform=transform,
        target_transform=target_transform
    )

    val_count = int(len(ds) * val_ratio)
    # TODO note can't find good way to break out val set yet
    train, val = torch.utils.data.random_split(ds, [len(ds)-val_count, val_count])


    trainloader = DataLoader(
        train, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=workers,
        persistent_workers=True, # 'True' makes short epochs start faster
        pin_memory=True
    )

    valloader = DataLoader(
        val, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False, 
        num_workers=workers,
        persistent_workers=True, # 'True' makes short epochs start faster
        pin_memory=True
    )
    return trainloader, valloader