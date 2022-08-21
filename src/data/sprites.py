from collections import namedtuple
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import yaml
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# poke_label = namedtuple('PokeLabel', ['shiny', 'id', 'mega', 'female', 'back', 'is flipped'])
# TODO could train to take in normal and output back with flag injecting into latent space

# TODO https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/tutorial9/AE_CIFAR10.ipynb#scrollTo=KwnvbWOkPayE
# TODO talks about not using batch norm with auto encoders

def extract_ega(name):
    # name = name.lower()
    string = ''
    if 'Mega' in name:
        string += '-mega'
    if 'Alolan' in name:
        string += '-alola'
    
    if string:
        if 'X' == name[-1]:
            string += '-x'
        if 'Y' == name[-1]:
            string += '-y'
    return string

def filename(in_df):
    filename = in_df['pokedex_number']
    name = in_df['name']
    print(name)
    name = name.apply(extract_ega)
    filename = filename.apply(lambda x: str(x))
    filename += name

    return filename + '.png'
    
def trim_dex(dex_df):
    out_df = pd.DataFrame()
    out_df['filename'] = filename(dex_df)
    out_df['type_1'] = dex_df['type_1'].astype("category")
    out_df['type_2'] = dex_df['type_2'].astype("category")
    out_df['status'] = dex_df['status'].astype("category")
    return out_df

class PokemonDataset(Dataset):
    
    normal_sprites_sub_dir = Path("pokemon")
    main_dir = Path("pokemon")
    female_dir = Path("female")
    shiny_dir = Path("shiny")
    back_dir = Path("back")
    models_dir = Path("model")
    art_dir = Path("other/official-artwork")

    # TODO do I want to use backs?
    # TODO filter question mark

    def __init__(
        self, 
        sprites_path, 
        transform=None,
        target_transform=None,
        all=True,
        main=False,
        shiny=False,
        backs=False,
        models=True,
        home=True,
        art=True,
    ):
        # NOTE just main looks okay.
        self.sprites_path = Path(sprites_path)
        self.transform = transform
        self.target_transform = target_transform
        dex_df = pd.read_csv('data/external/pokedex_(Update_05.20).csv')
        self.dex_data = trim_dex(dex_df)
        self.dex_encoder = OneHotEncoder()
        self.dex_encoder.fit(self.dex_data[['type_1', 'type_2', 'status']])
        self.dex_features = self.dex_encoder.transform(self.dex_data[['type_1', 'type_2', 'status']]).toarray()
        # self.dex_features = torch.tensor(self.dex_features, dtype=torch.float)

        self.files = []
        self.labels = []
        
        # TODO guard clauses?
        # ALL
        if all:
            self.files += list(self.sprites_path.glob(str(
                self.normal_sprites_sub_dir/'**'/'*.png')))
        else:

            # Main area
            if main:
                self.files += list(self.sprites_path.glob(str(
                    self.normal_sprites_sub_dir/'*.png')))
                self.files += list(self.sprites_path.glob(str(
                    self.main_dir/self.female_dir/'*.png')))
                # Shinys
                if shiny:
                    self.files += list(self.sprites_path.glob(str(
                        self.main_dir/self.shiny_dir/'*.png')))
                    self.files += list(self.sprites_path.glob(str(
                        self.main_dir/self.shiny_dir/self.female_dir/'*.png')))

                # Backs
                if backs:
                    self.files += list(self.sprites_path.glob(str(
                        self.main_dir/self.back_dir/'*.png')))
                    self.files += list(self.sprites_path.glob(str(
                        self.main_dir/self.back_dir/self.female_dir/'*.png')))
                    # Shiny Backs
                    if shiny:
                        self.files += list(self.sprites_path.glob(str(
                            self.main_dir/self.back_dir/self.shiny_dir/'*.png')))
                        self.files += list(self.sprites_path.glob(str(
                            self.main_dir/self.back_dir/self.shiny_dir/self.female_dir/'*.png')))

            # Models
            if models:
                self.files += list(self.sprites_path.glob(str(
                    self.main_dir/self.models_dir/'*.png')))

            # Home
            if home:
                self.files += list(self.sprites_path.glob(str(
                    self.main_dir/"other"/"home"/"**"/"*.png")))

            # Art
            if art:
                self.files += list(self.sprites_path.glob(str(
                    self.main_dir/self.art_dir/'*.png')))

        
        # TODO create shiny flag to make generation shiny.
        # TODO shiny generater?
        # TODO cooler shiny geneator trained with only "good" shinies
        # TODO mega pokemon generator

        print(f"{len(self.files)} files found")
        self.files = [str(file) for file in self.files]
        self.files = [file for file in self.files if "png" in file]

        # TODO figure out why PIL can't open this file
        self.files = [file for file in self.files if "10186.png" not in file]

        # Remove "?" image
        self.files = [file for file in self.files if "/0.png" not in file]

        # Remove duplicates
        self.files = [file for file in self.files if "transparent" not in file]
        if not backs:
            self.files = [file for file in self.files if "back" not in file]

        self.labels = []
        for file_path in self.files:
            label = np.zeros_like(self.dex_features[0])
            for i, filename in enumerate(self.dex_data['filename']):
                if filename in file_path:
                    # label = self.dex_features[i] # TODO add back when labesls are desired
                    break
            self.labels.append(label)
        print(f"{len(self.files)} files found")
            

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # image = Image.open(
        #     self.files[idx],
        # ).convert('RGB')
        # TODO pull out and test conversion
        file_path = self.files[idx]

        label = self.labels[idx]

        raw_image = Image.open(file_path)
        #print(raw_image)
        if raw_image.mode == 'L':
            image = raw_image.convert('RGB')
        elif raw_image.mode in ['RGBA', 'P', 'LA']:
            raw_image = raw_image.convert('RGBA')
            # Impose white background
            # TODO test generating png for sprite?
            raw_image.load()
            # TODO why does white background work best?
            background = Image.new("RGB", raw_image.size, (255, 255, 255))
            # background = Image.new("RGB", raw_image.size, (0, 0, 0))
            background.paste(raw_image, mask=raw_image.split()[3]) # 3 is alpha channel
            image = background
        else:
            print(raw_image.mode)


        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        # if self.target_transform:
        #     label = self.target_transform(image)
        # else:
        #     label = transformed_image

        # TODO consider dict when adding meta information
        #sample = {
            #'image': image
        #}

        # TODO return meta information (mega, evo, types, etc)
        # Return 3 so trainer has more flexibility
        label = torch.tensor(label, dtype=torch.float)
        return transformed_image, label
        print(transformed_image.shape)
        return {
            #'pil_image': transforms.ToTensor()(transforms.Reseize(image),
            'transformed_image': transformed_image, 
            'label': label
        }


# TODO put this inside get loader to not use it in main scope
with open("params.yaml") as f:
    config = yaml.safe_load(f)['pokemon_sprites']

# TODO pull to config file
def denormalize(data):
    return data.mul(torch.as_tensor(config['normalize_mean'])).add(torch.as_tensor(config['normalize_std']))

def get_loader(
    batch_size: int,
    path: str = config['data_dir'],
    resize_shape=config['resize_shape'],
    normalize_mean=config['normalize_mean'],
    normalize_std=config['normalize_std'],
    val_ratio=.1,
    workers=4,
    seed=4,
    ds_kwargs={},
):
    torch.manual_seed(seed)
    print(f"Resizing to {resize_shape}")
    print(f"Normalizing with mean {normalize_mean} and std {normalize_std}")

    transform = T.Compose([
        T.Resize(resize_shape),
        # T.Resize(64),
        # T.RandomRotation(90),
        # T.RandomVerticalFlip(),
        # T.Grayscale(),
        # TODO could train a model to colorize...
        T.RandomHorizontalFlip(),
        # T.RandomResizedCrop(resize_shape),
        # TODO vertical flip and rot90
        T.ToTensor(),
        # T.Normalize(normalize_mean, normalize_std)
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # T.Normalize(0.5, 0.5),
        # TODO test moving norm up before resize
    ])

    # target_transform = T.Compose([
    #     T.Resize(resize_shape),
    #     T.ToTensor(),
    #     #T.Normalize(normalize_mean, normalize_std)
    # ])

    ds = PokemonDataset(
        path,
        transform=transform,
        **ds_kwargs
    )

    if val_ratio > 0:
        val_count = int(len(ds) * val_ratio)
        # TODO note can't find good way to break out val set yet
        train, val = torch.utils.data.random_split(ds, [len(ds)-val_count, val_count])

        # TODO still inconsistent
        valloader = DataLoader(
            val, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False, 
            num_workers=workers//2,
            persistent_workers=True, # 'True' makes short epochs start faster
            pin_memory=True
        )
    else:
        train = ds
        valloader = None

    trainloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=True, # 'True' makes short epochs start faster
        pin_memory=True
    )

    return trainloader, valloader

import pytorch_lightning as pl

class PokeModule(pl.LightningDataModule):
    def __init__(self, *loader_args, **loarder_kwargs):
        super().__init__()
        self.trainloader, self.valloader = get_loader(
                *loader_args,
                batch_size=64,
                workers=16,
                val_ratio=0.1
                **loarder_kwargs
        )
    
    def prepare_data(self):
        pass

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

    def test_dataloader(self):
        return self.valloader
    
    def predict_dataloader(self):
        return self.valloader

    def size(self):
        return (3, 64, 64)
    

    