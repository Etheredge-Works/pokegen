from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import yaml


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
        target_transform=None
    ):
        self.sprites_path = Path(sprites_path)
        self.transform = transform
        self.target_transform = target_transform

        self.files = []
        # Main area
        self.files += list(self.sprites_path.glob(str(
            self.normal_sprites_sub_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(
            self.main_dir/self.female_dir/'*.png')))
        # Shinys
        self.files += list(self.sprites_path.glob(str(
            self.main_dir/self.shiny_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(
            self.main_dir/self.shiny_dir/self.female_dir/'*.png')))

        # Models
        # self.files += list(self.sprites_path.glob(str(
        #     self.main_dir/self.models_dir/'*.png')))

        # Backs
        """
        self.files += list(self.sprites_path.glob(str(
            self.main_dir/self.back_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(
            self.main_dir/self.back_dir/self.female_dir/'*.png')))
        # Shiny Backs
        self.files += list(self.sprites_path.glob(str(
            self.main_dir/self.back_dir/self.shiny_dir/'*.png')))
        self.files += list(self.sprites_path.glob(str(
            self.main_dir/self.back_dir/self.shiny_dir/self.female_dir/'*.png')))
        # Art
        #self.files += list(self.sprites_path.glob(str(
            #self.main_dir/self.art_dir/'*.png')))
        """

        
        # TODO create shiny flag to make generation shiny.
        # TODO shiny generater?
        # TODO cooler shiny geneator trained with only "good" shinies
        # TODO mega pokemon generator

        self.files = [str(file) for file in self.files]
        self.files = [file for file in self.files if "png" in file]

        # TODO figure out why PIL can't open this file
        self.files = [file for file in self.files if "10180.png" not in file]

        # Remove "?" image
        self.files = [file for file in self.files if "0.png" not in file]

    
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
        else:
            transformed_image = image

        if self.target_transform:
            label = self.target_transform(image)
        else:
            label = transformed_image

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
):
    torch.manual_seed(seed)

    transform = T.Compose([
        #T.RandomRotation(90),
        #T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
        T.Resize(resize_shape),
        #T.RandomResizedCrop(resize_shape),
        # TODO vertical flip and rot90
        T.ToTensor(),
        #T.Normalize(normalize_mean, normalize_std)
        # TODO test moving norm up before resize
        #transforms.RandomErasing(),
    ])

    target_transform = T.Compose([
        T.Resize(resize_shape),
        T.ToTensor(),
        #T.Normalize(normalize_mean, normalize_std)
    ])

    ds = PokemonDataset(
        path,
        transform=transform,
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

    # TODO still inconsistent
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