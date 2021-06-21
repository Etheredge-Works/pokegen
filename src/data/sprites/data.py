from torch.utils.data import Dataset
from torchvision import transforms


class PokemonDataset(Dataset):
    
    normal_sprites_sub_dir = "pokemon"
    female_sub_dir = "female"

    def __init__(self, sprites_path, transform=None):
        self.sprites_path = sprites_path
        self.transform = transform
        self.files = os.listdir(os.path.join(sprites_path, self.normal_sprites_sub_dir))
        print(self.files[:10])
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(
            os.path.join(
                self.sprites_path, 
                self.normal_sprites_sub_dir,
                self.files[idx]),
        ).convert('RGB')
        #image  = image.astype(float)

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image
        }

        return sample

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