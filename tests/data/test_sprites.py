import pytest
from data.sprites import PokemonDataset
from torchvision import transforms
from PIL import Image
import torch


@pytest.fixture
def ds():
    return PokemonDataset('data/external/sprites')


def test_class(ds):
    print(len(ds))
    for item in ds:
        assert item.mode == 'RGB'
        assert type(item) == Image.Image
    print(ds[0])
    #ds[0].save('pls.jpg')


def test_transform():
    tran_ds = PokemonDataset(
        'data/external/sprites',
        transform=transforms.ToTensor()
        )
    for item in tran_ds:
        assert type(item) == torch.Tensor