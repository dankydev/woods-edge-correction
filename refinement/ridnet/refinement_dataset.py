import numpy as np
import random
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


def get_files_in_dir_with_extension(dir_path,
                                    extension=(".png", ".jpg")):
    """
    """
    if dir_path is None or not isinstance(dir_path, Path):
        return
    filenames = [x for x in dir_path.iterdir() if x.is_file()]
    filenames = [x for x in filenames if (x.suffix in extension)]
    return filenames


def png_to_numpy(path):  # also jpg are ok
    if path is None or not isinstance(path, Path):
        return
    img = Image.open(path)
    numpy = np.array(img).copy()
    return numpy


class RefinementDataset(Dataset):
    def __init__(self, dataset_path):
        filenames = get_files_in_dir_with_extension(
            dir_path=dataset_path)
        self.images = []
        self.to_tensor = transforms.ToTensor()

        for path in filenames:
            array = png_to_numpy(path=path)
            self.images.append(array)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError

        both = self.images[i]
        x = both[:128, :, :]
        y = both[128:, :, :]

        return [self.to_tensor(x), self.to_tensor(y)]


if __name__ == "__main__":
    dataset = RefinementDataset(dataset_path=Path("C:\\Users\\manic\\Documents\\woods-edge-correction\\dataset\\refinement\\output\\train\\wood"))
    a = dataset[0]

    plt.figure()
    plt.imshow(a[0])
    plt.waitforbuttonpress()

    plt.figure()
    plt.imshow(a[1])
    plt.waitforbuttonpress()
