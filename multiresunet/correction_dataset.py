from torch.utils.data import Dataset
import os
import cv2


class WoodCorrectionDataset(Dataset):
    def __init__(self, size=128, dataset_path='./dataset'):
        self.images = []
        images = [cv2.imread(os.path.join(dataset_path, x)) for x in os.listdir(dataset_path)
                  if os.path.isfile(os.path.join(dataset_path, x))]
        self.images = images

    def __getitem__(self, index):
        return None

    def __len__(self):
        # type: () -> int
        return len(self.images)


if __name__ == "__main__":
    dataset = WoodCorrectionDataset()
