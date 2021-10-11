import numpy as np
import random
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


class WoodCorrectionDataset(Dataset):
    """
    load full imgs in RAM, left and right crops are generated randomly
    """

    def __init__(self,
                 dataset_path,
                 cut_size_h_w=(128, 256),
                 max_shift=15,
                 min_shift=0,
                 test_mode=False):
        """
        """
        self.cut_h_full, self.cut_w_full = cut_size_h_w
        self.cut_h, self.cut_w = self.cut_h_full, self.cut_w_full // 2

        self.max_shift = max_shift
        self.min_shift = min_shift
        # self.shift = shift
        self.possible_shifts = np.arange(start=0,
                                         stop=self.max_shift + 1,
                                         step=1)  # 0-15 included
        # 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        self.shifts_weights = [
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02,  # 0,1,2,3,4,5
            0.02, 0.02, 0.02, 0.02,  # ,6,7,8,9
            0.3,  # ,10,
            0.1, 0.1, 0.1, 0.1, 0.1  # 11, 12, 13, 14, 15
        ]
        self.max_cut_h = self.cut_h_full + self.max_shift * 2
        self.max_cut_w = self.cut_w_full + self.max_shift

        self.counter = 0
        self.full_imgs = []

        self.test_mode = test_mode

        filenames = get_files_in_dir_with_extension(
            dir_path=dataset_path,
            extension=(".png", ".jpg"))

        self.counter = 0
        self.counter_multiplier = 8
        self.to_tensor = transforms.ToTensor()

        for path in filenames:
            array = png_to_numpy(path=path)
            assert len(array.shape) == 3
            H, W, three = array.shape
            assert three == 3
            self.counter += self.counter_multiplier * ((H * W) // (self.max_cut_h * self.max_cut_w))
            self.full_imgs.append(array)

    @property
    def n_images(self):
        return len(self.full_imgs)

    def __len__(self):
        return self.counter

    def __getitem__(self, i):
        if i >= self.__len__():
            raise IndexError

        while True:  # find image that can be cut correctly
            idx = random.randint(0, self.n_images - 1)
            full_img = self.full_imgs[idx]
            H, W, three = full_img.shape
            if H > self.max_cut_h and W > self.max_cut_w:
                break

        if not self.test_mode:
            """
            data aug sequence here, use albumentations
            """
            # random flip ud and lr
            if random.random() < 0.5:
                full_img = np.flipud(full_img)
            if random.random() < 0.5:
                full_img = np.fliplr(full_img)

        # select random cut coordinates
        r = random.randint(0 + self.max_shift, H - self.cut_h_full - self.max_shift)
        c = random.randint(0, W - self.cut_w_full - self.max_shift)

        # true aligned cut
        h_start = r
        h_end = h_start + self.cut_h_full
        w_start = c
        w_end = w_start + self.cut_w_full
        cut_aligned = full_img[h_start:h_end, w_start:w_end, :].copy()

        # left cut
        cut_left = cut_aligned[0:self.cut_h, 0: self.cut_w, :].copy()

        # right cut -> misaligned
        up_down_no_shift = random.choices(
            population=self.possible_shifts,
            weights=self.shifts_weights,  # [0.45, 0.45, 0.1],
            k=1)[0]
        neg_pos = random.choice(seq=(-1, 1))
        up_down_no_shift = up_down_no_shift * neg_pos

        right_no_shift = random.choices(
            population=self.possible_shifts,  # (self.shift, 0),
            weights=self.shifts_weights,
            k=1)[0]

        h_start = r + up_down_no_shift
        h_end = h_start + self.cut_h
        w_start = c + self.cut_w + right_no_shift
        w_end = w_start + self.cut_w
        cut_right = full_img[h_start:h_end, w_start:w_end].copy()

        assert cut_left.shape == cut_right.shape
        cut_misaligned = np.concatenate((cut_left, cut_right), axis=1)

        return self.to_tensor(cut_misaligned), self.to_tensor(cut_aligned)


def main():
    ds = WoodCorrectionDataset(
        dataset_path=Path("../dataset/Legni02@resize_16x"),
        cut_size_h_w=(128, 256),
        max_shift=15,
        min_shift=0,
        test_mode=False
    )

    from torch.utils.data import DataLoader
    dl = DataLoader(
        dataset=ds,
        batch_size=1,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    count = 0
    for img in ds:  # for step, img in enumerate(dl):
        count = count + 1
        c, h, w = img[0].shape
        border = torch.zeros(c, 2, w, device="cpu")
        cuts = torch.cat(tensors=[img[0], border], dim=1)
        cuts = torch.cat(tensors=[cuts, img[1]], dim=1)
        #show_tensor(cuts)
        pass
    print(count)
    pass


if __name__ == '__main__':
    main()
