import numpy as np
from PIL import Image
from pathlib import Path


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
