# TODO: write keras.utils.Sequence() to load image
from tensorflow.python.keras.utils import Sequence

import numpy as np
import h5py
import PIL.Image as Image


def load_density(file_path):
    gt_file = h5py.File(file_path, 'r')
    groundtruth = np.asarray(gt_file['density'])
    return groundtruth


def random_crop(img, density_map, random_crop_size):
    """
    adapt from https://jkjung-avt.github.io/keras-image-cropping/
    :param img: image matrix (h, w, channel)
    :param density_map: (h, w, channel)
    :param random_crop_size: h_crop, w_crop
    :return:
    """
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)

    return img[y:(y+dy), x:(x+dx), :], density_map[y:(y+dy), x:(x+dx), :]


class DatasetSequence(Sequence):

    def __init__(self, image_path_list, density_path_list, random_crop_size=None):
        self.image_path_list = image_path_list
        self.density_path_list = density_path_list
        self.random_crop_size = random_crop_size

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        density_path = self.density_path_list[idx]

        density = load_density(density_path)
        image = np.array(Image.open(image_path, "r").convert("RGB"))
        density = np.expand_dims(density, axis=3)  # add channel dim

        if self.random_crop_size is not None:
            image, density = random_crop(image, density, self.random_crop_size)

        image = np.expand_dims(image, axis=0) # add batch dim
        density = np.expand_dims(density, axis=0) # add batch dim

        return image, density

    def get_random_crop_image(self, idx):
        image_path = self.image_path_list[idx]
        density_path = self.density_path_list[idx]

        density = load_density(density_path)
        image = np.array(Image.open(image_path, "r").convert("RGB"))
        density = np.expand_dims(density, axis=3)  # add channel dim

        if self.random_crop_size is not None:
            print("crop ", self.random_crop_size)
            image, density = random_crop(image, density, self.random_crop_size)

        image = np.expand_dims(image, axis=0)  # add batch dim
        density = np.expand_dims(density, axis=0)  # add batch dim

        return image, density