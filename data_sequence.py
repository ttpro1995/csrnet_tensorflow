# TODO: write keras.utils.Sequence() to load image

from keras.utils import Sequence
import numpy as np
import h5py
import PIL.Image as Image

def load_density(file_path):
    gt_file = h5py.File(file_path, 'r')
    groundtruth = np.asarray(gt_file['density'])
    return groundtruth


class DatasetSequence(Sequence):

    def __init__(self, image_path_list, density_path_list):
        self.image_path_list = image_path_list
        self.density_path_list = density_path_list

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        density_path = self.density_path_list[idx]

        density = load_density(density_path)
        image = np.array(Image.open(image_path, "r"))

        image = np.expand_dims(image, axis=0)
        density = np.expand_dims(density, axis=0)
        return image, density