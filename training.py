import glob
import h5py
import numpy as np
import PIL.Image as Image
import matplotlib
from matplotlib.image import  imsave

import tensorflow as tf

DATA_PATH = "/data/cv_data/UCFCrowdCountingDataset_CVPR13_with_people_density_map/UCF_CC_50/"

from keras_preprocessing.image import ImageDataGenerator
from data_sequence import load_density



def load_dataset():
    DATA_PATH = "/data/cv_data/UCFCrowdCountingDataset_CVPR13_with_people_density_map/UCF_CC_50/"
    p = glob.glob(DATA_PATH+"*.jpg")
    image_stack = []
    density_stack = []
    for image_path in p[:1]:
        print(image_path)
        density_path = image_path.replace(".jpg", ".h5")
        print(density_path)

        # image = np.expand_dims(np.array(Image.open(image_path, "r")), axis=0)
        # density = np.expand_dims(load_density(density_path), axis=0)

        image = np.array(Image.open(image_path, "r"))
        density = load_density(density_path)

        print("image shape ", image.shape)
        print("density shape ", density.shape)

        image_stack.append(image)
        density_stack.append(density)

    return image_stack, density_stack


if __name__ == "__main__":
    image_stack, density_stack = load_dataset()

    datagen = ImageDataGenerator(dtype='uint8')
    image_generator = datagen.flow_from_directory(directory="/data/cv_data/UCFCrowdCountingDataset_CVPR13_with_people_density_map/1_jpg/",
                                                  target_size=(256, 256),
                                                  class_mode=None,
                                                  color_mode="grayscale")

    image = image_generator.next()[0][:,:,0]
    print(image.shape)
    pil_img = Image.fromarray(image)
    pil_img.convert('L').save("file.png")
    print(image.shape)


    # imsave('name.png', image)

    # print(image_stack[0].shape)
    # imsave('original.png', image_stack[0])
    print(image_stack[0].dtype)
    # pil_img = Image.fromarray(image_stack[0])
    # pil_img.save("file.png")


    # datagen.fit(image_stack)
    # for image in image_stack:
    #     print(image.shape)