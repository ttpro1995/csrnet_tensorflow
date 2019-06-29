from tensorflow.python.keras.models import load_model
from data_sequence import DatasetSequence
import glob
import PIL.Image as Image
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import os
import numpy as np


def save_density_map(density_map, name):
    plt.figure(dpi=600)
    plt.axis('off')
    plt.margins(0, 0)
    plt.imshow(density_map, cmap=CM.jet)
    plt.savefig(os.path.join("experiment", name), dpi=600, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    MODEL_PATH = "experiment/single_image_random_crop_experiment_model_6.model"
    DATA_PATH = "/data/cv_data/UCFCrowdCountingDataset_CVPR13_with_people_density_map/UCF_CC_50/"


    image_list = glob.glob(DATA_PATH+"*.jpg")
    density_list = list(map(lambda s: s.replace(".jpg", ".h5"), image_list))

    dataset = DatasetSequence(image_list, density_list, random_crop_size=(250, 250))


    img_train, density_train = dataset.get_random_crop_image(0)
    pil_img = Image.fromarray(img_train[0])
    pil_img.save("experiment/train.png")


    model = load_model(MODEL_PATH)

    print("label ", density_train.sum())
    save_density_map(np.squeeze(density_train[0], axis=2), "label.png")

    pred = model.predict(img_train)

    save_density_map(np.squeeze(pred[0], axis=2), "predict.png")
    print("predict ", np.squeeze(pred[0], axis=2).shape, np.squeeze(pred[0], axis=2).sum())

    print("------------")


