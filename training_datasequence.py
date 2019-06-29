
import numpy as np
import glob
from data_sequence import DatasetSequence
from model import build_model


DATA_PATH = "/data/cv_data/UCFCrowdCountingDataset_CVPR13_with_people_density_map/UCF_CC_50/"

if __name__ == "__main__":
    image_list = glob.glob(DATA_PATH+"*.jpg")
    density_list = list(map(lambda s: s.replace(".jpg", ".h5"), image_list))

    dataset = DatasetSequence(image_list, density_list)
    model = build_model()

    for image, density in dataset:
        model.fit(image, density)




