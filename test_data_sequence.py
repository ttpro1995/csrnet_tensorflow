
from data_sequence import DatasetSequence


def test_data_sequence():
    """

    :return:
    """
    image_list = ["test_data/image/1.jpg", "test_data/image/2.jpg"]
    density_list = ["test_data/density/1.h5", "test_data/density/2.h5"]
    dataset = DatasetSequence(image_list, density_list)
    for image, density in dataset:
        print(image.shape)
        print(density.shape)
        print("--------")


if __name__ == "__main__":
    test_data_sequence()