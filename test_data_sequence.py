
from data_sequence import DatasetSequence
import glob
import PIL.Image as Image

def test_data_sequence():
    """

    :return:
    """
    print("test_data_sequence")
    image_list = ["test_data/image/1.jpg", "test_data/image/2.jpg"]
    density_list = ["test_data/density/1.h5", "test_data/density/2.h5"]
    dataset = DatasetSequence(image_list, density_list)
    for image, density in dataset:
        print(image.shape)
        print(density.shape)
        print("--------")
    print("end test data sequence")


def test_data_sequence_with_random_crop():
    print("test_data_sequence_with_random_crop")
    image_list = ["test_data/image/1.jpg", "test_data/image/2.jpg"]
    density_list = ["test_data/density/1.h5", "test_data/density/2.h5"]
    dataset = DatasetSequence(image_list, density_list, random_crop_size=(224, 224))
    for image, density in dataset:
        print(image.shape)
        print(density.shape)
        print("--------")
    print("end test_data_sequence_with_random_crop")


def test_mutiple_random_crop():
    print("test_data_sequence_with_random_crop")
    image_list = ["test_data/image/1.jpg", "test_data/image/2.jpg"]
    density_list = ["test_data/density/1.h5", "test_data/density/2.h5"]
    dataset = DatasetSequence(image_list, density_list, random_crop_size=(224, 224))
    imgs, densities = dataset.get_random_crop_image_batch(1, 3)
    for i in range(3):
        print(imgs.shape)
        print(densities.shape)
        pil_img = Image.fromarray(imgs[i])
        pil_img.save("test_data/train" + str(i) +".png")
        print("--------")
    print("end test_data_sequence_with_random_crop")


if __name__ == "__main__":
    test_data_sequence()
    test_data_sequence_with_random_crop()
    test_mutiple_random_crop()