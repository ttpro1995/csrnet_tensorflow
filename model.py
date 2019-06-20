from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D
import numpy as np


def build_model():
    vgg16_model = VGG16(weights='imagenet', include_top=False)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    x = vgg16_model.get_layer('block4_conv3').output
    x = UpSampling2D(size=(8, 8))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=2, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=2, padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=2, padding='same')(x)
    x = Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=1, padding='same')(x)
    model = Model(inputs=vgg16_model.input, outputs=x)
    return model