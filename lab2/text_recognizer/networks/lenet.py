from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Convolution2D(32, 3,3 ,border_mode="same",
            input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # second set of CONV => RELU => POOL
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    ##### Your code above (Lab 2)

    return model

