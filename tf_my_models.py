import tensorflow as tf
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils.vis_utils import plot_model
from keras.layers import Input, BatchNormalization, Activation, Add, ZeroPadding2D
from keras.models import Model

#----- GoogLeNet -----
# Inception module
def inception_module(input_layer, conv_filters, reduce_filters, pool_proj):
    layer0 = Conv2D(conv_filters[0], (1,1), padding='same', activation='relu')(input_layer)

    layer1 = Conv2D(reduce_filters[0], (1,1), padding='same', activation='relu')(input_layer)
    layer1 = Conv2D(conv_filters[1], (3,3), padding='same', activation='relu')(layer1)

    layer2 = Conv2D(reduce_filters[1], (1,1), padding='same', activation='relu')(input_layer)
    layer2 = Conv2D(conv_filters[2], (5,5), padding='same', activation='relu')(layer2)

    layer3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_layer)
    layer3 = Conv2D(pool_proj, (1,1), padding='same', activation='relu')(layer3)

    return tf.keras.layers.concatenate([layer0, layer1, layer2, layer3])

# Building network
def GoogLeNet_model(input_shape, no_classes):
    # Input Layer
    input_image = Input(shape=input_shape)

    # Whole architecture:
    layers = Conv2D(64, (7, 7), 2, padding='same', activation='relu')(input_image)
    layers = MaxPooling2D(3, 2, padding='same')(layers)
    layers = Conv2D(192, 3, padding='same', activation='relu')(layers)
    layers = MaxPooling2D(3, 2, padding='same')(layers)
    layers = inception_module(layers, (64, 128, 32), (96, 16), 32)
    layers = inception_module(layers, (128, 192, 96), (128, 32), 64)
    layers = MaxPooling2D(3, 2, padding='same')(layers)
    layers = inception_module(layers, (192, 208, 48), (96, 16), 64)
    layers = inception_module(layers, (160, 224, 64), (112, 24), 64)
    layers = inception_module(layers, (128, 256, 64), (128, 24), 64)
    layers = inception_module(layers, (112, 288, 64), (144, 32), 64)
    layers = inception_module(layers, (256, 320, 128), (160, 32), 128)
    layers = MaxPooling2D(3, 2, padding='same')(layers)
    layers = inception_module(layers, (256, 320, 128), (160, 32), 128)
    layers = inception_module(layers, (384, 384, 128), (192, 48), 128)
    layers = AveragePooling2D(7, 7, padding='same')(layers)
    layers = Dropout(0.4)(layers)
    layers = Flatten()(layers)
    layers = Dense(1000, activation='relu')(layers)
    layers = Dense(no_classes, activation=tf.nn.softmax)(layers)

    model = Model([input_image], layers, name='GoogLeNet')
    return model

#----- VGG -----
def VGG16_model(input_shape, no_classes):
    # Input layer
    input_image = Input(input_shape)

    # Output layers
    layers = Conv2D(64, (3,3), (1,1), padding='same', activation='relu')(input_image)
    layers = Conv2D(64, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Conv2D(128, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(128, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Dense(4096, activation='relu')(layers)
    layers = Dense(4096, activation='relu')(layers)
    layers = Dense(1000, activation='relu')(layers)
    layers = Dense(no_classes, activation='relu')(layers)

    # Model
    model = Model([input_image], layers, name='VGG-16')
    return model


def VGG19_model(input_shape, no_classes):
    # Input layer
    input_image = Input(input_shape)

    # Output layers
    layers = Conv2D(64, (3,3), (1,1), padding='same', activation='relu')(input_image)
    layers = Conv2D(64, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Conv2D(128, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(128, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = Conv2D(512, (3,3), (1,1), padding='same', activation='relu')(layers)
    layers = MaxPooling2D((2,2), (2,2))(layers)
    layers = Dense(4096, activation='relu')(layers)
    layers = Dense(4096, activation='relu')(layers)
    layers = Dense(1000, activation='relu')(layers)
    layers = Dense(no_classes, activation='relu')(layers)

    # Model
    model = Model([input_image], layers, name='VGG-19')
    return model


#----- ResNet -----
# Identity module:
def identity_module(input_layer, filter):
    layer1 = Conv2D(filter, (3, 3), (1, 1), padding='same')(input_layer)
    layer1 = BatchNormalization(axis=3)(layer1)
    layer1 = Activation('relu')(layer1)

    layer2 = Conv2D(filter, (3, 3), (1, 1), padding='same')(layer1)
    layer2 = BatchNormalization(axis=3)(layer2)

    # Adding residue
    output = Add()([input_layer, layer2])
    output = Activation('relu')(output)

    return output

# Convolutional module
def convolutional_module(input_layer, filter):
    layer1 = Conv2D(filter, (3, 3), (2, 2), padding='same')(input_layer)
    layer1 = BatchNormalization(axis=3)(layer1)
    layer1 = Activation('relu')(layer1)
    
    layer2 = Conv2D(filter, (3, 3), (1, 1), padding='same')(layer1)
    layer2 = BatchNormalization(axis=3)(layer2)

    skip_layer = Conv2D(filter, (1, 1), (2, 2))(input_layer)

    # Adding residue
    output = Add()([layer2, skip_layer])
    output = Activation('relu')(output)

    return output


def ResNet34_model(input_shape, no_classes):
    # Input layer
    input_image = Input(input_shape)

    # Initial layers
    init_layer = Conv2D(64, (7, 7), (2, 2), padding='same')(input_image)
    init_layer = BatchNormalization()(init_layer)
    init_layer = Activation('relu')(init_layer)
    init_layer = MaxPooling2D((3, 3), (2, 2), padding='same')(init_layer)

    # Residual layers
    block1 = identity_module(init_layer, 64)
    block1 = identity_module(block1, 64)
    block1 = identity_module(block1, 64)
    block1 = identity_module(block1, 64)

    block2 = convolutional_module(block1, 128)
    block2 = identity_module(block2, 128)
    block2 = identity_module(block2, 128)
    block2 = identity_module(block2, 128)

    block3 = convolutional_module(block2, 256)
    block3 = identity_module(block3, 256)
    block3 = identity_module(block3, 256)
    block3 = identity_module(block3, 256)
    block3 = identity_module(block3, 256)
    block3 = identity_module(block3, 256)

    block4 = convolutional_module(block3, 512)
    block4 = identity_module(block4, 512)
    block4 = identity_module(block4, 512)
    block4 = identity_module(block4, 512)

    # Output dense layers
    output = AveragePooling2D((2, 2), padding='same')(block4)
    output = Flatten()(output)
    output = Dense(512, activation='relu')(output)
    output = Dense(no_classes, activation=tf.nn.softmax)(output)

    # Model
    model = Model([input_image], output, name='ResNet34')
    return model


def ResNet18_model(input_shape, no_classes):
    # Input layer
    input_image = Input(input_shape)

    # Initial layers
    init_layer = Conv2D(64, (7, 7), (2, 2), padding='same')(input_image)
    init_layer = BatchNormalization()(init_layer)
    init_layer = Activation('relu')(init_layer)
    init_layer = MaxPooling2D((3, 3), (2, 2), padding='same')(init_layer)

    # Residual layers
    block1 = identity_module(init_layer, 64)
    block1 = identity_module(block1, 64)

    block2 = convolutional_module(block1, 128)
    block2 = identity_module(block2, 128)

    block3 = convolutional_module(block2, 256)
    block3 = identity_module(block3, 256)

    block4 = convolutional_module(block3, 512)
    block4 = identity_module(block4, 512)

    # Output dense layers
    output = AveragePooling2D((2, 2), padding='same')(block4)
    output = Flatten()(output)
    output = Dense(512, activation='relu')(output)
    output = Dense(no_classes, activation=tf.nn.softmax)(output)

    # Model
    model = Model([input_image], output, name='ResNet18')
    return model


