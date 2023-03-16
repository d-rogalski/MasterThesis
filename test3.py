# Libraries 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tf_utils
from time import time
from datetime import date
import json

# Parameters
EPOCHS = 50
BATCH_SIZE = 64
IMG_SIZE = (224, 224)
SEED = 123
FPATH = 'D:/Master thesis/MasterThesis-1'

print('DEVICES')
print(len(tf.config.list_logical_devices('GPU')))

# Dataset preparation

# Loading
ds_kaggle = tf.keras.utils.image_dataset_from_directory(
    'kaggle dataset',
    validation_split=0,
    shuffle=True,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

nature_ds = tf.keras.utils.image_dataset_from_directory(
    'nature dataset',
    validation_split=0,
    shuffle=True,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=138,
)

class_names = ds_kaggle.class_names
no_classes = len(class_names)

# Splitting
train_ds, valid_ds, test_ds = tf_utils.get_dataset_partitions(ds_kaggle, 0.7, 0.15, 0.15, seed=SEED)
print(f'Batches ({BATCH_SIZE} images per batch) in subsets:')
print(f'Train: {len(train_ds)}')
print(f'Validation: {len(valid_ds)}')
print(f'Test: {len(test_ds)}')

# Normalization
train_ds = train_ds.map(tf_utils.normalize)
valid_ds = valid_ds.map(tf_utils.normalize)
test_ds = test_ds.map(tf_utils.normalize)
nature_ds = nature_ds.map(tf_utils.normalize)

# Constructing a model
name = 'VGG16'
# Loading pretrained convolutional layers of the model
pretrained = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(*IMG_SIZE, 3),
    pooling='max'
)

# Disabling training of pretrained layers
for layer in pretrained.layers:
    layer.trainable = False

# Creating the model - adding fully connected layers on top of model
model = tf.keras.models.Sequential(name=name)
model.add(pretrained)
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5, seed=SEED))
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5, seed=2*SEED))
model.add(tf.keras.layers.Dense(no_classes, activation='softmax'))

# Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fit
t0 = time()

history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=valid_ds,
    steps_per_epoch=len(train_ds),
    validation_steps=len(valid_ds)
)
training_time = time() - t0
print(f'Elapsed: {training_time / 60} min')