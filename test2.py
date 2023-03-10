# Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import pathlib
data_dir = pathlib.Path('C:/Users/User/Desktop/train').with_suffix('')

print(f'Devices:') 
print(tf.config.list_physical_devices('GPU'))

SEED = 123
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=SEED)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0,
    seed=123,
    image_size=(224,224),
    batch_size=BATCH_SIZE
)

class_names = ds.class_names

train_ds, valid_ds, test_ds = get_dataset_partitions_tf(ds, len(ds), 0.7,0.2,0.1)

total_train = len(train_ds) * BATCH_SIZE
total_valid = len(valid_ds) * BATCH_SIZE
total_test = len(test_ds) * BATCH_SIZE
print(f'Sizes: {total_train}, {total_valid}, {total_test}')

train_ds = train_ds.map(normalize)
valid_ds = valid_ds.map(normalize)
test_ds = test_ds.map(normalize)

train_dataset = train_ds
valid_dataset = valid_ds
test_dataset = test_ds

num_classes = len(ds.class_names)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

t1 = time.time()
history = model.fit(train_dataset,epochs=EPOCHS,steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE) )),validation_steps=int(np.ceil(total_valid / float(BATCH_SIZE))),validation_data=valid_dataset)
print(f"Elapsed: {time.time()-t1}")

model.save('saved_models/model1')