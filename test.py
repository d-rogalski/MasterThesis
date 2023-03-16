# Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import time
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Creating directory to images
import pathlib
c = input('Choice: ')
if c == '0':
    data_dir = pathlib.Path('D:\Master thesis\Photos\european').with_suffix('')
else:
   data_dir = pathlib.Path('/mnt/d/Master thesis/Photos/european').with_suffix('')
print(len(list(data_dir.glob('*/*.jpg'))))

print(tf.config.list_physical_devices('GPU'))

SEED = 123

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

BATCH_SIZE = 32

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
print(total_train, total_valid, total_test)

# Normalizing data
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_ds = train_ds.map(normalize)
valid_ds = valid_ds.map(normalize)
test_ds = test_ds.map(normalize)

# Configuring performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
valid_dataset = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Training
num_classes = len(ds.class_names)
model1 = tf.keras.models.Sequential([
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

model2 = tf.keras.models.Sequential([
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

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

EPOCHS = 10
t1 = time.time()
history = model1.fit(train_dataset,epochs=EPOCHS,steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE) )),validation_steps=int(np.ceil(total_valid / float(BATCH_SIZE))),validation_data=valid_dataset)
print(f"Elapsed: {time.time()-t1}")

for i in range(EPOCHS):
   model2.fit(train_dataset, epochs=1,steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE) )),validation_steps=int(np.ceil(total_valid / float(BATCH_SIZE))),validation_data=valid_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

f = plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.close(f)
plt.savefig('1.png')

# Evaluation
test_loss, test_accuracy = model.evaluate(test_ds)
print(test_accuracy)

# Predictions
def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]==class_names[predicted_label]),
                                class_names[predicted_label])

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

for test_images, test_labels in test_ds.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

num_rows = 4
num_cols = 2
num_images = num_rows*num_cols
f = plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

plt.close(f)
plt.savefig('2.png')