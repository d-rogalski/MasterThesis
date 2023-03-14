__name__ = 'Utility functions'

def get_dataset_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000, seed=None):
    """
    Function for splitting complete dataset into train, validation and test partitions.
s
    Arguments:
        ds: dataset to split
        train_split: part of dataset for training
        validation split: part of dataset for validation
        test_split: part of dataset for testing
        shuffle: whether to shuffle the data before splitting
        shuffle_size: size of buffer for shuffling
        seed: seed for shuffling

    Returns:
        train_ds, val_ds, test_ds

    Example:
        `train_ds, val_ds, test_ds = get_dataset_partitions(dataset, 0.7, 0.2, 0.1, seed=123)`
    """
    assert (train_split + test_split + val_split - 1) <= 1e-5
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=seed)
    
    ds_size = len(ds)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


def normalize(images, labels):
    from tensorflow import cast, float32
    """
    Function normalizing batch of images in dataset from integer rgb values [0:255] to float32 values [0:1]

    Arguments:
        images: images in batch
        labels: corresponding labels

    Returns:
        normalized_images, labels

    Example:
        `dataset.map(normalize)`
    """
    normalized_images = cast(images, float32)
    normalized_images /= 255
    return normalized_images, labels


def plot_history(history, metric='accuracy', withValidation=True, figsize=(8,4)):
    """
    Plots loss and chosen metric history from training.

    Arguments:
        history: output of `model.fit` method
        metric: name of metric chosen in model compilation
        withValidation: wheter to plot validation history
        figsize: size of generated figure

    Returns:
        f: figure's handle
        ax: axes' handles
        
    Example:
        `plot_history(history)`
        `plt.show()`
    """
    import matplotlib.pyplot as plt

    acc = history.history[metric]
    loss = history.history['loss']
    epochs_range = range(1, len(acc) + 1)

    title = 'Training' + (' and validation' if withValidation else ' ')
    
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax1.plot(epochs_range, acc, label='Training')
    ax1.set_title(f'{title} {metric}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(metric.capitalize())

    ax2.plot(epochs_range, loss, label='Training loss')   
    ax2.set_title(f'{title} loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    if withValidation:
        val_acc = history.history['val_'+metric]
        val_loss = history.history['val_loss']

        ax1.plot(epochs_range, val_acc, label='Validation')
        ax1.legend(loc='lower right')
        
        ax2.plot(epochs_range, val_loss, label='Validation')
        ax2.legend(loc='upper right')

    return f, [ax1, ax2]


def plot_prediction(i, predictions, images, labels, class_names, figsize=(10,4)):
    """
    Plots image to predict, prediction, its true label, and probability distribution generated by model.

    Arguments:
        i: index of image in a given batch
        predictions: output of `model.predict` method
        images: batch of images that were given to prediction
        labels: corresponding batch of labels
        class_names: list of classes' names
        figsize: size of figure

    Returns:
        f: figure's handle
        axes: axes'handlers

    Example:
        `plot_prediction(0, predictions, images, labels, class_names)`
    """
    import matplotlib.pyplot as plt
    from numpy import argmax, max

    f, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    predictions_array, true_label, img = predictions[i], labels[i], images[i]
    
    axes[0].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])    
    axes[0].imshow(img)

    predicted_label = argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    axes[1].grid(False)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    thisplot = axes[1].bar(range(len(class_names)), predictions_array, color="#777777")
    axes[1].set_ylim([0, 1]) 
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

    plt.suptitle("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*max(predictions_array),
                                    class_names[true_label]))
    
    return f, axes

def confusion_matrix(predictions, labels, no_classes):
    """
    Constructs the confustion matrix based on provided predictions and target labels.
    Arguments:
        predictions: output of `model.predict` method
        labels: target labels of the dataset (as list of batches)
        no_classes: number of classes in the dataset
    Returns:
        mat: 2D numpy array where `mat[i, j]` is part of examples with label `i` predicted as label `j`
    """

    import numpy as np

    mat = np.zeros((no_classes, no_classes), np.float32)
    normalization = np.zeros(no_classes, np.int32)
    for pred_batch, labels_batch in zip(predictions, labels):
        for prediction, target in zip(pred_batch, labels_batch):
            normalization[target] += 1
            mat[target, np.argmax(prediction)] += 1

    for i in range(no_classes):
        mat[i, :] /= normalization[i]

    return mat

def plot_confusion_matrix(conf_matrix, class_labels=None, figsize=(10, 10)):
    """
    Plots the confusion matrix.
    Arguments:
        conf_matrix: 2D numpy array, output of `confusion_matrix` function
        class_labels: optional labels of classes for ticks (if None, it will show numbers)
        figsize: size of figure
    Returns:
        fig: handler of figure
        ax: handler of axes
    """

    import matplotlib.pyplot as plt
    N = conf_matrix.shape[0]
    MAX = conf_matrix.max().max()

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(conf_matrix)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Target')

    if class_labels != None:
        ax.set_xticks(N)
        ax.set_xticklabels(class_labels)
        ax.set_yticks(N)
        ax.set_yticklabels(class_labels)

    plt.colorbar()

    return fig, ax
    
