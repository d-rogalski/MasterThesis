__name__ = 'Utility functions'

def get_dataset_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=False, shuffle_size=10000, seed=None):
    """
    Function for splitting complete dataset into train, validation and test partitions.

    ### Arguments
        - `ds` (TensorFlow dataset): dataset to split
        - `train_split`: part of dataset for training (float)
        - `validation split`: part of dataset for validation (float)
        - `test_split`: part of dataset for testing (float)
        - `shuffle`: whether to shuffle the data before splitting (bool)
        - `shuffle_size`: size of buffer for shuffling
        - `seed`: seed for shuffling

    ### Returns
        `train_ds`, `val_ds`, `test_ds`

    ### Example
        `train_ds, val_ds, test_ds = get_dataset_partitions(dataset, 0.7, 0.2, 0.1, seed=123)`
    """

    # Check if the splits sum sufficiently close to 1
    assert (train_split + test_split + val_split - 1) <= 1e-5
    
    # Shuffle the data set
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=seed)
    
    # Calculate sizes of partitions
    ds_size = len(ds)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    # Partition dataset
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


def normalize(images, labels):
    """
    Function normalizing batch of images in dataset from integer values [0:255] to float32 values [0:1]

    ### Arguments
        - `images`: images in batch
        - `labels`: corresponding labels

    ### Returns
        `normalized_images`, `labels`

    ### Example
        `dataset.map(normalize)`
    """
    from tensorflow import cast, float32

    normalized_images = cast(images, float32) # Change type of images from int to float
    normalized_images /= 255 # Normalize to 0-1 values
    return normalized_images, labels


def plot_history(history, metric='accuracy', withValidation=True, figsize=(8,4)):
    """
    Plots loss and chosen metric history from training.

    ### Arguments
        - `history`: history element of `model.fit` method's output
        - `metric`: name of metric chosen in model compilation
        - `withValidation`: wheter to plot validation history
        - `figsize`: size of generated figure

    ### Returns
        - `fig`: figure's handle
        - `axes`: list of axes' handles
        
    ### Example
        `h = model.fit(...)`
        `plot_history(h.history)`
        `plt.show()`
    """
    import matplotlib.pyplot as plt

    # Extract training curves
    acc = history[metric]
    loss = history['loss']

    # Create epochs list
    epochs_range = range(1, len(acc) + 1)

    # Title for the plot
    title = 'Training' + (' and validation' if withValidation else ' ')
    
    # Creating figure
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Accuracy plot
    ax1.plot(epochs_range, acc, '.-', label='Training')
    ax1.set_title(f'{title} {metric}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(metric.capitalize())

    # Loss plot
    ax2.plot(epochs_range, loss, '.-', label='Training')   
    ax2.set_title(f'{title} loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    if withValidation:
        # Extracting validation curves
        val_acc = history['val_'+metric]
        val_loss = history['val_loss']

        # Accuracy plot
        ax1.plot(epochs_range, val_acc, '.-', label='Validation')
        ax1.legend(loc='lower right')

        # Loss plot
        ax2.plot(epochs_range, val_loss, '.-', label='Validation')
        ax2.legend(loc='upper right')

    return fig, [ax1, ax2]


def confusion_matrix(predictions, labels, no_classes):
    """
    Constructs the confustion matrix based on provided predictions and target labels.

    ### Arguments
        - `predictions`: output of `model.predict` method
        - `labels`: target labels of the dataset (as list of batches)
        - `no_classes`: number of classes in the dataset

    ### Returns
        `mat`: 2D numpy array where `mat[i, j]` is number of examples with label `i` predicted as label `j`
    """
    import numpy as np

    # Assert the lengths and batches are equal
    assert len(predictions) == len(labels)
    assert len(predictions[0]) == len(labels[0])

    # Initialise array
    mat = np.zeros((no_classes, no_classes), np.int32)

    # Loop over batches
    for pred_batch, labels_batch in zip(predictions, labels):
        # Loop over samples
        for prediction, target in zip(pred_batch, labels_batch):
            # Increment the value on the corresponding position
            mat[target, np.argmax(prediction)] += 1

    return mat


def plot_confusion_matrix(conf_matrix, normalize=True, class_labels=None, figsize=(10, 10)):
    """
    Plots the confusion matrix.

    ### Arguments
        - `conf_matrix`: 2D numpy array, output of `confusion_matrix` function
        - `normalize`: whether to display counts or percentages of the amount of images in given class
        - `class_labels`: optional labels of classes for ticks (if None, it will show numbers)
        - `figsize`: size of figure

    ### Returns
        - `fig`: handler of figure
        - `ax`: handler of axes
    """
    import matplotlib.pyplot as plt

    # Size of the matrix
    N = conf_matrix.shape[0]

    # Normalise the values row-wise
    if normalize:
        conf_matrix = conf_matrix.astype(float)
        for i in range(N):
            conf_matrix[i, :] = conf_matrix[i, :] / conf_matrix[i, :].sum()

    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Display the matrix and set labels
    i = ax.imshow(conf_matrix)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Target')

    # Display labels at ticks, if given
    if class_labels != None:
        ax.set_xticks(N)
        ax.set_xticklabels(class_labels)
        ax.set_yticks(N)
        ax.set_yticklabels(class_labels)

    # Display colorbar
    plt.colorbar(i)

    return fig, ax


def plot_predictions(X, y, distribution, labels, samples_dir, figsize=(8,4)):
    """
    Plots mosaic of tested image, reference images of top three predictions 
    (all with corresponding confidence levels) and the probability distribution of predictions.
    ### Arguments  
        - `X`: loaded image (data of an element from a batch of the dataset)
        - `y`: label index (label of an element from a batch of the dataset)
        - `labels`: list of all labels (strings)   
        - `samples_dir`: path to the directory containing the dataset
        - `figsize`: size of the figure (passed to pyplot)
    
    ### Returns
        - Handles of figure and axes `fig, axes` returned by the `pyplot.subplot_mosaic` function
    """
    no_classes = len(labels)
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    predictions = np.argsort(distribution)[-3::][::-1]
    
    species_true = labels[y].replace(" ","\n")
    species_pred = [(labels[p], labels[p].replace(" ","\n")) for p in predictions]

    fig, axes = plt.subplot_mosaic([['image', 'top1', 'distribution', 'distribution'],
                                   ['top2', 'top3', 'distribution', 'distribution']],
                                   figsize=figsize, layout='constrained')
    
    pal = sns.color_palette()

    clrs = [pal[0] for i in range(no_classes)]
    clrs[predictions[0]] = pal[3]
    clrs[y] = pal[2]

    axes['image'].imshow(X)
    axes['image'].set_title(f"Tested image:\n{species_true}")
    axes['image'].set_xticks([])
    axes['image'].set_yticks([])
    axes['image'].set_xlabel(f'Conf.: {distribution[y]*100:.2f}%', fontdict={'size':14})
    for i in range(3):
        axes[f'top{i+1}'].imshow(plt.imread(f'{samples_dir}/{species_pred[i][0]}/0002.jpg'))
        axes[f'top{i+1}'].set_title(f'Top {i+1} prediction:\n{species_pred[i][1]}')
        axes[f'top{i+1}'].set_xticks([])
        axes[f'top{i+1}'].set_yticks([])
        axes[f'top{i+1}'].set_xlabel(f'Conf.: {distribution[predictions[i]]*100:.2f}%', fontdict={'size':14})

    sns.set(font_scale=1.5, style='whitegrid')
    axes['distribution'].barh(range(1, no_classes + 1), distribution * 100, color=clrs, edgecolor=clrs)
    axes['distribution'].set_ylabel('Label index')
    axes['distribution'].set_xlabel('Confidence level (%)')
    axes['distribution'].set_yticks(range(0,75,10))

    return fig, axes


def get_precisions(matrix):
    """
    Computes precisions from the confusion matrix.

    ### Arguments
        - `matrix`: 2D array made by `confusion_matrix` function

    ### Returns
        - `precisions`: a numpy array of precisions calculated for each class with pandas.NA for
        classes not included in the matrix
    """
    import numpy as np
    from pandas import NA

    # Cast the matrix to numpy array
    matrix = np.array(matrix)

    # Replace NaN values (for matrices calculated on the Nature DS) with 0
    matrix[np.isnan(matrix)] = 0.0

    # Empty array for precisions
    precisions = []

    # Loop over matrix diagonal
    for i in range(matrix.shape[0]):
        # If the matrix includes i-th class
        if (matrix[i, i] != 0):
            precisions.append(matrix[i, i] / matrix[:, i].sum()) # Caculate precision
        else:
            precisions.append(NA) # Set with pandas' NA (as it is later processed in DataFrames)

    return np.array(precisions)


def get_recalls(matrix):
    """
    Computes recalls from the confusion matrix.

    ### Arguments
        - `matrix`: 2D array made by `confusion_matrix` function

    ### Returns
        - `recalls`: a numpy array of recalls calculated for each class
    """
    import numpy as np
    from pandas import NA

    # Cast the matrix to numpy array
    matrix = np.array(matrix)

    # Replace NaN values (for matrices calculated on the Nature DS) with 0
    matrix[np.isnan(matrix)] = 0.0

    # Empty array for recalls
    recalls = []
    
    # Loop over matrix diagonal
    for i in range(matrix.shape[0]):
        # If the matrix includes i-th class
        if (matrix[i, i] != 0):
            recalls.append(matrix[i, i] / matrix[i, :].sum()) # Calculate recall
        else:
            recalls.append(NA) # Set with pandas' NA (as it is later processed in DataFrames)

    return np.array(recalls)


def f1_score(precision, recall):
    """
    Computes F1 score based on the precision and recall. Works both for single values and numpy arrays

    ### Arguments
        - `precision`: a single value or numpy array of values
        - `recall`: a single vlaue or numpy array of values (of the same dimensions)

    ### Returns
        A F1 scores of corresponding to the arguments dimensionality
    """
    
    return 2 * precision * recall / (precision + recall)    


def save_json(path, d):
    '''
    Save dictionary into json file on given path.

    ### Arguments
        - `path`: path to the file that should be created (name included)
        - `d`: dictionary with info to save
    '''
    from json import dump
    with open(path, 'w') as src:
        dump(d, src)


def load_json(path):
    '''
    Load dictionary from json file on a given path.

    ### Arguments
        - `path`: path to the file that should be loaded

    ### Returns
        - `d`: loaded dictionary
    '''
    from json import load
    with open(path, 'r') as src:
        d = load(src)
        return d
    

def transform_image(image, transforms):
    '''
    Transforms an image using the albumentations transformation.

    ### Arguments
        - `image`: image to transform (uint8 or float)
        - `transforms`: single or composed albumentations

    ### Returns
        `aug_img`: image after transformation

    ### Example
        `new_img = transform_image(img, albumentations.GaussNoise())`
    '''

    aug_data = transforms(image=image)
    aug_img = aug_data['image']
    return aug_img


def prepare_mapping(transforms):
    '''
    Prepares mapping function for mapping a tensorflow dataset from albumentations transformation. 
    Requires using `set_shapes` mapping afterwards.
    
    ### Arguments
        - `transforms`: single or composed albumentations
    
    ### Returns
        - `mapping`: function transforming dataset elements

    ### Example
        - `new_dataset = dataset.map(prepare_mapping(albumentations.GaussNoise()))`
    '''
    from tensorflow import numpy_function, float32

    # Augmentation function with predefined transformations
    def aug_fn(image):
        return transform_image(image, transforms)
    
    # Mapping function
    def mapping(image, label):
        aug_img = numpy_function(func=aug_fn, inp=[image], Tout=float32)
        return aug_img, label
    
    return mapping


def set_shapes(img_shape=(224, 224, 3)):
    '''
    Mapping function for tensorflow dataset setting new shapes of images. Required after performing
    any albumentations for proper work with models.

    Arguments:
        `img_shape`: original shape of images (befeore albumentations)

    Example:
        `new_dataset = dataset.map(set_shapes())`
    '''
    # Restoring the shapes
    def inner(img, label):
        img.set_shape(img_shape)
        label.set_shape([])
        return img, label
    
    return inner


def plot_quality_test(fpath, name, xlabel, xticks='auto', scale='linear', figsize=(10,4), legend='12'):
    """
    Plots the results of testing the influence of image processing on the model stored in csv file.
    ### Arguments
        - `fpath`: path to the csv file with results
        - `name`: name of the tested transformation
        - `xlabel`: string to display as label of X axis of the plots
        - `xticks`: list of ticks on X axis, based on DataFrame if 'preset' or auto if 'auto'
        - `scale`: scale of X axis ('log' or 'linear')
        - `figisze`: size of figure passed to pyplot    
        - `legend`: in which plots place the legend - '1' for first, '2' for second, '12' for both

    ### Returns:
    Handles of figure and axes `fig, axes` returned by the `pyplot.subplots` function
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Loading the DataFrame
    df = pd.read_csv(fpath)
    # Renaming the models
    df['model'].replace({'InceptionV3':'Inception', 'ResNet50v2': 'ResNet50'}, inplace=True)
    
    # Creating figuer
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot accuracy
    sns.lineplot(
        data=df, 
        x=name, 
        y='accuracy', 
        hue='model',
        style='model',
        dashes=False,
        markers=True,
        markersize=7, 
        errorbar=None,
        linewidth=2, 
        ax=axes[0])

    # Plot loss
    sns.lineplot(
        data=df, 
        x=name,
        y='loss',
        hue='model',
        style='model',
        dashes=False,
        markers=True,
        markersize=7, 
        errorbar=None,
        linewidth=2, 
        ax=axes[1])

    # Setting labels
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Accuracy')

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Loss')

    # Placing legend
    if '1' in legend:
        axes[0].legend(title='Model', fontsize='12')
    else: axes[0].legend([],[],frameon=False)
    if '2' in legend:
        axes[1].legend(title='Model',fontsize='12')
    else:
        axes[1].legend([],[],frameon=False)

    # Scaling x axis
    axes[0].set_xscale(scale)
    axes[1].set_xscale(scale)

    # Settig ticks of x axis
    if xticks != 'auto':
        if xticks == 'preset':
            xticks = df[name].unique()[1::4]
        axes[0].set_xticks(xticks)
        axes[1].set_xticks(xticks)

    return fig, axes


def plot_transformations(image, transformations, titles):
    """
    Plots images transformed with given transformations and titles them accordingly in horizontal subplots.
    ### Arguments
        - `image`: a 2D array than can be displayed using pyplot
        - `transformations`: a list of Albumentations transformations to apply to the image
        - `titles`: a list of titles for each subplot / transformation
    ### Returns
        Handles of figure and axes `fig, axes` returned by the `pyplot.subplots` function.
    """
    import albumentations as A
    import matplotlib.pyplot as plt

    # Asserting matching lenghts of lists
    assert len(transformations) == len(titles)

    # Changing the image to floating-point type
    img = A.to_float(image)

    # Creating figure
    fig, axes = plt.subplots(1, len(transformations), figsize=(len(transformations)*2, 2.5))
    fig.tight_layout(pad=0) # Decreasing spacing between subplots

    i = 0
    # Loop over transformations/axes/titles
    for i in range(len(transformations)):
        new_img = transform_image(img, transformations[i])
        axes[i].imshow(new_img)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(titles[i])

    return fig, axes