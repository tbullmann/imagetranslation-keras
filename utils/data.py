import glob
import random
import numpy as np
from skimage.io import imread
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, batch_size=1, dim=(32,32), X_channels=1, Y_channels=1,
                 shuffle=True, **params):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.X_channels = X_channels
        self.Y_channels = Y_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, pair_filenames):
        """ Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.X_channels))
        Y = np.empty((self.batch_size, *self.dim, self.Y_channels))

        # Load data
        for i, (X_filename, Y_filename) in enumerate(pair_filenames):
            X[i,] = image_to_tensor(X_filename)
            Y[i,] = image_to_tensor(Y_filename)

        return X, Y


def image_to_tensor(X_filename):
    # print('Reading %s' % X_filename)
    tensor = np.array(imread(X_filename))
    if len(tensor.shape) < 3:
        tensor = tensor[:, :, np.newaxis]
    return tensor / 255.


def partition_dataset(pairs_filename, shuffle=True, training=70, validation=30):

    if shuffle:
        random.shuffle(pairs_filename)

    total = training + validation
    pos = len(pairs_filename) * training // total

    train_data = pairs_filename[:pos]
    validation_data = pairs_filename[pos:]

    partition = {'train': train_data, 'validation': validation_data}

    return partition


def load_dataset(X_dir, Y_dir):
    # Ording filename by alphabet assuming corresponding filenames have them (trailing number) in each folder
    X_filenames = sorted(glob.glob(X_dir))
    Y_filenames = sorted(glob.glob(Y_dir))
    pairs_filename = list(zip(X_filenames, Y_filenames))

    return pairs_filename
