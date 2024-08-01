from tensorflow import keras
import h5py
import numpy as np

class imerg_input_generator(keras.utils.Sequence) :

    def __init__(self, filenames, batch_size, log_transform = True):
        self.filenames = filenames
        self.batch_size = batch_size
        self.log_transform = log_transform

    def __len__(self):
        return (np.ceil(len(self.filenames) / 
                float(self.batch_size))).astype('int')

    def __getitem__(self, idx):
        batch = self.filenames[idx * self.batch_size : 
                               (idx+1) * self.batch_size]

        x = np.zeros([self.batch_size, 3600, 1800], 'f4')
        for n, filename in enumerate(batch):
            with h5py.File(filename, 'r') as f:
                p = f['Grid/precipitation'][0]
                p[p < 0] = 0
                if self.log_transform:
                    x[n] = np.log10(p + 1)
                else:
                    x[n] = p

        return x, x