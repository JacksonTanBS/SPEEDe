from tensorflow import keras
import h5py
import numpy as np

class gprof_input_generator(keras.utils.Sequence) :

    def __init__(self, filenames, batch_size, ntrack, nswath,
                 log_transform = True):
        self.filenames = filenames
        self.batch_size = batch_size
        self.ntrack = ntrack
        self.nswath = nswath
        self.log_transform = log_transform

    def __len__(self):
        return (np.ceil(len(self.filenames) / 
                float(self.batch_size))).astype('int')

    def __getitem__(self, idx):
        batch = self.filenames[idx * self.batch_size : 
                               (idx+1) * self.batch_size]

        x = np.zeros([self.batch_size, self.ntrack, self.nswath], 'f4')
        for n, filename in enumerate(batch):
            with h5py.File(filename, 'r') as f:
                p = f['S1/surfacePrecipitation'][:]
            p = p[:self.ntrack, :self.nswath]    # truncate if too long
            if p.shape[0] < self.ntrack:    # pad if too short (along-track)
                p = np.pad(p, ((0, self.ntrack - p.shape[0]), (0, 0)))
            if p.shape[1] < self.nswath:    # pad if too short (along-swath)
                p = np.pad(p, ((0, 0), (0, self.nswath - p.shape[1])))
            p[p < 0] = 0
            if self.log_transform:
                x[n] = np.log10(p + 1)
            else:
                x[n] = p

        return x, x