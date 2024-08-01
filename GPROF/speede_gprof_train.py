#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
from calendar import monthrange
from gprof_input_generator import gprof_input_generator
from glob import glob

sensor = sys.argv[1]

modelName = f'SPEEDe.GPROF-{sensor}.v1.0.0'
modelpath = './saved_model/'
os.makedirs(modelpath, exist_ok = True)

# Define the model settings.

nsample = 1000      # no. of subsampled training/validation data
batch_size = 50     # no. of files to read for each batch
val_frac = 0.2      # frac. of samples to set aside as validation data
nepoch = 100        # max. number of training epochs
verbose = 1         # verbose level for model fit callbacks
es_patience = 5     # patience in early stopping
es_delta = 0.00005  # min_delta in early stopping

# Define the grid settings and training year.

if   sensor == 'AMSRE':
    ntrack, nswath = 3950, 390    # 3956, 392
    year = 2007
elif sensor == 'SSMI':
    ntrack, nswath = 3210, 120    # 3218, 128
    year = 2007
elif sensor == 'AMSUB':
    ntrack, nswath = 2270, 90     # 2276, 90
    year = 2007
elif sensor == 'TMI':
    ntrack, nswath = 2920, 210    # 2919, 208
    year = 2007
elif sensor == 'SSMIS':
    ntrack, nswath = 3200, 180    # 3206, 180
    year = 2015
elif sensor == 'MHS':
    ntrack, nswath = 2280, 90     # 2280, 90
    year = 2015
elif sensor == 'GMI':
    ntrack, nswath = 2960, 220    # 2961, 221
    year = 2015
elif sensor == 'ATMS':
    ntrack, nswath = 2280, 90     # 2283, 96
    year = 2015
elif sensor == 'AMSR2':
    ntrack, nswath = 3950, 480    # 3956, 486
    year = 2015
else:
    sys.exit(f'Error: grid setting for {sensor} unspecified.')

# Get the filenames for training.

inpath = '/path/to/GPROF/files/'

files = []
for month in range(1, 13):
    for day in range(1, monthrange(year, month)[1] + 1):
        files += sorted(glob(f'{inpath}{year}/{month:02d}/{day:02d}/'
                             f'2A-CLIM.*.{sensor}.*.HDF5'))

if len(files) == 0:
    sys.exit(f'Error: no files for {sensor} found.')

# Split the filenames into training and validation.

np.random.default_rng(71).shuffle(files)
files = files[:nsample]

gen_trn = gprof_input_generator(files[int(nsample * val_frac):], batch_size, 
                                ntrack, nswath)
gen_val = gprof_input_generator(files[:int(nsample * val_frac)], batch_size, 
                                ntrack, nswath)

# Define the autoencoder model.

in_img = tf.keras.Input(shape = (ntrack, nswath, 1))
nfilter1 = 64
nfilter2 = 32

x = tf.keras.layers.Conv2D(nfilter1, (5, 5), activation = 'relu', 
                           padding = 'same')(in_img)
x = tf.keras.layers.MaxPooling2D((5, 5), padding = 'same')(x)
x = tf.keras.layers.Conv2D(nfilter2, (2, 2), activation = 'relu', 
                           padding = 'same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding = 'same')(x)
x = tf.keras.layers.Conv2D(nfilter2, (2, 2), activation = 'relu', 
                           padding = 'same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2DTranspose(nfilter1, (2, 2), activation = 'relu', 
                                    padding = 'same')(x)
x = tf.keras.layers.UpSampling2D((5, 5))(x)
out_img = tf.keras.layers.Conv2D(1, (5, 5), activation = 'relu', 
                                 padding = 'same')(x)

autoencoder = tf.keras.Model(in_img, out_img)
autoencoder.compile(optimizer = 'adam', loss = 'mse')

# Train the model.

es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', 
                                      verbose = verbose, 
                                      patience = es_patience,
                                      min_delta = es_delta)
mc = tf.keras.callbacks.ModelCheckpoint('%s%s.h5' % (modelpath, modelName), 
                                        monitor = 'val_loss', mode = 'min', 
                                        verbose = verbose, 
                                        save_best_only = True)

autoencoder.fit(gen_trn, epochs = nepoch, verbose = 0,
                validation_data = (gen_val),
                callbacks = [es, mc])