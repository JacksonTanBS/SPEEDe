#!/usr/bin/env python

'''
Copyright © 2024 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.   All Rights Reserved.

Disclaimer:
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from calendar import monthrange
from imerg_input_generator import imerg_input_generator

modelName = 'SPEEDe.IMERG.v1.0.2'

modelpath = '/gpm3/btan1/IMERG_ML/SPEEDe/IMERG/'
os.makedirs(modelpath, exist_ok = True)

# Define the model settings.

nsample = 500       # no. of subsampled training/validation data
batch_size = 10     # no. of files to read for each batch
val_frac = 0.2      # frac. of samples to set aside as validation data
nepoch = 100        # max. number of training epochs
verbose = 1         # verbose level for model fit callbacks
es_patience = 5     # patience in early stopping
es_delta = 0.00001  # min_delta in early stopping

# Define the grid settings.

nlon, nlat = 3600, 1800
lonedges = np.linspace(-180, 180, nlon + 1)
latedges = np.linspace(-90, 90, nlat + 1)
lons = lonedges[1:] - 0.5 * np.diff(lonedges)
lats = latedges[1:] - 0.5 * np.diff(latedges)

fv = -9999.9

# Get the filenames for training/validation using the list of known good dates.

files = []

with open('/gpm3/btan1/IMERG_ML/SPEEDe/IMERG/fields.good.231013.txt', 'r') as f:
    for orbit in f.readlines():

        year, month, day, hhr = [int(ii) for ii in orbit.split()]
        t = datetime(year, month, day, hhr // 2, (hhr % 2) * 30)
        t0 = t.strftime('%H%M%S')
        t1 = (t + timedelta(seconds = 1799)).strftime('%H%M%S')
        t2 = int((t - datetime(year, month, day)).total_seconds() / 60)

        files.append(f'/gpm3/data/IMERG/Final/HHR/V07A/' 
                     f'{year}/{month:02d}/{day:02d}/' 
                     f'3B-HHR.MS.MRG.3IMERG.{year}{month:02d}'
                     f'{day:02d}-S{t0}-E{t1}.{t2:04d}.V07A.HDF5')

# Split the filenames into training and validation.

np.random.default_rng(71).shuffle(files)
files = files[:nsample]

gen_trn = imerg_input_generator(files[int(nsample * val_frac):], batch_size)
gen_val = imerg_input_generator(files[:int(nsample * val_frac)], batch_size)

# Define the autoencoder model.

in_img = tf.keras.Input(shape = (nlon, nlat, 1))
nfilter1 = 64
nfilter2 = 32

x = tf.keras.layers.Conv2D(nfilter1, (5, 5), activation = 'relu', 
                           padding = 'same')(in_img)
x = tf.keras.layers.MaxPooling2D((5, 5), padding = 'same')(x)
x = tf.keras.layers.Conv2D(nfilter2, (3, 3), activation = 'relu', 
                           padding = 'same')(x)
x = tf.keras.layers.MaxPooling2D((3, 3), padding = 'same')(x)
x = tf.keras.layers.Conv2D(nfilter2, (2, 2), activation = 'relu', 
                           padding = 'same')(x)
x = tf.keras.layers.UpSampling2D((3, 3))(x)
x = tf.keras.layers.Conv2DTranspose(nfilter1, (3, 3), activation = 'relu', 
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