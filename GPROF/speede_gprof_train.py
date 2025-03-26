#!/usr/bin/env python

'''
Copyright © 2024 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.   All Rights Reserved.

Disclaimer:
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
'''

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
modelpath = '/gpm3/btan1/IMERG_ML/SPEEDe/GPROF/'
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

# JT: ntrack and nswath should be divisible by 10 due to model architecture,
# not sure what happens if it is not (model can run, but...???)

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

# Get the filenames for training/validation using the list of known good dates.

inpath = '/gpm3/btan1/data_repository/GPROF/V07A/2A-CLIM/'

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