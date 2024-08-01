#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from ande_funcs import read_IMERG_hhr

# Preliminaries.

modelName = sys.argv[1]

ver = 'V07A'
var = 'precipitation'

fv = -9999.9

# Set up the directories.

modelpath = '/gpm3/btan1/IMERG_ML/AnDe/IMERG/'
outpath = f'/gpm3/btan1/IMERG_ML/AnDe/IMERG/{modelName}/'

os.makedirs(outpath, exist_ok = True)

# Load the model.

autoencoder = tf.keras.models.load_model('%s%s.h5' % (modelpath, modelName))

# Read the list of bad dates, marginal dates, and good dates.

with open('/gpm3/btan1/IMERG_ML/AnDe/orbits.bad.231013.txt', 'r') as f:
    orbits_bad = sorted([l.strip() for l in f.readlines()])

with open('/gpm3/btan1/IMERG_ML/AnDe/orbits.marginal.231013.txt', 'r') as f:
    orbits_marginal = sorted([l.strip() for l in f.readlines()])

with open('/gpm3/btan1/IMERG_ML/AnDe/orbits.good.231013.txt', 'r') as f:
    orbits_good = sorted([l.strip() for l in f.readlines()])

# randomly select only 200 good orbits to analyze
np.random.default_rng(71).shuffle(orbits_good)
orbits_good = sorted(orbits_good[:200])

# Get the AnDe scores for the bad and good dates.

scores_bad = []
for orbit in orbits_bad:
    year, month, day, hhr = [int(ii) for ii in orbit.split()]
    P0 = read_IMERG_hhr(year, month, day, hhr, ver = ver, var = var)
    P0[P0 == fv] = 0
    p0 = np.log10(P0 + 1)
    p1 = autoencoder.predict(p0, verbose = 0)[0, :, :, 0]
    scores_bad.append(np.mean((p0 - p1) ** 2) * 100)

scores_marginal = []
for orbit in orbits_marginal:
    year, month, day, hhr = [int(ii) for ii in orbit.split()]
    P0 = read_IMERG_hhr(year, month, day, hhr, ver = ver, var = var)
    P0[P0 == fv] = 0
    p0 = np.log10(P0 + 1)
    p1 = autoencoder.predict(p0, verbose = 0)[0, :, :, 0]
    scores_marginal.append(np.mean((p0 - p1) ** 2) * 100)

scores_good = []
for orbit in orbits_good:
    year, month, day, hhr = [int(ii) for ii in orbit.split()]
    P0 = read_IMERG_hhr(year, month, day, hhr, ver = ver, var = var)
    P0[P0 == fv] = 0
    p0 = np.log10(P0 + 1)
    p1 = autoencoder.predict(p0, verbose = 0)[0, :, :, 0]
    scores_good.append(np.mean((p0 - p1) ** 2) * 100)

# Analyze the good and bad scores

labels = ['bad cases', 'marginal cases', 'normal cases']
line_gmax = np.max(scores_good)
line_bmin = np.min(scores_bad)
line_bp10 = np.percentile(scores_bad, 10)

plt.figure()
plt.boxplot([scores_bad, scores_marginal, scores_good], 
            whis = [5, 95], sym = '.', labels = labels, zorder = 5)
plt.axhline(line_gmax, 0.15, 1, ls = '--', color = 'C0', zorder = 3)
plt.axhline(line_bmin, 0, 0.85, ls = '--', color = 'C1', zorder = 2)
plt.axhline(line_bp10, 0, 0.85, ls = '--', color = 'C3', zorder = 1)
plt.text(3.15, line_gmax, f'max = {line_gmax:5.3f}',
         color = 'C0', ha = 'left', va = 'bottom')
plt.text(0.85, line_bmin, f'min = {line_bmin:5.3f}',
         color = 'C1', ha = 'right', va = 'top')
plt.text(0.85, line_bp10, f'p10 = {line_bp10:5.3f}',
         color = 'C3', ha = 'right', va = 'bottom')
plt.ylabel(f'{modelName} score')
plt.grid()
plt.savefig(f'{outpath}boxplot_scores.png', dpi = 150, bbox_inches = 'tight')
plt.close()

# Write the percentiles to file.

with open(f'{outpath}thresholds.bad.txt', 'w') as f:
    for p in range(101):
        f.write(f'{p:3d} {np.percentile(scores_bad, p):6.4f}\n')

with open(f'{outpath}thresholds.good.txt', 'w') as f:
    for p in range(101):
        f.write(f'{p:3d} {np.percentile(scores_good, p):6.4f}\n')