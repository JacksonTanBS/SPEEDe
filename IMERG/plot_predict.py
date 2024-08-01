#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from datetime import datetime
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
from matplotlib import patheffects
from ande_funcs import read_IMERG_hhr, reduce_imerg

# Preliminaries.

year, month, day, hhr = [int(ii) for ii in sys.argv[1 : 5]]
modelName = sys.argv[5]
try:
    plot_prediction = sys.argv[6].lower() == 'true'
except IndexError:
    plot_prediction = False

ver = 'V07A'
var = 'precipitation'
dpi = 100    # 100 = 1152x648, 166.667 = 1920x1080, 333.334 = 3840x2160
useHighResMap = True    # use high res. background map
useStaticMap = True    # use fixed background map as opposed to monthly map

plt.rcParams['font.sans-serif'] = ['Helvetica', 'TeX Gyre Heros', 'DejaVu Sans']

# Set up the directories.

mappath = '../bkgnd_maps/'
modelpath = './saved_model/'
outpath = './'

os.makedirs(outpath, exist_ok = True)

# Set up the grid.

lats = read_IMERG_hhr(2015, 1, 1, 0, ver = ver, var = 'lat')
lons = read_IMERG_hhr(2015, 1, 1, 0, ver = ver, var = 'lon')

nlon, nlat = len(lons), len(lats)
lonedges = np.linspace(-180, 180, nlon + 1)
latedges = np.linspace(-90, 90, nlat + 1)

fv = -9999.9

# Get the precipitation field and apply the AnDe model.

autoencoder = tf.keras.models.load_model('%s%s.h5' % (modelpath, modelName))

P0 = read_IMERG_hhr(year, month, day, hhr, ver = ver, var = var)[0]
P0[P0 == fv] = 0
p0 = np.log10(P0 + 1)
p1 = autoencoder.predict(p0[None], verbose = 0)[0, :, :, 0]
P1 = 10 ** p1 - 1
mse = np.mean((p0 - p1) ** 2) * 100

# Determine the color thresholds.

try:
    with open(f'{outpath}thresholds.bad.txt', 'r') as f:
        thresholds_bad = {int(l.split()[0]): float(l.split()[1]) 
                          for l in f.readlines()}
    with open(f'{outpath}thresholds.good.txt', 'r') as f:
        thresholds_good = {int(l.split()[0]): float(l.split()[1]) 
                           for l in f.readlines()}
    threshold1 = thresholds_good[100]
    threshold2 = thresholds_bad[10]
    threshold3 = thresholds_bad[50]
except FileNotFoundError:
    threshold1 = 0.225
    threshold2 = 0.250
    threshold3 = 0.300

if   mse < threshold1:  # "all nominal"
    AnDe_col = '#ffffff'
elif mse < threshold2:  # "looks off"
    AnDe_col = '#ffffb0'
elif mse < threshold3:  # "likely wrong"
    AnDe_col = '#ff9300'
else:                   # "definitely bad"
    AnDe_col = '#ff0000'

# Set up the plot.

fig_w, fig_h = 11.52, 6.48

bounds = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10, 20, 50)

# "liquid" colors
cols1 = ('#008114', '#008b52', '#00b330', '#60d300', '#b4e700',
         '#fffb00', '#ffc400', '#ff9300', '#ff0000', '#c80000')
cmap1 = ListedColormap(cols1)
cmap1.set_under('#3a3d48')
cmap1.set_over('#910000')

# "frozen" colors
cols2 = ('#45ffff', '#49ddff', '#4dbbff', '#4f9aff', '#517aff', 
         '#525dff', '#5346ff', '#5a2fd9', '#7321bb', '#8c149c')
cmap2 = ListedColormap(cols2)
cmap2.set_under('#3a3d48')
cmap2.set_over('#8c149c')

if useHighResMap:
    mapres = '3x21600x10800'
    maprat = 6
else:
    mapres = '3x5400x2700'
    maprat = 1.5

if useStaticMap:
    bkgndmap = plt.imread('%sworld.200408.%s.png' % (mappath, mapres))
else:
    bkgndmap = plt.imread('%sworld.2004%02d.%s.png' % (mappath, month, mapres))

# Plot the data.

plt.figure(figsize = (fig_w, fig_h))

# draw map
proj = ccrs.PlateCarree()
ax = plt.axes([0, 0, 1, 0.88889], projection = proj)
ax.imshow(bkgndmap, origin = 'upper', extent = (-180, 180, -90, 90),
          interpolation = 'none', transform = ccrs.PlateCarree())
ax.set_aspect('equal')    # 'equal' or 'auto'
ax.axis('off')

# draw missing values
ax.pcolormesh(lonedges, latedges, P0.T == fv,
              cmap = ListedColormap(('#ffffff00', '#7f7f7f7f')),
              transform = ccrs.PlateCarree())

# draw the precip.
ax.pcolormesh(lons, lats, np.ma.masked_less(P0, bounds[0]).T, 
              cmap = cmap1, norm = BoundaryNorm(bounds, cmap1.N),
              transform = ccrs.PlateCarree())

# write the AnDe score
txt = plt.figtext(0.6, 0.95, f'{modelName} score: {mse:7.5f}', 
                  color = AnDe_col, ha = 'left', va = 'center', fontsize = 16)
txt.set_path_effects([patheffects.Stroke(linewidth = 1, foreground = 'w'),
                      patheffects.Normal()])

# add time
ts = datetime(year, month, day, hhr // 2, hhr % 2 * 30)
plt.figtext(0.05, 0.95, ts.strftime('%Y/%m/%d %H:%M:%S'), 
            color = 'w', fontsize = 14, ha = 'left', va = 'center')

ax.set_global()
fname = '%sorig.%s.png' % (outpath, ts.strftime('%Y%m%d.%H%M'),)
plt.savefig(fname, dpi = dpi, facecolor = 'k', edgecolor = 'none')

plt.close()

if plot_prediction:

    plt.figure(figsize = (fig_w, fig_h))

    # draw map
    proj = ccrs.PlateCarree()
    ax = plt.axes([0, 0, 1, 0.88889], projection = proj)
    ax.imshow(bkgndmap, origin = 'upper', extent = (-180, 180, -90, 90),
              interpolation = 'none', transform = ccrs.PlateCarree())
    ax.set_aspect('equal')    # 'equal' or 'auto'
    ax.axis('off')

    # draw missing values
    ax.pcolormesh(lonedges, latedges, P1.T == fv,
                  cmap = ListedColormap(('#ffffff00', '#7f7f7f7f')),
                  transform = ccrs.PlateCarree())

    # draw the precip.
    ax.pcolormesh(lons, lats, np.ma.masked_less(P1, bounds[0]).T, 
                  cmap = cmap1, norm = BoundaryNorm(bounds, cmap1.N),
                  transform = ccrs.PlateCarree())

    # write the AnDe score.

    txt = plt.figtext(0.6, 0.95, f'{modelName} score: {mse:7.5f}', 
                      color = AnDe_col, ha = 'left', va = 'center', fontsize = 16)
    txt.set_path_effects([patheffects.Stroke(linewidth = 1, foreground = 'w'),
                          patheffects.Normal()])

    # add time
    ts = datetime(year, month, day, hhr // 2, hhr % 2 * 30)
    plt.figtext(0.05, 0.95, ts.strftime('%Y/%m/%d %H:%M:%S'), 
                color = 'w', fontsize = 14, ha = 'left', va = 'center')

    ax.set_global()
    fname = '%spred.%s.png' % (outpath, ts.strftime('%Y%m%d.%H%M'),)
    plt.savefig(fname, dpi = dpi, facecolor = 'k', edgecolor = 'none')

    plt.close()
