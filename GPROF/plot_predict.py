#!/usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import h5py
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
from matplotlib import patheffects
from speede_funcs import read_IMERG_hhr, fix_GPROF_shape

# Preliminaries.

filename = sys.argv[1]

sensor = filename.split('/')[-1].split('.')[2]
modelName = f'SPEEDe.GPROF-{sensor}.v1.0.0'
plot_prediction = False

dpi = 100    # 100 = 1152x648, 166.667 = 1920x1080, 333.334 = 3840x2160
useHighResMap = True    # use high res. background map
useStaticMap = True    # use fixed background map as opposed to monthly map

plt.rcParams['font.sans-serif'] = ['Helvetica', 'TeX Gyre Heros', 'DejaVu Sans']

fv = -9999.9

# Set up the directories.

mappath = '/gpm3/btan1/plots/bkgnd_maps/'
modelpath = '/gpm3/btan1/IMERG_ML/SPEEDe/GPROF/'
outpath = f'/gpm3/btan1/IMERG_ML/SPEEDe/GPROF/{modelName}/'

os.makedirs(outpath, exist_ok = True)

# Load the model.

autoencoder = tf.keras.models.load_model('%s%s.h5' % (modelpath, modelName))
_, ntrack, nswath, _ = autoencoder.layers[0].input_shape[0]

# Read the GPROF data and pre-process it.

with h5py.File(filename, 'r') as f:
    P0 = fix_GPROF_shape(f['S1/surfacePrecipitation'][:], ntrack, nswath)
    lons_gprof = fix_GPROF_shape(f['S1/Longitude'][:], ntrack, nswath, fv)
    lats_gprof = fix_GPROF_shape(f['S1/Latitude'][:], ntrack, nswath, fv)

lons_gprof[lons_gprof == -9999] = fv
lats_gprof[lats_gprof == -9999] = fv

# Apply the SPEEDe model.

isMissing = P0 < 0
P0[isMissing] = 0
p0 = np.log10(P0 + 1)
p1 = autoencoder.predict(p0[None], verbose = 0)[0, :, :, 0]
P1 = 10 ** p1 - 1
P0[isMissing] = fv
P1[isMissing] = fv
mse = np.mean((p0[~isMissing] - p1[~isMissing]) ** 2) * 100
logmean = np.mean(p0[~isMissing])
pixelcount = np.sum(~isMissing)

# Grid the Level 2 data for plotting.

try:
    from grid_orbit_backward import grid_orbit_backward
except ModuleNotFoundError:
    import subprocess
    command = 'f2py -c -m {0} {0}.f90 --fcompiler=gnu95 --compiler=unix'
    subprocess.run(command.format('grid_orbit_backward'), shell = True)
    from grid_orbit_backward import grid_orbit_backward

lats = read_IMERG_hhr(2015, 1, 1, 0, var = 'lat')
lons = read_IMERG_hhr(2015, 1, 1, 0, var = 'lon')
nlon, nlat = len(lons), len(lats)
lonedges = np.linspace(-180, 180, nlon + 1)
latedges = np.linspace(-90, 90, nlat + 1)

P0_grid = grid_orbit_backward(lons_gprof, lats_gprof, P0, lons, lats, 20)
P1_grid = grid_orbit_backward(lons_gprof, lats_gprof, P1, lons, lats, 20)

# Determine the color thresholds.

SPEEDe_col = '#ffffff'

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
ax.pcolormesh(lonedges, latedges, P0_grid.T == fv,
              vmin = 0, vmax = 1, 
              cmap = ListedColormap(('#ffffff00', '#7f7f7f7f')),
              transform = ccrs.PlateCarree())

# draw the precip.
ax.pcolormesh(lons, lats, np.ma.masked_less(P0_grid, bounds[0]).T, 
              cmap = cmap1, norm = BoundaryNorm(bounds, cmap1.N),
              transform = ccrs.PlateCarree())

# write the SPEEDe score
txt = plt.figtext(0.6, 0.97, f'{modelName} score: {mse:7.5f}', 
                  color = SPEEDe_col, ha = 'left', va = 'center', 
                  fontsize = 14)
txt.set_path_effects([patheffects.Stroke(linewidth = 1, foreground = 'w'),
                      patheffects.Normal()])
plt.figtext(0.6, 0.93, f'log-mean: {logmean:7.5f}', 
            color = SPEEDe_col, ha = 'left', va = 'center', fontsize = 14)
plt.figtext(0.77, 0.93, f'pixel count: {pixelcount:6d}', 
            color = SPEEDe_col, ha = 'left', va = 'center', fontsize = 14)

# add filename
plt.figtext(0.05, 0.95, filename.split('/')[-1], 
            color = 'w', fontsize = 10, ha = 'left', va = 'center')

ax.set_global()
fname = f'{outpath}{filename.split("/")[-1]}.png'
plt.savefig(fname, dpi = dpi, facecolor = 'k', edgecolor = 'none')
plt.close()

if plot_prediction:

    os.makedirs(f'{outpath}predict/', exist_ok = True)

    plt.figure(figsize = (fig_w, fig_h))

    # draw map
    proj = ccrs.PlateCarree()
    ax = plt.axes([0, 0, 1, 0.88889], projection = proj)
    ax.imshow(bkgndmap, origin = 'upper', extent = (-180, 180, -90, 90),
              interpolation = 'none', transform = ccrs.PlateCarree())
    ax.set_aspect('equal')    # 'equal' or 'auto'
    ax.axis('off')

    # draw missing values
    ax.pcolormesh(lonedges, latedges, P1_grid.T == fv,
                  vmin = 0, vmax = 1, 
                  cmap = ListedColormap(('#ffffff00', '#7f7f7f7f')),
                  transform = ccrs.PlateCarree())

    # draw the precip.
    ax.pcolormesh(lons, lats, np.ma.masked_less(P1_grid, bounds[0]).T, 
                  cmap = cmap1, norm = BoundaryNorm(bounds, cmap1.N),
                  transform = ccrs.PlateCarree())

    # write the SPEEDe score
    txt = plt.figtext(0.6, 0.95, f'{modelName} score: {mse:7.5f}', 
                      color = SPEEDe_col, ha = 'left', va = 'center', 
                      fontsize = 14)
    txt.set_path_effects([patheffects.Stroke(linewidth = 1, foreground = 'w'),
                          patheffects.Normal()])
    plt.figtext(0.6, 0.93, f'log-mean: {logmean:7.5f}', 
                color = SPEEDe_col, ha = 'left', va = 'center', fontsize = 14)
    plt.figtext(0.77, 0.93, f'pixel count: {pixelcount:6d}', 
                color = SPEEDe_col, ha = 'left', va = 'center', fontsize = 14)

    # add filename
    plt.figtext(0.05, 0.95, filename.split('/')[-1], 
                color = 'w', fontsize = 10, ha = 'left', va = 'center')

    ax.set_global()
    fname = f'{outpath}predict/{filename.split("/")[-1]}.png'
    plt.savefig(fname, dpi = dpi, facecolor = 'k', edgecolor = 'none')
    plt.close()