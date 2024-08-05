#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
from datetime import datetime, timedelta
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from funcs import *

option = sys.argv[1]

# Plot configurations

figtype = 'png'
scol = 3.503    # single column (89 mm)
dcol = 7.204    # double column (183 mm)
flpg = 9.724    # full page length (247 mm)
plt.rcParams['figure.figsize'] = (scol, 0.75 * scol)
plt.rcParams['font.size'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['font.sans-serif'] = ['TeX Gyre Heros', 'Helvetica',
                                   'DejaVu Sans']
plt.rcParams['savefig.dpi'] = 300

spl = ('(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)')

# Plots.

if option == 'snapshot_SPEEDe_badorbit':

    import tensorflow as tf
    import matplotlib.patches as patches
    import h5py
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs

    # IMERG part

    modelpath = '../IMERG/saved_model/'
    modelName = 'SPEEDe.IMERG.v1.0.2'
    fv = -9999.9

    year, month, day, hhr = 2016, 12, 18, 18
    autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')
    P1 = read_IMERG_hhr(year, month, day, hhr, var = 'precipitation')[0]
    P1[P1 == fv] = 0
    p1 = np.log10(P1 + 1)
    p2 = autoencoder.predict(p1[None], verbose = 0)[0, :, :, 0]
    P2 = 10 ** p2 - 1
    mse1 = np.mean((p1 - p2) ** 2) * 100

    # GPROF part

    filename = ('/path/to/GPROF/files/'
                '2A-CLIM.F16.SSMIS.GPROF2021v1.20161218-S082058-E100252.'
                '067944.V07A.HDF5')

    modelpath = '../GPROF/saved_model/'
    modelName = 'SPEEDe.GPROF-SSMIS.v1.0.0'

    autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')
    _, ntrack, nswath, _ = autoencoder.layers[0].input_shape[0]
    fv = -9999.9

    with h5py.File(filename, 'r') as f:
        P3 = fix_GPROF_shape(f['S1/surfacePrecipitation'][:], ntrack, nswath)
        lons_gprof = fix_GPROF_shape(f['S1/Longitude'][:], ntrack, nswath, fv)
        lats_gprof = fix_GPROF_shape(f['S1/Latitude'][:], ntrack, nswath, fv)
    lons_gprof[lons_gprof == -9999] = fv
    lats_gprof[lats_gprof == -9999] = fv

    isMissing = P3 < 0
    P3[isMissing] = 0
    p3 = np.log10(P3 + 1)
    p4 = autoencoder.predict(p3[None], verbose = 0)[0, :, :, 0]
    P4 = 10 ** p4 - 1
    P3[isMissing] = fv
    P4[isMissing] = fv
    mse2 = np.mean((p3[~isMissing] - p4[~isMissing]) ** 2) * 100

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

    P3_grid = grid_orbit_backward(lons_gprof, lats_gprof, P3, lons, lats, 20)
    P4_grid = grid_orbit_backward(lons_gprof, lats_gprof, P4, lons, lats, 20)

    # plotting part

    bounds = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2)
    cmap = get_cmap(option = 'tropical', background = 'light')
    ts = datetime(year, month, day, hhr // 2, hhr % 2 * 30)
    nrow = 2

    fig = plt.figure(figsize = (dcol, dcol * (0.275 * nrow + 0.07)))
    subfigs = fig.subfigures(2, 1, height_ratios = [0.07, 0.275 * nrow])
    plt.subplots_adjust(left = 0.025, right = 0.975, 
                        bottom = 0.025 / nrow, top = 1 - 0.025 / nrow,
                        wspace = 0.05, hspace = 0.025 * nrow)

    ax0 = subfigs[0].add_axes([0.575, 0.65, 0.35, 0.15])
    cb = ColorbarBase(ax0, cmap = cmap,
                      norm = BoundaryNorm(bounds, cmap.N),
                      orientation = 'horizontal', ticks = bounds,
                      extend = 'both', format = '%g')
    cb.ax.tick_params(labelsize = 7) 
    cb.set_label('Precipitation Rate (mm / h)', fontsize = 7, labelpad = -0.8)

    ax1 = subfigs[1].add_subplot(nrow, 2, 1, projection = ccrs.PlateCarree())
    ax1.set_title(f'(a) IMERG ({ts.strftime("%Y/%m/%d %H:%M:%S")})', 
                  loc = 'left', y = 0.97)
    plot_imerg_snapshot(ax1, P1, cmap, use_bkgndmap = False)
    ax1.coastlines(resolution = '110m', color = '0.5', lw = 0.375)
    rect = patches.Rectangle((-87, -10), 7, 20, lw = 0.5, 
                             edgecolor = 'k', facecolor = 'none',
                             rotation_point = 'center', zorder = 8)
    ax1.add_patch(rect)

    ax5 = subfigs[1].add_axes([0.2, 0.6, 0.05, 0.05 * 20 / 7],
                              projection = ccrs.PlateCarree())
    plot_imerg_snapshot(ax5, P1, cmap, use_bkgndmap = False)
    ax5.set_extent([-87, -80, -10, 10], crs = ccrs.PlateCarree())

    ax2 = subfigs[1].add_subplot(nrow, 2, 2, projection = ccrs.PlateCarree())
    ax2.set_title(f'(b) SPEEDe-IMERG (score: {mse1:5.3f})', 
                  loc = 'left', y = 0.97)
    plot_imerg_snapshot(ax2, P2, cmap, use_bkgndmap = False)
    ax2.coastlines(resolution = '110m', color = '0.5', lw = 0.375, zorder = 7)
    rect = patches.Rectangle((-87, -10), 7, 20, lw = 0.5, 
                             edgecolor = 'k', facecolor = 'none',
                             rotation_point = 'center', zorder = 8)
    ax2.add_patch(rect)

    ax6 = subfigs[1].add_axes([0.7, 0.6, 0.05, 0.05 * 20 / 7],
                              projection = ccrs.PlateCarree())
    plot_imerg_snapshot(ax6, P2, cmap, use_bkgndmap = False)
    ax6.set_extent([-87, -80, -10, 10], crs = ccrs.PlateCarree())

    ax3 = subfigs[1].add_subplot(nrow, 2, 3, projection = ccrs.PlateCarree())
    ax3.set_title(f'(c) F16 SSMIS orbit 067944', 
                  loc = 'left', y = 0.97)
    plot_imerg_snapshot(ax3, P3_grid, cmap, use_bkgndmap = False)
    ax3.coastlines(resolution = '110m', color = '0.5', lw = 0.375)

    ax4 = subfigs[1].add_subplot(nrow, 2, 4, projection = ccrs.PlateCarree())
    ax4.set_title(f'(d) SPEEDe-GPROF (score: {mse2:5.3f})', 
                  loc = 'left', y = 0.97)
    plot_imerg_snapshot(ax4, P4_grid, cmap, use_bkgndmap = False)
    ax4.coastlines(resolution = '110m', color = '0.5', lw = 0.375)

    plt.savefig(f'fig.{option}.png')
    plt.close()

elif option == 'snapshot_SPEEDe-IMERG_goodcases':

    import tensorflow as tf

    modelpath = '../IMERG/saved_model/'
    modelName = 'SPEEDe.IMERG.v1.0.2'
    autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')
    fv = -9999.9

    timestamps = (datetime(2014, 1, 7, 8, 30), 
                  datetime(2014, 4, 21, 0, 30), 
                  datetime(2014, 7, 30, 11, 0), 
                  datetime(2014, 11, 4, 21, 0))

    P1, P2, mse = {}, {}, {}

    for n, ts in enumerate(timestamps):

        p1 = read_IMERG_hhr(ts.year, ts.month, ts.day, 
                            ts.hour * 2 + ts.minute % 30, 
                            var = 'precipitation')[0]
        P1[n] = p1.copy()
        p1[p1 == fv] = 0
        p1 = np.log10(p1 + 1)
        p2 = autoencoder.predict(p1[None], verbose = 0)[0, :, :, 0]
        P2[n] = 10 ** p2 - 1
        mse[n] = np.mean((p1 - p2) ** 2) * 100

    bounds = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2)
    cmap = get_cmap(option = 'tropical', background = 'light')
    nrow = len(timestamps)

    fig = plt.figure(figsize = (dcol, dcol * (0.275 * nrow + 0.07)))
    subfigs = fig.subfigures(2, 1, height_ratios = [0.07, 0.275 * nrow])
    plt.subplots_adjust(left = 0.025, right = 0.975, 
                        bottom = 0.025 / nrow, top = 1 - 0.025 / nrow,
                        wspace = 0.05, hspace = 0.025 * nrow)

    ax0 = subfigs[0].add_axes([0.575, 0.65, 0.35, 0.15])
    cb = ColorbarBase(ax0, cmap = cmap,
                      norm = BoundaryNorm(bounds, cmap.N),
                      orientation = 'horizontal', ticks = bounds,
                      extend = 'both', format = '%g')
    cb.ax.tick_params(labelsize = 7) 
    cb.set_label('Precipitation Rate (mm / h)', fontsize = 7, labelpad = -0.8)

    for n, ts in enumerate(timestamps):
        ax1 = subfigs[1].add_subplot(nrow, 2, 2 * n + 1, 
                                     projection = ccrs.PlateCarree())
        ax1.set_title(f'{spl[2 * n]} IMERG ({ts.strftime("%Y/%m/%d %H:%M:%S")})', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax1, P1[n], cmap, use_bkgndmap = False)
        ax1.coastlines(resolution = '110m', color = '0.5', lw = 0.375)
        ax2 = subfigs[1].add_subplot(nrow, 2, 2 * n + 2, 
                                     projection = ccrs.PlateCarree())
        ax2.set_title(f'{spl[2 * n + 1]} SPEEDe-IMERG (score: {mse[n]:5.3f})', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax2, P2[n], cmap, use_bkgndmap = False)
        ax2.coastlines(resolution = '110m', color = '0.5', lw = 0.375)

    plt.savefig(f'fig.{option}.png')
    plt.close()

elif option == 'snapshot_SPEEDe-IMERG_marginalcases':

    import tensorflow as tf

    modelpath = '../IMERG/saved_model/'
    modelName = 'SPEEDe.IMERG.v1.0.2'
    autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')
    fv = -9999.9

    timestamps = (datetime(2004, 5, 27, 21, 0),
                  datetime(2006, 11, 11, 8, 0), 
                  datetime(2009, 9, 16, 18, 0), 
                  datetime(2022, 12, 8, 6, 0))

    P1, P2, mse = {}, {}, {}

    for n, ts in enumerate(timestamps):

        p1 = read_IMERG_hhr(ts.year, ts.month, ts.day, 
                            ts.hour * 2 + ts.minute % 30, 
                            var = 'precipitation')[0]
        P1[n] = p1.copy()
        p1[p1 == fv] = 0
        p1 = np.log10(p1 + 1)
        p2 = autoencoder.predict(p1[None], verbose = 0)[0, :, :, 0]
        P2[n] = 10 ** p2 - 1
        mse[n] = np.mean((p1 - p2) ** 2) * 100

    bounds = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2)
    cmap = get_cmap(option = 'tropical', background = 'light')
    nrow = len(timestamps)

    fig = plt.figure(figsize = (dcol, dcol * (0.275 * nrow + 0.07)))
    subfigs = fig.subfigures(2, 1, height_ratios = [0.07, 0.275 * nrow])
    plt.subplots_adjust(left = 0.025, right = 0.975, 
                        bottom = 0.025 / nrow, top = 1 - 0.025 / nrow,
                        wspace = 0.05, hspace = 0.025 * nrow)

    ax0 = subfigs[0].add_axes([0.575, 0.65, 0.35, 0.15])
    cb = ColorbarBase(ax0, cmap = cmap,
                      norm = BoundaryNorm(bounds, cmap.N),
                      orientation = 'horizontal', ticks = bounds,
                      extend = 'both', format = '%g')
    cb.ax.tick_params(labelsize = 7) 
    cb.set_label('Precipitation Rate (mm / h)', fontsize = 7, labelpad = -0.8)

    for n, ts in enumerate(timestamps):
        ax1 = subfigs[1].add_subplot(nrow, 2, 2 * n + 1, 
                                     projection = ccrs.PlateCarree())
        ax1.set_title(f'{spl[2 * n]} IMERG ({ts.strftime("%Y/%m/%d %H:%M:%S")})', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax1, P1[n], cmap, use_bkgndmap = False)
        ax1.coastlines(resolution = '110m', color = '0.5', lw = 0.375)
        ax2 = subfigs[1].add_subplot(nrow, 2, 2 * n + 2, 
                                     projection = ccrs.PlateCarree())
        ax2.set_title(f'{spl[2 * n + 1]} SPEEDe-IMERG (score: {mse[n]:5.3f})', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax2, P2[n], cmap, use_bkgndmap = False)
        ax2.coastlines(resolution = '110m', color = '0.5', lw = 0.375)

    plt.savefig(f'fig.{option}.png')
    plt.close()

elif option == 'scores_SPEEDe':

    import h5py
    import tensorflow as tf
    from calendar import monthrange
    from glob import glob

    # IMERG part

    modelpath = '../IMERG/saved_model/'
    modelName = 'SPEEDe.IMERG.v1.0.2'
    fv = -9999.9
    nsample = 200

    autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')

    with open('../IMERG/saved_model/fields.bad.231013.txt', 'r') as f:
        orbits_bad = sorted([l.strip() for l in f.readlines()])
    with open('../IMERG/saved_model/fields.marginal.231013.txt', 'r') as f:
        orbits_marginal = sorted([l.strip() for l in f.readlines()])
    with open('../IMERG/saved_model/fields.good.231013.txt', 'r') as f:
        orbits_good = sorted([l.strip() for l in f.readlines()])
        np.random.default_rng(71).shuffle(orbits_good)
        orbits_good = sorted(orbits_good[:nsample])

    scores_bad = []
    for orbit in orbits_bad:
        year, month, day, hhr = [int(ii) for ii in orbit.split()]
        P0 = read_IMERG_hhr(year, month, day, hhr, var = 'precipitation')
        P0[P0 == fv] = 0
        p0 = np.log10(P0 + 1)
        p1 = autoencoder.predict(p0, verbose = 0)[0, :, :, 0]
        scores_bad.append(np.mean((p0 - p1) ** 2) * 100)

    scores_marginal = []
    for orbit in orbits_marginal:
        year, month, day, hhr = [int(ii) for ii in orbit.split()]
        P0 = read_IMERG_hhr(year, month, day, hhr, var = 'precipitation')
        P0[P0 == fv] = 0
        p0 = np.log10(P0 + 1)
        p1 = autoencoder.predict(p0, verbose = 0)[0, :, :, 0]
        scores_marginal.append(np.mean((p0 - p1) ** 2) * 100)

    scores_good = []
    for orbit in orbits_good:
        year, month, day, hhr = [int(ii) for ii in orbit.split()]
        P0 = read_IMERG_hhr(year, month, day, hhr, var = 'precipitation')
        P0[P0 == fv] = 0
        p0 = np.log10(P0 + 1)
        p1 = autoencoder.predict(p0, verbose = 0)[0, :, :, 0]
        scores_good.append(np.mean((p0 - p1) ** 2) * 100)

    # GPROF part

    sensors = ('TMI', 'AMSR2', 'SSMI', 'SSMIS', 'AMSUB', 'MHS', 
               'GMI', 'ATMS', 'AMSRE')
    sensors = ('AMSR2', 'SSMIS', 'MHS', 'GMI', 'ATMS')
    years = {'SSMIS': 2015, 'MHS': 2015, 'GMI': 2015, 'ATMS': 2015, 
             'AMSR2': 2015, 'AMSRE': 2007, 'AMSUB': 2007, 'SSMI': 2007, 
             'TMI': 2007}
    gprofpath = '/path/to/GPROF/files/'
    modelpath = '../GPROF/saved_model/'

    scores_gprof = {sensor: [] for sensor in sensors}

    for sensor in sensors:

        modelName = f'SPEEDe.GPROF-{sensor}.v1.0.0'

        autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')
        _, ntrack, nswath, _ = autoencoder.layers[0].input_shape[0]
        fv = -9999.9

        files = []
        for month in range(1, 13):
            for day in range(1, monthrange(years[sensor], month)[1] + 1):
                files += sorted(glob(f'{gprofpath}{years[sensor]:02d}/'
                                     f'{month:02d}/{day:02d}/'
                                     f'2A-CLIM.*.{sensor}.*.HDF5'))
        np.random.default_rng(71).shuffle(files)
        files = sorted(files[:nsample])

        for file in files:
            with h5py.File(file, 'r') as f:
                P0 = fix_GPROF_shape(f['S1/surfacePrecipitation'][:], 
                                     ntrack, nswath)

            isMissing = P0 < 0
            if np.all(isMissing): continue    # skip if array is empty
            P0[isMissing] = 0
            p0 = np.log10(P0 + 1)
            p1 = autoencoder.predict(p0[None], verbose = 0)[0, :, :, 0]
            scores_gprof[sensor].append(np.mean((p0[~isMissing] - 
                                                 p1[~isMissing]) ** 2) * 100)

    # plotting part

    labels = ['bad cases', 'marginal cases', 'good cases']

    plt.figure(figsize = (dcol, 0.75 * scol))
    plt.subplots_adjust(wspace = 0.25)

    plt.subplot(121)
    plt.title('(a) SPEEDe-IMERG', loc = 'left', y = 0.97)
    plt.boxplot([scores_bad, scores_marginal, scores_good], 
                whis = [5, 95], sym = '.', labels = labels)
    plt.ylabel(f'score')
    plt.grid()

    plt.subplot(122)
    plt.title('(b) SPEEDe-GPROF', loc = 'left', y = 0.97)
    plt.boxplot([scores_gprof[sensor] for sensor in sensors], 
                whis = [5, 95], sym = '.', labels = sensors)
    plt.ylabel(f'score')
    plt.grid()
    plt.savefig(f'fig.{option}.png', bbox_inches = 'tight')
    plt.close()

elif option == 'scores_histogram':

    from calendar import monthrange

    nlon, nlat = 3600, 1800
    fv = -9999.9

    bins = (0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 200)

    C_ref = np.zeros([len(bins) - 1], 'f4')
    year = 2014
    for month in range(1, 13):
        for day in range(1, monthrange(year, month)[1] + 1):
            for hhr in range(48):
                P = read_IMERG_hhr(year, month, day, hhr)[0]
                C_ref += np.histogram(P, bins)[0]
    C_ref = C_ref / np.sum(C_ref)

    with open('../IMERG/saved_model/fields.bad.231013.txt', 'r') as f:
        orbits_bad = sorted([l.strip() for l in f.readlines()])
    with open('../IMERG/saved_model/fields.marginal.231013.txt', 'r') as f:
        orbits_marginal = sorted([l.strip() for l in f.readlines()])
    with open('../IMERG/saved_model/fields.good.231013.txt', 'r') as f:
        orbits_good = sorted([l.strip() for l in f.readlines()])
        np.random.default_rng(71).shuffle(orbits_good)
        orbits_good = sorted(orbits_good[:200])

    scores_good = []
    for orbit in orbits_good:
        year, month, day, hhr = [int(ii) for ii in orbit.split()]
        P = read_IMERG_hhr(year, month, day, hhr)
        P[P == fv] = 0
        P = np.mean(P.reshape(nlon // 10, 10, nlat // 10, 10), (1, 3))
        C = np.histogram(P, bins)[0]
        C = C / np.sum(C)
        scores_good.append(np.mean((C - C_ref) ** 2) * 1000)

    scores_bad = []
    for orbit in orbits_bad:
        year, month, day, hhr = [int(ii) for ii in orbit.split()]
        P = read_IMERG_hhr(year, month, day, hhr)
        P[P == fv] = 0
        P = np.mean(P.reshape(nlon // 10, 10, nlat // 10, 10), (1, 3))
        C = np.histogram(P, bins)[0]
        C = C / np.sum(C)
        scores_bad.append(np.mean((C - C_ref) ** 2) * 1000)

    scores_marginal = []
    for orbit in orbits_marginal:
        year, month, day, hhr = [int(ii) for ii in orbit.split()]
        P = read_IMERG_hhr(year, month, day, hhr)
        P[P == fv] = 0
        P = np.mean(P.reshape(nlon // 10, 10, nlat // 10, 10), (1, 3))
        C = np.histogram(P, bins)[0]
        C = C / np.sum(C)
        scores_marginal.append(np.mean((C - C_ref) ** 2) * 1000)

    labels = ['bad cases', 'marginal cases', 'good cases']

    plt.figure()
    plt.boxplot([scores_bad, scores_marginal, scores_good], 
                whis = [5, 95], sym = '.', labels = labels)
    plt.ylabel(f'MSE in histogram counts (× 0.001)')
    plt.grid()
    plt.savefig(f'fig.{option}.png', bbox_inches = 'tight')
    plt.close()

elif option == 'snapshot_SPEEDe-GPROF_goodorbits':

    import h5py
    import tensorflow as tf
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs

    modelpath = '../GPROF/saved_model/'
    filenames = ('/path/to/GPROF/files/'
                 '2A-CLIM.F16.SSMIS.GPROF2021v1.20150218-S010630-E024824.'
                 '058487.V07A.HDF5', 
                 '/path/to/GPROF/files/'
                 '2A-CLIM.METOPB.MHS.GPROF2021v1.20150714-S045849-E064010.'
                 '014626.V07A.HDF5', 
                 '/path/to/GPROF/files/'
                 '2A-CLIM.NPP.ATMS.GPROF2021v1.20151003-S212205-E230334.'
                 '020380.V07A.HDF5',
                 '/path/to/GPROF/files/'
                 '2A-CLIM.GCOMW1.AMSR2.GPROF2021v1.20151223-S030241-E044133.'
                 '019144.V07A.HDF5')
    modelNames = ('SPEEDe.GPROF-SSMIS.v1.0.0', 
                  'SPEEDe.GPROF-MHS.v1.0.0', 
                  'SPEEDe.GPROF-ATMS.v1.0.0',
                  'SPEEDe.GPROF-AMSR2.v1.0.0')
    labels = ('F16 SSMIS orbit 058487',
              'MetOp-B MHS orbit 014626',
              'SNPP ATMS orbit 020380',
              'GCOM-W1 AMSR2 orbit 019144')

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

    P1_grid, P2_grid, mse = {}, {}, {}

    for n, (filename, modelName) in enumerate(zip(filenames, modelNames)):

        autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')
        _, ntrack, nswath, _ = autoencoder.layers[0].input_shape[0]
        fv = -9999.9

        with h5py.File(filename, 'r') as f:
            P1 = fix_GPROF_shape(f['S1/surfacePrecipitation'][:], ntrack, nswath)
            lons_gprof = fix_GPROF_shape(f['S1/Longitude'][:], ntrack, nswath, fv)
            lats_gprof = fix_GPROF_shape(f['S1/Latitude'][:], ntrack, nswath, fv)
        lons_gprof[lons_gprof == -9999] = fv
        lats_gprof[lats_gprof == -9999] = fv

        isMissing = P1 < 0
        P1[isMissing] = 0
        p1 = np.log10(P1 + 1)
        p2 = autoencoder.predict(p1[None], verbose = 0)[0, :, :, 0]
        P2 = 10 ** p2 - 1
        P1[isMissing] = fv
        P2[isMissing] = fv
        mse[n] = np.mean((p1[~isMissing] - p2[~isMissing]) ** 2) * 100

        P1_grid[n] = grid_orbit_backward(lons_gprof, lats_gprof, P1, 
                                         lons, lats, 20)
        P2_grid[n] = grid_orbit_backward(lons_gprof, lats_gprof, P2, 
                                         lons, lats, 20)

    bounds = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2)
    cmap = get_cmap(option = 'tropical', background = 'light')
    nrow = len(filenames)

    fig = plt.figure(figsize = (dcol, dcol * (0.275 * nrow + 0.07)))
    subfigs = fig.subfigures(2, 1, height_ratios = [0.07, 0.275 * nrow])
    plt.subplots_adjust(left = 0.025, right = 0.975, 
                        bottom = 0.025 / nrow, top = 1 - 0.025 / nrow,
                        wspace = 0.05, hspace = 0.025 * nrow)

    ax0 = subfigs[0].add_axes([0.575, 0.65, 0.35, 0.15])
    cb = ColorbarBase(ax0, cmap = cmap,
                      norm = BoundaryNorm(bounds, cmap.N),
                      orientation = 'horizontal', ticks = bounds,
                      extend = 'both', format = '%g')
    cb.ax.tick_params(labelsize = 7) 
    cb.set_label('Precipitation Rate (mm / h)', fontsize = 7, labelpad = -0.8)

    for n in range(len(filenames)):
        ax1 = subfigs[1].add_subplot(nrow, 2, 2 * n + 1, 
                                     projection = ccrs.PlateCarree())
        ax1.set_title(f'{spl[2 * n]} {labels[n]}', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax1, P1_grid[n], cmap, use_bkgndmap = False)
        ax1.coastlines(resolution = '110m', color = '0.5', lw = 0.375)
        ax2 = subfigs[1].add_subplot(nrow, 2, 2 * n + 2, 
                                     projection = ccrs.PlateCarree())
        ax2.set_title(f'{spl[2 * n + 1]} SPEEDe-GPROF (score: {mse[n]:5.3f})', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax2, P2_grid[n], cmap, use_bkgndmap = False)
        ax2.coastlines(resolution = '110m', color = '0.5', lw = 0.375)

    plt.savefig(f'fig.{option}.png')
    plt.close()

elif option == 'snapshot_SPEEDe-GPROF_badorbits':

    import h5py
    import tensorflow as tf
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs

    modelpath = '../GPROF/saved_model/'
    filenames = ('/path/to/GPROF/files/'
                 '2A-CLIM.METOPA.MHS.GPROF2021v1.20151014-S122615-E140736.'
                 '046624.V07A.HDF5', 
                 '/path/to/GPROF/files/'
                 '2A-CLIM.AQUA.AMSRE.GPROF2021v1.20100729-S160007-E173859.'
                 '043811.V07A.HDF5', 
                 '/path/to/GPROF/files/'
                 '2A-CLIM.NOAA17.AMSUB.GPROF2021v1.20051029-S134849-E153000.'
                 '017396.V07A.HDF5')
    modelNames = ('SPEEDe.GPROF-MHS.v1.0.0', 
                  'SPEEDe.GPROF-AMSRE.v1.0.0', 
                  'SPEEDe.GPROF-AMSUB.v1.0.0')
    labels = ('MetOp-A MHS orbit 046624',
              'Aqua AMSR-E orbit 043811',
              'NOAA-17 AMSU-B orbit 017396')

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

    P1_grid, P2_grid, mse = {}, {}, {}

    for n, (filename, modelName) in enumerate(zip(filenames, modelNames)):

        autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')
        _, ntrack, nswath, _ = autoencoder.layers[0].input_shape[0]
        fv = -9999.9

        with h5py.File(filename, 'r') as f:
            P1 = fix_GPROF_shape(f['S1/surfacePrecipitation'][:], ntrack, nswath)
            lons_gprof = fix_GPROF_shape(f['S1/Longitude'][:], ntrack, nswath, fv)
            lats_gprof = fix_GPROF_shape(f['S1/Latitude'][:], ntrack, nswath, fv)
        lons_gprof[lons_gprof == -9999] = fv
        lats_gprof[lats_gprof == -9999] = fv

        isMissing = P1 < 0
        P1[isMissing] = 0
        p1 = np.log10(P1 + 1)
        p2 = autoencoder.predict(p1[None], verbose = 0)[0, :, :, 0]
        P2 = 10 ** p2 - 1
        P1[isMissing] = fv
        P2[isMissing] = fv
        mse[n] = np.mean((p1[~isMissing] - p2[~isMissing]) ** 2) * 100

        P1_grid[n] = grid_orbit_backward(lons_gprof, lats_gprof, P1, 
                                         lons, lats, 20)
        P2_grid[n] = grid_orbit_backward(lons_gprof, lats_gprof, P2, 
                                         lons, lats, 20)

    bounds = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2)
    cmap = get_cmap(option = 'tropical', background = 'light')
    nrow = len(filenames)

    fig = plt.figure(figsize = (dcol, dcol * (0.275 * nrow + 0.07)))
    subfigs = fig.subfigures(2, 1, height_ratios = [0.07, 0.275 * nrow])
    plt.subplots_adjust(left = 0.025, right = 0.975, 
                        bottom = 0.025 / nrow, top = 1 - 0.025 / nrow,
                        wspace = 0.05, hspace = 0.025 * nrow)

    ax0 = subfigs[0].add_axes([0.575, 0.65, 0.35, 0.15])
    cb = ColorbarBase(ax0, cmap = cmap,
                      norm = BoundaryNorm(bounds, cmap.N),
                      orientation = 'horizontal', ticks = bounds,
                      extend = 'both', format = '%g')
    cb.ax.tick_params(labelsize = 7) 
    cb.set_label('Precipitation Rate (mm / h)', fontsize = 7, labelpad = -0.8)

    for n in range(len(filenames)):
        ax1 = subfigs[1].add_subplot(nrow, 2, 2 * n + 1, 
                                     projection = ccrs.PlateCarree())
        ax1.set_title(f'{spl[2 * n]} {labels[n]}', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax1, P1_grid[n], cmap, use_bkgndmap = False)
        ax1.coastlines(resolution = '110m', color = '0.5', lw = 0.375)
        ax2 = subfigs[1].add_subplot(nrow, 2, 2 * n + 2, 
                                     projection = ccrs.PlateCarree())
        ax2.set_title(f'{spl[2 * n + 1]} SPEEDe-GPROF (score: {mse[n]:5.3f})', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax2, P2_grid[n], cmap, use_bkgndmap = False)
        ax2.coastlines(resolution = '110m', color = '0.5', lw = 0.375)

    plt.savefig(f'fig.{option}.png')
    plt.close()

elif option == 'snapshot_SPEEDe-GPROF_variability':

    import h5py
    import tensorflow as tf
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs

    modelpath = '../GPROF/saved_model/'
    filenames = ('/path/to/GPROF/files/'
                 '2A-CLIM.NOAA19.MHS.GPROF2021v1.20210603-S070441-E084640.'
                 '063492.V07A.HDF5', 
                 '/path/to/GPROF/files/'
                 '2A-CLIM.F17.SSMIS.GPROF2021v1.20090805-S151159-E165355.'
                 '014198.V07A.HDF5')
    modelNames = ('SPEEDe.GPROF-MHS.v1.0.0', 
                  'SPEEDe.GPROF-SSMIS.v1.0.0')
    labels = ('NOAA-19 MHS orbit 063492',
              'F17 SSMIS orbit 014192')

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

    P1_grid, P2_grid, mse = {}, {}, {}

    for n, (filename, modelName) in enumerate(zip(filenames, modelNames)):

        autoencoder = tf.keras.models.load_model(f'{modelpath}{modelName}.h5')
        _, ntrack, nswath, _ = autoencoder.layers[0].input_shape[0]
        fv = -9999.9

        with h5py.File(filename, 'r') as f:
            P1 = fix_GPROF_shape(f['S1/surfacePrecipitation'][:], ntrack, nswath)
            lons_gprof = fix_GPROF_shape(f['S1/Longitude'][:], ntrack, nswath, fv)
            lats_gprof = fix_GPROF_shape(f['S1/Latitude'][:], ntrack, nswath, fv)
        lons_gprof[lons_gprof == -9999] = fv
        lats_gprof[lats_gprof == -9999] = fv

        isMissing = P1 < 0
        P1[isMissing] = 0
        p1 = np.log10(P1 + 1)
        p2 = autoencoder.predict(p1[None], verbose = 0)[0, :, :, 0]
        P2 = 10 ** p2 - 1
        P1[isMissing] = fv
        P2[isMissing] = fv
        mse[n] = np.mean((p1[~isMissing] - p2[~isMissing]) ** 2) * 100

        P1_grid[n] = grid_orbit_backward(lons_gprof, lats_gprof, P1, 
                                         lons, lats, 20)
        P2_grid[n] = grid_orbit_backward(lons_gprof, lats_gprof, P2, 
                                         lons, lats, 20)

    bounds = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2)
    cmap = get_cmap(option = 'tropical', background = 'light')
    nrow = len(filenames)

    fig = plt.figure(figsize = (dcol, dcol * (0.275 * nrow + 0.07)))
    subfigs = fig.subfigures(2, 1, height_ratios = [0.07, 0.275 * nrow])
    plt.subplots_adjust(left = 0.025, right = 0.975, 
                        bottom = 0.025 / nrow, top = 1 - 0.025 / nrow,
                        wspace = 0.05, hspace = 0.025 * nrow)

    ax0 = subfigs[0].add_axes([0.575, 0.65, 0.35, 0.15])
    cb = ColorbarBase(ax0, cmap = cmap,
                      norm = BoundaryNorm(bounds, cmap.N),
                      orientation = 'horizontal', ticks = bounds,
                      extend = 'both', format = '%g')
    cb.ax.tick_params(labelsize = 7) 
    cb.set_label('Precipitation Rate (mm / h)', fontsize = 7, labelpad = -0.8)

    for n in range(len(filenames)):
        ax1 = subfigs[1].add_subplot(nrow, 2, 2 * n + 1, 
                                     projection = ccrs.PlateCarree())
        ax1.set_title(f'{spl[2 * n]} {labels[n]}', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax1, P1_grid[n], cmap, use_bkgndmap = False)
        ax1.coastlines(resolution = '110m', color = '0.5', lw = 0.375)
        ax2 = subfigs[1].add_subplot(nrow, 2, 2 * n + 2, 
                                     projection = ccrs.PlateCarree())
        ax2.set_title(f'{spl[2 * n + 1]} SPEEDe-GPROF (score: {mse[n]:5.3f})', 
                      loc = 'left', y = 0.97)
        plot_imerg_snapshot(ax2, P2_grid[n], cmap, use_bkgndmap = False)
        ax2.coastlines(resolution = '110m', color = '0.5', lw = 0.375)

    plt.savefig(f'fig.{option}.png')
    plt.close()