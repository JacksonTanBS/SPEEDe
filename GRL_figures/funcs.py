def read_IMERG_hhr(year, month, day, hhr, ver = 'V07A', run = 'F', 
                   var = 'precipitation', basedir = '/gpm3/data/IMERG/'):

    # last updated: 2023 09 10

    import h5py
    from datetime import datetime, timedelta

    runname = {'F': 'Final', 'L': 'Late', 'E': 'Early'}

    if int(ver[1 : 3]) >= 7:
        useOldDir = False
    else:
        useOldDir = True

    # define the subdirectory and filename
    t = datetime(year, month, day, hhr // 2, (hhr % 2) * 30)
    t0 = t.strftime('%H%M%S')
    t1 = (t + timedelta(seconds = 1799)).strftime('%H%M%S')
    t2 = (t - datetime(year, month, day)).total_seconds() / 60
    if   run == 'F':
        imergpath = ('%s%s/HHR/%s/%4d/%02d/%02d/' % 
                     (basedir, runname[run], ver, year, month, day))
        imergfile = ('3B-HHR.MS.MRG.3IMERG.%4d%02d%02d-S%s-E%s.%04d.%s.HDF5'
                      % (year, month, day, t0, t1, t2, ver))
    elif run == 'L' or run == 'E':
        imergpath = ('%s%s/HHR/%s/%4d%02d/' % 
                     (basedir, runname[run], ver, year, month))
        imergfile = ('3B-HHR-%s.MS.MRG.3IMERG.%4d%02d%02d-S%s-E%s.%04d.%s.RT-H5'
                      % (run, year, month, day, t0, t1, t2, ver))

    # define the directory structure for the variable in the file
    if useOldDir:
        dir = 'Grid/'
    else:
        if var in ('precipitationUncal', 'MWprecipitation', 'MWprecipSource',
                   'MWobservationTime', 'IRinfluence', 'IRprecipitation'):
            dir = 'Grid/Intermediate/'
        else:
            dir = 'Grid/'

    with h5py.File('%s%s' % (imergpath, imergfile), 'r') as f:
        x = f['%s%s' % (dir, var)][:]

    return x

def get_months(year, year0, month0, year1, month1):
    if   year0 == year1: return range(month0, month1 + 1)
    elif year  == year0: return range(month0, 13)
    elif year  == year1: return range(1, month1 + 1)
    else               : return range(1, 13)

def fix_GPROF_shape(X, ntrack, nswath, pad_value = 0):
    import numpy as np
    X = X[:ntrack, :nswath]    # truncate if too long
    if X.shape[0] < ntrack:    # pad if too short (along-track)
        X = np.pad(X, ((0, ntrack - X.shape[0]), (0, 0)), 
                   constant_values = pad_value)
    if X.shape[1] < nswath:    # pad if too short (along-swath)
        X = np.pad(X, ((0, 0), (0, nswath - X.shape[1])), 
                   constant_values = pad_value)
    return X

def get_cmap(option = 'tropical', background = 'dark'):

    # last updated: 2023 11 28

    import sys
    from matplotlib.colors import ListedColormap

    if option == 'cosmic':
        cols = ('#45226e', '#5b1fa8', '#6127e7', '#4d53e6', 
                '#3472e8', '#2488df', '#1598de', '#01aedd', 
                '#00c2d9', '#18ddd4', '#49eecf')
    elif option == 'ember':
        cols = ('#542040', '#7b2348', '#a01f42', '#c81d35', 
                '#e63c16', '#f25907', '#fa7700', '#fe9700', 
                '#fbb61b', '#f2d82a', '#ffef00')
    elif option == 'tropical':
        cols = ('#900ea5', '#b00689', '#ca1464', '#d7383d', 
                '#d85e1e', '#cf8203', '#bba218', '#98c143', 
                '#5edd7d', '#24f0c0', '#44fcfc')
    elif option == 'SVS':
        cols = ('#008114', '#008b52', '#00b330', '#60d300', 
                '#b4e700', '#fffb00', '#ffc400', '#ff9300', 
                '#ff0000', '#c80000', '#910000')
    else:
        sys.exit(f'Error: colormap option {option} unknown.')

    if background == 'dark':
        cmap = ListedColormap(cols[0:10])
        cmap.set_under('#000000')
        cmap.set_over(cols[10])
    elif background == 'light':
        cmap = ListedColormap(cols[10:0:-1])
        cmap.set_under('#ffffff')
        cmap.set_over(cols[0])
    else:
        sys.exit(f'Error: background option {background} unknown.')

    return cmap

def plot_imerg_snapshot(ax, P, cmap, use_bkgndmap = True):

    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs

    lats = read_IMERG_hhr(2015, 1, 1, 0, var = 'lat')
    lons = read_IMERG_hhr(2015, 1, 1, 0, var = 'lon')
    nlon, nlat = len(lons), len(lats)
    lonedges = np.linspace(-180, 180, nlon + 1)
    latedges = np.linspace(-90, 90, nlat + 1)
    fv = -9999.9

    #bounds = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10, 20, 50)
    bounds = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2)

    if use_bkgndmap:
        mappath = '/gpm3/btan1/plots/bkgnd_maps/'
        bkgndmap = plt.imread(f'{mappath}world.200407.3x5400x2700.png')
        ax.imshow(bkgndmap, origin = 'upper', extent = (-180, 180, -90, 90),
                  interpolation = 'none', transform = ccrs.PlateCarree())

    ax.set_aspect('equal')    # 'equal' or 'auto'
    ax.pcolormesh(lonedges, latedges, P.T == fv,
                  cmap = ListedColormap(('#ffffff00', '#7f7f7f7f')),
                  transform = ccrs.PlateCarree())
    ax.pcolormesh(lons, lats, np.ma.masked_less(P, bounds[0]).T, 
                  cmap = cmap, norm = BoundaryNorm(bounds, cmap.N),
                  transform = ccrs.PlateCarree())
    ax.set_global()

    return None