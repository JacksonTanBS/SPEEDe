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

def reduce_imerg(p, nx, ny, method = 'mean', 
                 nlon = 3600, nlat = 1800, fv = -9999.9):

    import numpy as np

    p[p == fv] = 0
    if method == 'mean':
        p = np.mean(p.reshape(nlon // nx, nx, nlat // ny, ny), (1, 3))
    elif method == 'max':
        p = np.max(p.reshape(nlon // nx, nx, nlat // ny, ny), (1, 3))
    return np.log10(p + 1)

def get_months(year, year0, month0, year1, month1):
    if   year0 == year1: return range(month0, month1 + 1)
    elif year  == year0: return range(month0, 13)
    elif year  == year1: return range(1, month1 + 1)
    else               : return range(1, 13)

def cond_mse(y_true, y_pred):

    # last updated: 2023 10 18

    from tensorflow.math import square, reduce_sum, greater
    from tensorflow.math import logical_or, count_nonzero
    from tensorflow import float32

    num = reduce_sum(square(y_true - y_pred))
    den = count_nonzero(logical_or(greater(y_true, 0), greater(y_pred, 0)),
                        dtype = float32)

    return num / den