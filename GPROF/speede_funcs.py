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

def cond_mse(y_true, y_pred, threshold = 0.01):

    # last updated: 2023 10 31

    from tensorflow.math import square, reduce_sum, greater
    from tensorflow.math import logical_or, count_nonzero
    from tensorflow import float32

    num = reduce_sum(square(y_true - y_pred))
    den = count_nonzero(logical_or(greater(y_true, threshold), 
                                   greater(y_pred, threshold)),
                        dtype = float32)

    return num / den