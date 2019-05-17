import numpy as np
import pandas as pd


class nn_2d(object):
    ''' Nearest neighbor 2D interpolator '''
    def __init__(self, xaxis, yaxis, z):
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.z = z
        
    def get_index(self, val, axis):
        return np.abs(axis - val).argmin()
    
    def __call__(self, xval, yval):
        xi = self.get_index(xval, self.xaxis)
        yi = self.get_index(yval, self.yaxis)
        return self.z[xi][yi]


def get_effective_area_interpolator(filename='simulator/effective_area_files/IC86-2012-TabulatedAeff.txt'):
    ''' 
    Gets a 2D nearest neighbor interpolator for the effective area
    using the specified file.
    Returns a interpolator that takes logE (GeV) and cos(theta) as arguments.
    '''
    filelayout = ['E_min[GeV]', 'E_max[GeV]', 'cos(zenith)_min', 'cos(zenith)_max', 'Aeff[m^2]']
    df = pd.read_csv(filename, delim_whitespace=True, comment='#', names=filelayout)

    # Values are saved with bin edges
    es = (np.log10(df['E_min[GeV]']) + np.log10(df['E_max[GeV]'])) / 2.
    cosths = (df['cos(zenith)_min'] + df['cos(zenith)_max']) / 2.
    e = np.sort(np.unique(es))
    costh = np.sort(np.unique(cosths))
    aeff = np.array(df['Aeff[m^2]']).reshape((len(e),len(costh)))

    interp = nn_2d(e, costh, aeff)
    return interp
