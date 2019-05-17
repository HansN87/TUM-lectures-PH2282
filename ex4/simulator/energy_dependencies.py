from scipy.interpolate import UnivariateSpline
from scipy.special import erfc
import numpy as np
import os


def filter_duplicate_x(x, y):
    """ UnivariateSpline requires x to be strictly increasing,
        removes data points with duplicate x values """
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = np.array(y)[idx]
    return x_unique, y_unique


class extra_interpolator(object):
    """ Interpolator wrapper with user-defined border behavior

    border_control argument takes tuple with function that defines
    border behaviour as well as which border it should be applied at
    ('lower', 'upper', 'both'). In case of 'lower' or 'upper',
    the respectively last or first point of the interpolation range is
    used instead.
    """
    def __init__(self, x, y, border_control=None, s=None):
        self.lower = min(x)
        self.upper = max(x)
        x_unique, y_unique = filter_duplicate_x(x, y)
        self.interpolator = UnivariateSpline(x_unique, y_unique, s=s)

        if border_control == None:
            def b(x):
                u = self.lower if x <= self.lower else self.upper
                return self.interpolator(u)

        elif border_control[1] == 'both':
            def b(x):
                return border_control[0](x)

        elif border_control[1] == 'lower':
            def b(x):
                if x <= self.lower:
                    return border_control[0](x)
                else:
                    return self.interpolator(self.upper)

        elif border_control[1] == 'upper':
            def b(x):
                if x <= self.lower:
                    return self.interpolator(self.lower)
                else:
                    return border_control[0](x)

        self.border_control = b


    def __call__(self, x):
        def evaluate(a):
            if self.lower <= a <= self.upper:
                return self.interpolator(a)
            else:
                return self.border_control(a)
        v = np.vectorize(evaluate)
        return v(x)


def get_xy_from_file(filename):
    d = np.genfromtxt(filename, skip_header=2, delimiter=',')
    return d[:,0], d[:,1]


def angular_border_conrol(x):
    """ Function to evaluate when outside the
    angular resolution interpolation range"""

    def erfc_log(X, a, b, c, d):
        scale = (a - c)/2.
        return scale * erfc(b * (np.log(X) - np.log(d))) + c

    args = [180, .6, 2.4, .8e4] # Not a sophisticated pick atm
    return erfc_log(x, *args)


def upper_eres_border_control(x):
    a = 0.00691827
    b = -0.02570866
    return a * np.log(x) + b


dirname = os.path.dirname(__file__)


def get_angres_exinterpolator():
    filename =  os.path.join(dirname, 'edep_plots/angular_resolution.csv')
    x, y = get_xy_from_file(filename)
    return extra_interpolator(x, y, (angular_border_conrol, 'both'))


def get_eres_exinterpolator():
    filename =  os.path.join(dirname, 'edep_plots/erec_resolution.csv')
    x, y = get_xy_from_file(filename)
    return extra_interpolator(x[:-5], 1e-2 * y[:-5],
                        border_control=(upper_eres_border_control, 'upper'),
                        s=2e-4)
