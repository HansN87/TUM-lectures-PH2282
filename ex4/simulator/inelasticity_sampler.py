import numpy as np


class inelasticity_sampler(object):
    '''
    Samples the Bjorken-y inelasticity following the parametrization
    in arXiv:1102.0691.
    Sample generators acquired using inverse cdf method;
    differential cross sections are split into two regions,
    low-y and high-y.
    '''

    high_A = {}
    high_A['nubar_CC'] = [-.0026, .085, 4.1, 1.7]
    high_A['nu_CC'] = [-0.008, 0.26, 3.0, 1.7]
    high_A['nubar_NC'] = [-0.005, 0.23, 3.0, 1.7,]
    high_A['nu_NC'] = [-0.005, 0.23, 3.0, 1.7,]

    low_A = [0.0, 0.0941, 4.72, 0.456]

    def C_1(self, logE, a):
        return a[0] - a[1] * np.exp(-(logE - a[2])/ a[3])

    def C_2(self, logE):
        return 2.55 - 0.0949 * logE

    def f(self, logE):
        '''
        Determines the probability whether the draw is in
        low-y or high-y region
        '''
        return 0.128 * np.sin(-.197 * (logE - 21.8) / 180. * 2*np.pi)

    def low_y(self, logE):
        ymin = 0.
        ymax = 1e-3

        r = np.random.uniform()
        c1 = self.C_1(logE, self.low_A)
        c2 = self.C_2(logE)
        k = -1./c2+1.
        y = c1 + (r * (ymax - c1)**k + (1.-r) * (ymin - c1)**k)**(c2/(c2-1.))
        return y

    def high_y(self, logE, interaction):
        ymin = 1e-3
        ymax = 1.

        r = np.random.uniform()
        c1 = self.C_1(logE, self.high_A[interaction])
        y = (ymax - c1)**r / (ymin-c1)**(r-1) + c1
        return y

    def draw_inelasticity(self, energy, interaction):
        logE = np.log10(energy)
        r1 = np.random.uniform()
        if r1 < self.f(logE):
            return self.low_y(logE)
        else:
            return self.high_y(logE, interaction)
