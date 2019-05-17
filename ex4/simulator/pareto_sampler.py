import numpy as np


class pareto(object):
    def __init__(self, gamma, lower, upper):
        self.gamma = gamma
        self.lower = lower
        self.upper = upper

        # Variables for convenience
        self.g_int = 1. - self.gamma
        self.d = (self.upper**self.g_int - self.lower**self.g_int)

        self.norm = self.d / self.g_int # Normalization factor from pdf
        self.C = -1. * self.lower**self.g_int / self.d # constant to make cdf(lower)=0

    def pdf(self, x):

        def p(X):
            if self.lower < X < self.upper:
                return 1./self.norm * X**-self.gamma
            else:
                return 0.

        return np.vectorize(p)(x)

    def cdf(self, x):

        def c(X):
            if self.lower < X < self.upper:
                return 1./ self.norm * X**self.g_int / self.g_int + self.C
            elif X < self.lower:
                return 0.
            elif X > self.upper:
                return 1.

        return np.vectorize(c)(x)

    def inv_cdf(self, x):
        return ((x-self.C) * self.g_int * self.norm)**(1./self.g_int)

    def sample_pareto(self, size=1):
        u = np.random.uniform(size=size)
        return self.inv_cdf(u)
