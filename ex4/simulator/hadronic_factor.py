import numpy as np


def hadronic_reduction_factor(energy):
    ''' Hadronic light yield reduction factor from Leif Raedel's thesis '''

    def F(E, Es, f, m, sig, gamma):
        mean = 1 - (1-f) * (E/Es)**(-m)
        sigma = sig * np.log(E)**(-gamma)
        return mean, sigma

    # Parameterization for a pi+ particle
    pi_plus = [0.15591, 0.27273, 0.15782, 0.40626, 1.01771]

    lower_bound = 1e1 # Lower energy boundary where function is defined

    return F(energy, *pi_plus) if energy > lower_bound else F(lower_bound, *pi_plus)

def sample_hadronic_reduction_factor(energy):
    mean, sigma = hadronic_reduction_factor(energy)

    factor = 0.
    while factor <= 0.:
        factor = np.random.normal(mean, sigma)
    return factor
