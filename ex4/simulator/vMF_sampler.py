#!/usr/bin/env python
import numpy as np

class VonMisesFisher:
    def __init__(self):
        pass

    def sample_vMF(self, mu, kappa, num_samples):
        """Generate num_samples N-dimensional samples from von Mises Fisher
        distribution around center mu \in R^N with concentration kappa.
        """
        dim = len(mu)
        result = np.zeros((num_samples, dim))
        for nn in range(num_samples):
            # sample offset from center (on sphere) with spread kappa
            w = self._sample_weight(kappa, dim)

            # sample a point v on the unit sphere that's orthogonal to mu
            v = self._sample_orthonormal_to(mu)

            # compute new point
            result[nn, :] = v * np.sqrt(1. - w ** 2) + w * mu

        return result


    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)

        while True:
            z = np.random.beta(dim / 2., dim / 2.)
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
                return w


    def _sample_orthonormal_to(self, mu):
        """Sample point on sphere orthogonal to mu.
        """
        v = np.random.randn(mu.shape[0])
        proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
        orthto = v - proj_mu_v
        return orthto / np.linalg.norm(orthto)


    def randomize_about_point(self, V, kappa=1.0, num_samples=1):
        x = V[:, 0]
        y = V[:, 1]
        z = V[:, 2]
        U = self.sample_vMF(np.array([x[0], y[0], z[0]]),
                            kappa,
                            num_samples)
        return U





