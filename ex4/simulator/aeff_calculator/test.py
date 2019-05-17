#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from aeff_calculator import effective_area

aeff = effective_area("nue_CC", "cascades")

x = np.linspace(3.0, 7.0, 200)
y = np.linspace(-1.0,1.0,200)
xx, yy = np.meshgrid(x, y)
xx = xx.T
yy = yy.T
z = aeff.eval(x, y)

fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(xx, yy, z, rstride=5, cstride=5)

ax.set_xlabel('log10(energy/GeV)')
ax.set_ylabel('cos(zenith)')
ax.set_xlim([3.0, 7.0])
ax.set_ylim([-1.0, 1.0])
ax.set_zlim([-4.0, 3.0])

# Generate a scatter plot of the points
x = aeff.lE_vals
y = aeff.ct_vals

# print(self.effective_area)

z = aeff.aeff_vals

ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('log10(energy/GeV)')
ax.set_ylabel('cos(zenith)')
ax.set_zlabel('log10(Aeff)')


plt.show()

