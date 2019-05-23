#!/usr/bin/env python

import numpy as np
from scipy.interpolate import RectBivariateSpline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lower_bound_low=-5.0 # log aeff to zero
lower_bound_high=-3.0
lE_limit_low = 3.0
lE_limit_high = 7.0

#infile = "../aeff_input/effective_area.per_bin.nu_mu.cc.cascade.txt"
#smoother = 150 ### problem

#infile = "../aeff_input/effective_area.per_bin.nu_mu.nc.cascade.txt"
#smoother = 300 ### problem ### approximate with nue nc events

#infile = "../aeff_input/effective_area.per_bin.nu_tau.cc.cascade.txt"
#smoother = 5

infile = "../aeff_input/effective_area.per_bin.nu_tau.nc.cascade.txt"
smoother = 30 ### problem

#infile = "../aeff_input/effective_area.per_bin.nu_e.cc.cascade.txt"
#smoother = 1

#infile = "../aeff_input/effective_area.per_bin.nu_e.nc.cascade.txt"
#smoother = 5

#infile = "../aeff_input/effective_area.per_bin.nu_e_bar.gr.cascade.txt"
#smoother=15
#lower_bound_high = -4.0 # for GR

table = np.loadtxt(infile, skiprows=2)

enu_low = np.log10(table[:,0])
enu_high = np.log10(table[:,1])

ctnu_low = table[:,2]
ctnu_high = table[:,3]

log_enu_binc = 0.5*(enu_low + enu_high)
log_enu_binc = log_enu_binc.reshape((len(log_enu_binc), 1))

ctnu_binc = 0.5*(ctnu_low + ctnu_high)
ctnu_binc = ctnu_binc.reshape((len(ctnu_binc), 1))

table = np.append(table, log_enu_binc, axis=1)
table = np.append(table, ctnu_binc, axis=1)

aeff_arr = []
logE_arr = []
ct_arr = []


# enforce bounds
xvals = np.unique(table[:,-2])
idx = xvals < lE_limit_low
k = len(xvals[idx])-2
xvals = xvals[k:]
idx = xvals < lE_limit_high
k = len(xvals[idx])+2
xvals = xvals[:k]
print xvals[-1]
 
yvals = np.unique(table[:,-1])

# need to expand boundary

for logE in xvals:
    for ct in yvals:
        idx = np.logical_and(table[:,-1]==ct, table[:,-2]==logE)
        aeff = np.log10(np.sum(table[idx][:,-4]))
        if np.isinf(aeff):
            if logE < 5.0:
                aeff=lower_bound_low
            else:
                aeff=lower_bound_high

        #print logE, ct, aeff
        aeff_arr.append(aeff)
        logE_arr.append(logE)
        ct_arr.append(ct)



spline = RectBivariateSpline(xvals, yvals, np.asarray(aeff_arr).reshape((len(xvals), len(yvals))), s=smoother)

x = np.linspace(xvals[0], xvals[-1], 200)
y = np.linspace(yvals[0], yvals[-1], 200)
xx, yy = np.meshgrid(x, y)
xx = xx.T
yy = yy.T
z = spline(x, y)
fig = plt.figure(figsize=(18, 10)) 
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(xx, yy, z, rstride=5, cstride=5)

ax.set_xlabel('log10(energy/GeV)')
ax.set_ylabel('cos(zenith)')
ax.set_xlim([lE_limit_low, lE_limit_high])
ax.set_ylim([-1.0, 1.0])
ax.set_zlim([-4.0, 3.0])

# Generate a scatter plot of the points
x = logE_arr
y = ct_arr

# print(self.effective_area)
z = aeff_arr
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('log10(energy/GeV)')
ax.set_ylabel('cos(zenith)')
ax.set_zlabel('log10(Aeff)')

plt.show()


xvals_avg = (0.5*(xvals[1:]+xvals[:-1]))[::2] # only every second

new = np.ones(len(xvals_avg)*len(yvals))
new = new.reshape((len(xvals_avg), len(yvals)))

# average along energy axis
aeff_arr_shaped = np.asarray(aeff_arr).reshape((len(xvals), len(yvals)))
for column_idx in range(len(yvals)):
    aeff_avg = aeff_arr_shaped[:, column_idx]
    aeff_avg = (0.5*(aeff_avg[1:]+aeff_avg[:-1]))[::2]
    new[:, column_idx]=np.array(aeff_avg)


#spline = RectBivariateSpline(xvals_avg, yvals, new, s=1.5)
spline = RectBivariateSpline(xvals_avg, yvals, new, s=smoother)

#x = np.linspace(xvals_avg[0], xvals_avg[-1], 200)
x = np.linspace(3.0, 7.0, 200)
#y = np.linspace(yvals[0], yvals[-1], 200)
y = np.linspace(-1.0,1.0,200)
xx, yy = np.meshgrid(x, y)
xx = xx.T
yy = yy.T
z = spline(x, y)
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(xx, yy, z, rstride=5, cstride=5)

ax.set_xlabel('log10(energy/GeV)')
ax.set_ylabel('cos(zenith)')
ax.set_xlim([lE_limit_low, lE_limit_high])
ax.set_ylim([-1.0, 1.0])
ax.set_zlim([-4.0, 3.0])

# Generate a scatter plot of the points
x = logE_arr
y = ct_arr

# print(self.effective_area)
z = aeff_arr
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('log10(energy/GeV)')
ax.set_ylabel('cos(zenith)')
ax.set_zlabel('log10(Aeff)')

plt.show()

   









#idx = np.logical_and(table[:,-1]==table[:,-1][10], table[:,-2]==table[:,-2][0])
#print len(table[idx])
#print table[idx][:,-4]
#for obj in zip(table[:,-1], table[:,-2]):
#    print obj

#print np.unique(table[:,-1])
#print np.unique(table[:,-2])



