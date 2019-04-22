#!/usr/bin/env python

import numpy
import pylab as plt
from scipy import optimize

def model(x, sgnExpectation, bkgExpectation):
   # define parameters for the signal distribution, i.e. mean and sigma of the guassian
   mean       =   0;
   sigma      =   1;

   # define parameters for the background distribution, i.e. the window edges
   rangeMin   = -10; 
   rangeMax   =  10;
   rangeWidth = rangeMax - rangeMin;

   # define binning
   return bkgExpectation*(1.0/rangeWidth) + sgnExpectation / numpy.sqrt(2*numpy.pi*sigma*sigma) * numpy.exp(-1./2.*((x-mean)/sigma)**2);


# define ranges
x = numpy.arange(-10, 10, 0.01)
y = model(x, 1, 1)

# plot model
fig = plt.figure(figsize=(7, 6), dpi=200)
ax = plt.axes()
ax.plot(x,y,"b-",label="true model")
ax.set_ylabel('counts', position=(0., 1.), va='top', ha='right')
ax.set_xlabel('energy [keV]', position=(1., 0.), va='bottom', ha='right')
ax.yaxis.set_label_coords(-0.12, 1.)
ax.xaxis.set_label_coords(1.0, -0.1)
ax.legend()
plt.show()




