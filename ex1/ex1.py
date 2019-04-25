#!/usr/bin/env python

import numpy
import pylab as plt
from scipy import optimize

def model(x, sgnExpectation, bkgExpectation):
   # define parameters for the signal distribution, i.e. mean and sigma of the guassian
   mean       =  10;
   sigma      =   1;
   # define parameters for the background distribution, i.e. the window edges
   rangeMin   =   0; 
   rangeMax   =  20;
   rangeWidth = rangeMax - rangeMin;
   # define binning
   return bkgExpectation*(1.0/rangeWidth) + sgnExpectation / numpy.sqrt(2*numpy.pi*sigma*sigma) * numpy.exp(-1./2.*((x-mean)/sigma)**2);


# define ranges
x = numpy.arange(0, 20, 0.01)
y = model(x, 1, 1)

# plot model
ax = plt.axes()
ax.set_ylabel('counts')
ax.set_xlabel('energy [keV]')
ax.plot(x,y)
plt.show()
