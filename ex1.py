#############################################################################
# Statistical and Machine Learning Methods in Particle and Astrophysics
#
# TUM - summer term 2019
# M. Agostini <matteo.agostini@tum.de> and Hans Niederhausen@tum.de <hans.niederhausen@tum.de>
#
# Ex 1, conceptual steps:
#   1) build a model 
#   2) build a data set by sampling the model and storing the events in a hist
#   3) plot model and pseudo-data set for different background and signals rates
#
#############################################################################

#!/usr/bin/env python

import numpy
import pylab
from scipy.stats import norm

# define expectations
lambda_s = 1000.
lambda_b = 1000.

# define energy range
xmin =  0.0
xmax = 20.0

# define Gaussian mean and sigma
gaussian_mean =  10.0
gaussian_sigma =  1.0

# define model as a function. The function works on a vector of samples x
def model(x, lambda_s=lambda_s, lambda_b=lambda_b, xmin=xmin, xmax=xmax, gaussian_mean=gaussian_mean, gaussian_sigma=gaussian_sigma): 
    # uniform background + gaussian signal 
    result = (1./(lambda_s+lambda_b)) * (lambda_s  * norm.pdf(x, gaussian_mean, gaussian_sigma)  + lambda_b * 1./(xmax - xmin) )

    # check bounds and set to zero samples out of defined energy range
    idx = numpy.where(x < xmin)[0]
    result[idx]=numpy.zeros(len(idx))
    idx = numpy.where(x > xmax)[0]
    result[idx]=numpy.zeros(len(idx))
    return result

# build data set
def BuildDataset(lambda_s=lambda_s, lambda_b=lambda_b, xmin=xmin, xmax=xmax, gaussian_mean=gaussian_mean, gaussian_sigma=gaussian_sigma): 
    # define number of counts generating a random number from poisson distribution
    lambda_tot = lambda_s+lambda_b
    N = numpy.random.poisson(lambda_tot,1)

    # data are generated in two steps. First each event is randomly attributed
    # to the signal or background. Then the energy distribution is sampled form
    # the proper pdf
    #
    # First step generate random numbers between 0 and 1 and attribute each
    # event to signal or background using the weight between the expectation for
    # signal counts and the total
    rvars = numpy.random.uniform(0,1, N)

    idx_s = numpy.where(rvars <= lambda_s / lambda_tot)[0]
    idx_b = numpy.where(rvars >  lambda_s / lambda_tot)[0]

    # Second step. Store in "samples" the energy value of each event
    samples = numpy.zeros(N)
    # draw samples that are realized as background events from the background distribution 
    samples[idx_b]=numpy.random.uniform(xmin, xmax, len(idx_b))
    # draw samples that are realized as signal events from the signal distribution 
    samples[idx_s]=numpy.random.normal(gaussian_mean, gaussian_sigma, len(idx_s))

    return samples


# plot model
tx = numpy.linspace(xmin, xmax, 10000)
pylab.plot(tx, model(tx), color='red', linestyle='dashed', linewidth=1.5, label='model')
pylab.xlabel('energy [a.u.]')
pylab.ylabel('pdf')
pylab.legend()
pylab.show()

# set random seed
numpy.random.seed(0)
# generate data set
samples = BuildDataset()

# define binning of the histogram used to display the data
nbins = 100
bins = numpy.linspace(xmin, xmax, nbins+1)
obs_probabilities, bins, _ = pylab.hist(samples, bins=bins, normed=False, histtype='step', color='black', label='data')
pylab.xlabel('energy [a.u.]')
pylab.ylabel('pdf')
pylab.legend()
pylab.show()

