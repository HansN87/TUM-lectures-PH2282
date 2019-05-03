'''
Statistical and Machine Learning Methods in Particle and Astrophysics

TUM - summer term 2019
M. Agostini <matteo.agostini@tum.de> and Hans Niederhausen@tum.de <hans.niederhausen@tum.de>

Ex 1, conceptual steps
    1) build a model 
    2) build a data set by sampling the model and storing the events in a hist
    3) plot model and pseudo-data set for a given background and signals rate
    4) compute the maximum likelihood estimator (MLE) for the background and 
       signal expectation
    5) repeat 1000 times the points from 2 to 5. Store the MLE values and compute the 
       median, confirming that it converges to the injected value when the number of 
       sample increases
'''

import math 
import numpy as np
import pylab as pl
from scipy import optimize
from scipy import stats
from scipy.stats import norm
from scipy.integrate import simps
from scipy.interpolate import interp1d

#
# define model as a function. The function works on a vector of samples x
#
def PDF_Energy(x, lambda_s, lambda_b, gaussian_mean, gaussian_sigma, \
        x_min, x_max, check_normalization = True): 
    
    # first, we evaluate the normal/truncated normal distribution.
    # if necessary, we correctly normalize to unity. this is useful if the guassian 
    # is close to the borders
    norm_pdf_values = norm.pdf(x, gaussian_mean, gaussian_sigma)

    if (check_normalization):
        # compute normalization of the pdf
        normalization =  simps (norm_pdf_values, x, axis = 0)
        # check if pdf integral is close enough to 1
        if (np.fabs(normalization - 1.0) > 1E-8 ): 
            norm_pdf_values /= normalization
    

    # define actual probability function: sum of gaussian + uniform
    pdf = (1./(lambda_s+lambda_b)) * \
          (lambda_b * 1./(x_max - x_min) + lambda_s * norm_pdf_values )

    # check that the probability is tested within its defined range
    x_underflow = np.where(x < x_min)[0]
    x_overflow  = np.where(x > x_max)[0]
    x_out_of_range = x_underflow + x_overflow
    if (len(x_out_of_range) != 0):
        print('error: energy of an event out of range')
        exit()
  
    return pdf

#
# define negative log likelihood function as sum of probabilities
#
def nll (pseudo_data, lambda_s, lambda_b, gaussian_mean, gaussian_sigma, \
        x_min, x_max, check_normalization=False):
    
    # start by computing the extended term, i.e. the Poisson probability 
    # of observing N counts given an expectation of lambda)
    nll_extended_term = - stats.poisson.logpmf(k = pseudo_data.size, mu = lambda_s + lambda_b)
 
    # now compute the probability for the energy of each event given the model and
    # sum the logs. The implementation based on a loop would be
    #
    #for x in pseudo_data: 
    #    nll_energy -= math.log(
    #        PDF_Energy(x, lambda_s, lambda_b, gaussian_mean, gaussian_sigma, \
    #                   x_min, x_max, check_normalization=False) )
    #
    # however this is not efficient and can be reimplemented using numpy 
    #
    nll_energy = -np.sum( np.log(PDF_Energy(pseudo_data, lambda_s, lambda_b, \
                                            gaussian_mean, gaussian_sigma, \
                                            x_min, x_max, \
                                            check_normalization=check_normalization)) )
    return nll_extended_term + nll_energy;

#
# Define a function to sample a number of events given an expectation and then to 
# sample for each of them an energy given a pdf.
# The xmesh defines a vector of point used to invert the cdf before sampling
#
def GeneratePseudoData(expectation, pdf, xmesh):
    
    # sample the size of the pseudodata given an expectation
    size = stats.poisson.rvs(mu=expectation)
    
    # create vector of random samples betwen 0 and 1
    samples = np.random.uniform(0.0, 1.0, int(size))
    
    # compute cumulative distribution of the pdf
    cdf = np.cumsum(pdf); cdf /= max(cdf)
    
    # compute inverse of the cumulative over
    inverse_cdf = interp1d(cdf, xmesh, bounds_error=False, fill_value = 0)
    
    # assign values to the random samples
    return inverse_cdf(samples)

#
# This is the main program
#
def RunOnSingleDataset():
    
    # define expectations
    lambda_s, lambda_b = 100.0, 100.0
    
    # define Gaussian mean and sigma
    gaussian_mean, gaussian_sigma = 10.0, 1.0
    
    # define energy range
    x_min, x_max =  0.0, 20.0
  
    print("Executing anlaysis for lambda_s = %d and lambda_b = %d" % (lambda_s, lambda_b))
    
    # array of values used to interpolate the pdf and cdf
    x = np.linspace(x_min, x_max, 10001)

    # build energy pdf 
    pdf_energy = PDF_Energy(x, \
                     lambda_s = lambda_s, lambda_b = lambda_b, \
                     gaussian_mean = gaussian_mean, gaussian_sigma = gaussian_sigma, \
                     x_min = x_min, x_max = x_max)
    
    # plot energy PDF
    pl.plot(x,pdf_energy)
    pl.xlabel('energy [a.u.]')
    pl.ylabel('probability density')
    pl.show()
    
    # generate pseudo-data
    pseudo_data = GeneratePseudoData(expectation = lambda_s + lambda_b, pdf = pdf_energy, xmesh = x)
        
    # Compute a histogram of the sample
    bins = np.linspace(x_min, x_max, 100)
    hist = np.histogram(pseudo_data, bins=bins, normed=False)

    # plot the histogram
    pl.hist(pseudo_data, bins=bins, normed=False, label = 'pseudo-data')
    pl.plot(x, pdf_energy * (lambda_s + lambda_b) * (x_max-x_min) / bins.size, label = 'model')
    pl.xlabel('energy [a.u.]')
    pl.ylabel('counts')
    pl.legend(loc='best')
    pl.show()
    
    # scan the likelihood function around the expected minimum
    l_scan = np.linspace(lambda_s-5*math.sqrt(lambda_s), lambda_s+5*math.sqrt(lambda_s), 100)
    l_resu = []

    for scan_value in l_scan:
        l_resu.append(nll(pseudo_data = pseudo_data, lambda_s = scan_value, lambda_b = lambda_b, \
                          gaussian_mean = gaussian_mean, gaussian_sigma = gaussian_sigma, \
                          x_min = x_min, x_max = x_max))
    
    # plot likelihood function
    pl.plot(l_scan,l_resu)
    pl.xlabel('$\lambda_s$ [event number]')
    pl.ylabel('negative log likelihood')
    pl.show()
    
    # find minimimum of likelihood function as a function of lambda_s and lambda_b, 
    # i.e. the parameter of interest (POI) and a nuisance parameter.
    # The minimization performed through optimize.fmin requires to parse a function whose 
    # first argument is a vector of the parameters to vary
    def fmin(par): 
        return nll(pseudo_data = pseudo_data, lambda_s = par[0], lambda_b = par[1], \
                   gaussian_mean = gaussian_mean, gaussian_sigma = gaussian_sigma, \
                   x_min = x_min, x_max = x_max )

    nll_abs_min = optimize.minimize(fmin, (1,1),bounds=((0, None), (0, None)))
   
    print(nll_abs_min)
    
#
# This is the main program
#
def RunOnMultipleDatasets(n_datasets):
    
    # define expectations
    lambda_s, lambda_b = 100.0, 100.0
    
    # define Gaussian mean and sigma
    gaussian_mean, gaussian_sigma =  10.0, 1.0
    
    # define energy range
    x_min, x_max =  0.0, 20.0
  
    print("Executing anlaysis for lambda_s = %d and lambda_b = %d" % (lambda_s, lambda_b))
    
    # array of values used to interpolate the pdf and cdf
    x = np.linspace(x_min, x_max, 10001)

    # build energy pdf 
    pdf_energy = PDF_Energy(x, \
                     lambda_s = lambda_s, lambda_b = lambda_b, \
                     gaussian_mean = gaussian_mean, gaussian_sigma = gaussian_sigma, \
                     x_min = x_min, x_max = x_max)
      
    # loop over multiple pseudo-data sets and store info in lambda_estimator
    lambda_s_estimator =[]
    lambda_b_estimator =[]

    for i in np.arange(n_datasets):
        
        if (i%1000)==0: print(i,"/",n_datasets,"trials done.")

        # generate pseudo-data
        pseudo_data = GeneratePseudoData(expectation = lambda_s + lambda_b, pdf = pdf_energy, xmesh = x)
        
        # find minimimum of likelihood function as a function of lambda_s and lambda_b, 
        # i.e. the parameter of interest (POI) and a nuisance parameter.
        # The minimization performed through optimize.fmin requires to parse a function whose 
        # first argument is a vector of the parameters to vary
        def fmin(par): 
            return nll(pseudo_data = pseudo_data, lambda_s = par[0], lambda_b = par[1], \
                       gaussian_mean = gaussian_mean, gaussian_sigma = gaussian_sigma, \
                       x_min = x_min, x_max = x_max, check_normalization=False)

        nll_abs_min = optimize.minimize(fmin, (100,100), \
                                        bounds=((0, None), (0, None)), \
                                        options={'disp': True})

        lambda_s_estimator.append(nll_abs_min.x[0])
        lambda_b_estimator.append(nll_abs_min.x[1])
   
    lambda_s_estimator = np.array(lambda_s_estimator)
    lambda_b_estimator = np.array(lambda_b_estimator)

    # create and plot hist of estimators
    pl.hist(lambda_s_estimator, normed=False)
    pl.xlabel('maximum likelihood estimator for $\lambda_s$ [counts]')
    pl.ylabel('pseudo-data sets')
    pl.show()
    
    pl.hist(lambda_b_estimator, normed=False)
    pl.xlabel('maximum likelihood estimator for $\lambda_b$ [counts]')
    pl.ylabel('pseudo-data sets')
    pl.show()
 
    print("median value of the estimator for lambda_s is: ", np.median(lambda_s_estimator))
    print("median value of the estimator for lambda_b is: ", np.median(lambda_b_estimator))
    
if __name__ == "__main__":
    np.random.seed(10)
    RunOnSingleDataset() 
    n_datasets=1000
    RunOnMultipleDatasets(n_datasets)
