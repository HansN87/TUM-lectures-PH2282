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
import numpy
import pylab as pl
from scipy import optimize
from scipy import stats
from scipy.stats import norm
from scipy.integrate import simps
from scipy.integrate import quad
from scipy.interpolate import interp1d

gaussian_mean = 10.0
gaussian_sigma = 1.0
x_min = 0.0
x_max = 20.0

# fix normalization of cramer rao
f = lambda x: norm.pdf(x, gaussian_mean, gaussian_sigma) 
truncated_gaus_norm = quad(f, x_min, x_max, points=[x_min, gaussian_mean, x_max])[0]

def pdf(x, p_s, gaussian_mean, gaussian_sigma, xmin, xmax):  
    return p_s * norm.pdf(x, gaussian_mean, gaussian_sigma) / truncated_gaus_norm + (1. - p_s) * 1./(xmax - xmin)


def partial_log_pdf(x, p_s, gaussian_mean, gaussian_sigma, xmin, xmax):
    fpdf = norm.pdf(x, gaussian_mean, gaussian_sigma) / truncated_gaus_norm
    bpdf = 1/(xmax-xmin)
    return (fpdf - bpdf) / ( p_s * fpdf + (1. - p_s) * bpdf)

def BuildDataset(N, p_s, xmin=x_min, xmax=x_max, gaussian_mean=gaussian_mean, gaussian_sigma=gaussian_sigma):

    # data are generated in two steps. First each event is randomly attributed
    # to the signal or background. Then the energy distribution is sampled form
    # the proper pdf
    #
    # First step generate random numbers between 0 and 1 and attribute each
    # event to signal or background using the weight between the expectation for
    # signal counts and the total
    rvars = numpy.random.uniform(0,1, N)

    idx_s = numpy.where(rvars <= p_s)[0]
    idx_b = numpy.where(rvars >  p_s)[0]

    # Second step. Store in "samples" the energy value of each event
    samples = numpy.zeros(N)
    # draw samples that are realized as background events from the background distribution 
    samples[idx_b]=numpy.random.uniform(xmin, xmax, len(idx_b))
    # draw samples that are realized as signal events from the signal distribution 
    sig_samples=numpy.random.normal(gaussian_mean, gaussian_sigma, len(idx_s))

    # remove events outside xmin, xmax
    within_bounds = False
    while not within_bounds: 
        idx_too_small = np.where(sig_samples<xmin)[0]
        idx_too_large = np.where(sig_samples>xmax)[0]
        idx = sorted(np.concatenate((idx_too_small, idx_too_large)))
        if len(idx)==0:
            within_bounds = True
        else:
            sig_samples[idx] = np.random.normal(gaussian_mean, gaussian_sigma, len(idx)) 

    samples[idx_s]=sig_samples

    return samples


def nll (x, p_s, gaussian_mean, gaussian_sigma, xmin, xmax):
    pdf_vals = pdf(x, p_s, gaussian_mean, gaussian_sigma, xmin, xmax)
   
    # fix underflow (otherwise numeric problems for p_s -> 1) when assuming true_p_s = 0.1
    eps = 1.e-20
    idx = np.where(pdf_vals < eps)[0]
    if len(x[idx])>0:
        pdf_vals[idx] = eps
   
    logl = np.log(pdf_vals)
 
    return -np.sum( logl )
    
true_p_s = 0.2 # (the probability to generate a signal event is 10%)
#nsamples = 100


ndatasets = 100000

mle_vars = []
crb_vars = []

nsvals0 = (np.asarray(range(8))+2).tolist()
nsvals1 = ((np.asarray(range(20))+1)*5).tolist()[1:]

for ns in nsvals0+nsvals1: 
    nsamples = ns 

    def get_results(true_p_s, nsamples):                   
        vals = []
        for i in range(ndatasets):
            pseudo_data = BuildDataset(nsamples, p_s = true_p_s)
        
            fmin = lambda p_s: nll(pseudo_data, p_s, gaussian_mean, gaussian_sigma, x_min, x_max)
            result = optimize.minimize(fmin, [0.2], bounds=[(0., 1.)])
            vals.append(result.x[0])
        return vals
    
    vals = get_results(true_p_s, nsamples)

    mean_central = np.average(vals)
    std = np.sqrt(np.var(vals))
    print "mean is", mean_central, "pm", np.sqrt(np.var(vals)/nsamples), np.sum(vals) / len(vals)
    print "std is", std
    
    # define the integrad for the calculation of the average value
    fint = lambda x: pdf(x, true_p_s, gaussian_mean, gaussian_sigma, x_min, x_max) * partial_log_pdf(x, true_p_s, gaussian_mean, gaussian_sigma, x_min, x_max)**2
    
    result = quad(fint, x_min, x_max, points=[x_min, gaussian_mean, x_max])
    print result
    
    avg_denom = result[0]
    
    # now we need to take care of the nominator
    # need numerical estimate of the derivative
    rel_precision = 0.1
    abs_precision = rel_precision * true_p_s
    
    mean_low = np.mean(get_results(true_p_s-abs_precision, nsamples))
    mean_high = np.mean(get_results(true_p_s+abs_precision, nsamples))
    
    xm = true_p_s
    xvals = np.array([true_p_s-abs_precision, true_p_s, true_p_s+abs_precision])
    yvals = np.array([mean_low, mean_central, mean_high])
    ym = np.mean(yvals)
    slope = np.sum( (xvals - xm) * (yvals - ym) ) / np.sum( (xvals - xm) ** 2)
    
    var_cramer_bound = slope**2 / (nsamples * avg_denom)
    std_cramer_bound = np.sqrt(var_cramer_bound)
    
    print "the cramer-rao bound for unbiased estimators in this problem with sample size", nsamples, "is:", std_cramer_bound
    
    import matplotlib.pyplot as plt
    edges = np.linspace(0,1,101)
    xvals = np.linspace(0,1,10000)
    yvals = norm.pdf(xvals, mean_central, std)
    plt.hist(vals, bins=edges, density=True, label="ps=0.2, nsamples=%d" %(nsamples))
    plt.plot(xvals, yvals, "r-", linewidth=2)
    plt.ylabel("counts", fontsize=18)
    plt.xlabel("best-fit ps", fontsize=18)
    plt.tick_params(labelsize=18)
    plt.xlim([0.0,1.0])
    plt.legend(fontsize=16)
    plt.axvline(x=mean_central, color='k', linestyle='dashed')
    plt.axvline(x=0.2, color='k', linestyle='solid')
    plt.tight_layout()
    plt.savefig("./pdf/result_%d.png" %(nsamples))
    plt.clf()

    mle_vars.append(std)
    crb_vars.append(std_cramer_bound)

plt.scatter(crb_vars, mle_vars, c='r')
plt.ylabel("MLE std.", fontsize=18)
plt.xlabel("CRB std.", fontsize=18)
plt.plot([0,np.amax(mle_vars)], [0, np.amax(mle_vars)], "k--")
plt.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig("summary.png")


     
