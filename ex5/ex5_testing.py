'''
Statistical and Machine Learning Methods in Particle and Astrophysics

TUM - summer term 2019
M. Agostini <matteo.agostini@tum.de> and Hans Niederhausen@tum.de <hans.niederhausen@tum.de>

Ex 1, conceptual steps
    1) build a model 
    2) build a data set by sampling the model and storing the events in a hist
    3) plot model and pseudo-data set for a given background and signals rate
    4) compute the maximum likelihood estimator (MLE) for the signal fraction p_s = lambda_s / N 
    5) calculate the TS distributions for the hypothesis tests given on slide 37
    6) for each tests one should observe convergenve to chi-sq distributions as function of the sample size
       (dof=1 for first test, dof=2 for second test)
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
from matplotlib.pyplot import cm

gaussian_mean = 10.0
gaussian_sigma = 1.0
x_min = 0.0
x_max = 20.0


def pdf(x, p_s, gaussian_mean, gaussian_sigma, xmin, xmax, truncated_gaus_norm = 0):  
    if not truncated_gaus_norm:
        # normalize the truncated gaussian
        truncated_gaus_norm = norm.cdf(xmax, gaussian_mean, gaussian_sigma) - norm.cdf(xmin, gaussian_mean, gaussian_sigma)
    return p_s * norm.pdf(x, gaussian_mean, gaussian_sigma) / truncated_gaus_norm + (1. - p_s) * 1./(xmax - xmin)

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

    # calculate truncated gaus norm
    tgnorm = norm.cdf(xmax, gaussian_mean, gaussian_sigma) - norm.cdf(xmin, gaussian_mean, gaussian_sigma)
    pdf_vals = pdf(x, p_s, gaussian_mean, gaussian_sigma, xmin, xmax, truncated_gaus_norm = tgnorm)
   
    # fix underflow (otherwise numeric problems for p_s -> 1) when assuming true_p_s = 0.1
    eps = 1.e-20
    idx = np.where(pdf_vals < eps)[0]
    if len(x[idx])>0:
        pdf_vals[idx] = eps
   
    logl = np.log(pdf_vals)
 
    return -np.sum( logl )
    
true_p_s = 0.2 # (the probability to generate a signal event is 10%)
true_gaussian_mean = 10.

tss_dof1_all = []
tss_dof2_all = []


nsvals0 = (np.asarray(range(8))+2).tolist()
nsvals1 = ((np.asarray(range(20))+1)*5).tolist()[1:]

sample_sizes = nsvals0+nsvals1
print sample_sizes

for nsamples in sample_sizes:
    tss_dof1 = []
    tss_dof2 = []
    print "currently doing N=", nsamples
    for k in range(100000): 
        pseudo_data = BuildDataset(nsamples, p_s = true_p_s, gaussian_mean = true_gaussian_mean)
        
        # minimize in p_s and gaussian_mean
        fmin = lambda pars: nll(pseudo_data,pars[0], pars[1], gaussian_sigma, x_min, x_max)
        result = optimize.minimize(fmin, [true_p_s, true_gaussian_mean], bounds=[(0., 1.), (x_min, x_max)])
        logl1 = result.fun
        
        # minimize gaussian_mean assuming p_s = ps_true = 0.2
        fmin = lambda g_mean: nll(pseudo_data, true_p_s, g_mean, gaussian_sigma, x_min, x_max)
        result = optimize.minimize(fmin, [true_gaussian_mean], bounds=[(x_min, x_max)])

        logl0 = result.fun
    
        # calculate the TS for hypothesis test 1: H0:p_s = 0.2 vs H1:p_s \neq 0.2
        # this will have distribution of chisq(dof=1) if N is large
        # see slide 35 of lecture 5
        ts = 2 * (logl0 - logl1)
        tss_dof1.append(ts)
    
    
        # eval neglogl for gaussian_mean = gaussian_mean_true. p_s = p_s_true
        logl0 = nll(pseudo_data, true_p_s, true_gaussian_mean, gaussian_sigma, x_min, x_max)

        # calculate the TS for hypothesis test 2: H0: p_s = 0.2, mean = 10.0 vs H1: p_s and mean are different from H0
        # this will have distribution of chisq(dof=2) if N is large
        # see slide 38 of lecture
        ts = 2 * (logl0 - logl1)
        tss_dof2.append(ts)

    tss_dof1_all.append(tss_dof1)
    tss_dof2_all.append(tss_dof2)


import matplotlib.pyplot as plt
from scipy.stats import chi2

xmax_p = 20.0
xmin_p = 0.0
nbins = 50

# do all plots
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
xvals = np.linspace(xmin_p, xmax_p, 1000)
yvals = chi2.pdf(xvals, 1)
edges = np.linspace(xmin_p, xmax_p, nbins+1)
centers = 0.5*(edges[1:]+edges[:-1])

color=iter(cm.rainbow(np.linspace(0,1,len(sample_sizes))))
for obj in zip(tss_dof1_all, sample_sizes):
    c = next(color)
    tss_dof1, ns = obj
    counts, edges = np.histogram(tss_dof1, bins=edges, density=True) 
    ax.plot(centers, counts, "r-", label="N=%d" %(ns), color=c, alpha=0.5)

ax.plot(xvals, yvals, "k--", label="chi-sq(k=1)", zorder=50, linewidth=2)
ax.set_ylabel("counts", fontsize=18)
ax.set_xlabel("TS-1", fontsize=18)
for axis in ['top','bottom','left','right']: 
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.0')
ax.tick_params(labelsize=18, axis='both', which='both', width=1.5, colors='0.0')
ax.set_xlim([0.0, 20.0])
ax.set_ylim([1.e-5, 10.0])
ax.set_yscale('log')
ax.legend(bbox_to_anchor=[1.02, 1.02], loc='upper left', prop={'size':9}, ncol=1, fancybox=True, frameon=True)
plt.tight_layout()
plt.savefig("TS1.png")


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
xvals = np.linspace(xmin_p,xmax_p, 1000)
yvals = chi2.pdf(xvals, 2)
edges = np.linspace(xmin_p, xmax_p, nbins+1)
centers = 0.5*(edges[1:]+edges[:-1])

color=iter(cm.rainbow(np.linspace(0,1,len(sample_sizes))))
for obj in zip(tss_dof2_all, sample_sizes):
    c = next(color)
    tss_dof2, ns = obj
    counts, edges = np.histogram(tss_dof2, bins=edges, density=True)
    ax.plot(centers, counts, "r-", label="N=%d" %(ns), color=c, alpha=0.5)


ax.plot(xvals, yvals, "k--", label="chi-sq(k=2)", zorder=50, linewidth=2)
ax.set_ylabel("pdf", fontsize=18)
ax.set_xlabel("TS-2", fontsize=18)
for axis in ['top','bottom','left','right']: 
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.0')
ax.tick_params(labelsize=18, axis='both', which='both', width=1.5, colors='0.0')
ax.set_xlim([0.0, 20])
ax.set_ylim([1.e-5, 10.0])
ax.set_yscale('log')
ax.legend(bbox_to_anchor=[1.02, 1.02], loc='upper left', prop={'size':9}, ncol=1, fancybox=True, frameon=True)
plt.tight_layout()
plt.savefig("TS2.png")


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
xvals = np.linspace(xmin_p, xmax_p, 1000)
yvals = chi2.pdf(xvals, 1)
edges = np.linspace(xmin_p, xmax_p, nbins+1)
centers = 0.5*(edges[1:]+edges[:-1])


for obj in zip(tss_dof1_all, sample_sizes)[-1:]:
    tss_dof1, ns = obj
    counts, edges = np.histogram(tss_dof1, bins=edges, density=True)
    ax.plot(centers, counts, "r-", label="N=%d" %(ns), color='red', alpha=1.0)

ax.plot(xvals, yvals, "k--", label="chi-sq(k=1)", zorder=50, linewidth=2)
ax.set_ylabel("counts", fontsize=18)
ax.set_xlabel("TS-1", fontsize=18)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.0')
ax.tick_params(labelsize=18, axis='both', which='both', width=1.5, colors='0.0')
ax.set_xlim([0.0, 20.0])
ax.set_ylim([1.e-5, 10.0])
ax.set_yscale('log')
ax.legend(bbox_to_anchor=[1.02, 1.02], loc='upper left', prop={'size':9}, ncol=1, fancybox=True, frameon=True)
plt.tight_layout()
plt.savefig("TS1_last.png")


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
xvals = np.linspace(xmin_p,xmax_p, 1000)
yvals = chi2.pdf(xvals, 2)
edges = np.linspace(xmin_p, xmax_p, nbins+1)
centers = 0.5*(edges[1:]+edges[:-1])


for obj in zip(tss_dof2_all, sample_sizes)[-1:]:
    tss_dof2, ns = obj
    counts, edges = np.histogram(tss_dof2, bins=edges, density=True)
    ax.plot(centers, counts, "r-", label="N=%d" %(ns), color='red', alpha=1.0)


ax.plot(xvals, yvals, "k--", label="chi-sq(k=2)", zorder=50, linewidth=2)
ax.set_ylabel("pdf", fontsize=18)
ax.set_xlabel("TS-2", fontsize=18)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
    ax.spines[axis].set_color('0.0')
ax.tick_params(labelsize=18, axis='both', which='both', width=1.5, colors='0.0')
ax.set_xlim([0.0, 20])
ax.set_ylim([1.e-5, 10.0])
ax.set_yscale('log')
ax.legend(bbox_to_anchor=[1.02, 1.02], loc='upper left', prop={'size':9}, ncol=1, fancybox=True, frameon=True)
plt.tight_layout()
plt.savefig("TS2_last.png")

