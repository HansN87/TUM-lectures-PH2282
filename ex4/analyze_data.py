'''
Statistical and Machine Learning Methods in Particle and Astrophysics
TUM - summer term 2019
M. Agostini <matteo.agostini@tum.de> and Hans Niederhausen@tum.de <hans.niederhausen@tum.de>

Ex 4, "A realistic maximum likelihood analysis: 
        The discovery of a diffuse flux of high-energy astrophysical neutrinos by IceCube"

    learning goals: perform a maximum likelihood analysis with binned observations
    1) build a (complex) statistical model from events generated with Monte Carlo techniques
    2) generate data (counts) from the model prediction
    3) extract estimated (MLE) model parameters from the generated data
    4) perform hypothesis test (MLR): H0: bkg only - against - H1: bkg + signal 
        4a) calculate observed test-statistic value
        4b) repeat steps 2-3-4 and build probability distribution of the test statistic for the bkg only case
        4c) calculate p-value (evidence against bkg only case)
       
the example is motivated by the following papers: 
Phys. Rev. Lett. 113, 101101 (2014)
Phys. Rev. D 91, 022001 (2015)

Many thanks to Martin Ha Minh for his contributions to the pseudo IceCube MC generator.
'''

from generate_mcevents import generate_pseudoMC
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize

class diffuse_analysis(object):

    def __init__(self, infiles, seed=0):
        '''
        arguments
        infiles := python dict with keys corresponding to the 
                  three neutrino components that were simulated
                  the values must contain the paths to the 
                  corresponding simulation files on disk

        seed := random number seed for numpy.random
        '''

        # load the different MC events from disk
        self.df_nueNC = pd.read_hdf(infiles['nue_NC'])
        self.df_nueCC = pd.read_hdf(infiles['nue_CC'])
        self.df_numuNC = pd.read_hdf(infiles['numu_NC'])
        
        # combine dataframes
        self.df = pd.concat([self.df_nueNC, self.df_nueCC, self.df_numuNC])

        # assume 3 years of data taking as default
        self.ltime = 3. * 365 * 24 * 3600

        # we increase the conv background by a factor of 2, since the assumption of no contribution from numu CC
        # is overly optimistic
        self.df['conv_flux'] = self.df['conv_flux'] * 2.0

        self.gen_weight = self.df['generation_weight'].values

        # log10 reconstructed energy
        self.log_erec = np.log10(self.df['rec_energy'])

        np.random.seed(seed)


    def set_ltime(self, ltime):
        self.ltime = ltime * 365 * 24 * 3600


    def get_astro_flux(self, neutrino_energies, gamma):
        '''
        arguments
        neutrino_energies := vector of mc neutrino energies
        gamma := astrophysical spectral index (float)
        '''
        # the flux of astrophysical neutrinos is assumed to follow a powerlaw
        # the normalization uses units of 1/GeV 1/cm^2 1/s 1/sr
        return 1.e-18 * np.power(neutrino_energies/1.e5,-gamma) 


    def get_expectations(self, bins, pars):
        '''
        arguments
        bins := vector of bin edges ( len(bins) = number_of_bins + 1 )
        pars := vector {conv_norm, astro_norm, gamma}
        '''
        conv_norm, astro_norm, astro_gamma = pars

        # the total neutrino flux is the sum of conventional atmospheric neutrino background
        # and astrophysical signal
        total_flux =  astro_norm * self.get_astro_flux(self.df['prim_energy'].values, astro_gamma) 
        total_flux +=  conv_norm * self.df['conv_flux'].values
        
        # to calculate correct weights we have to take into account the generation pdfs used in
        # the original MC simulation
        mc_weights = self.gen_weight * self.ltime * total_flux

        # we get the prediction in each bin as the sum over all weights that fall in this bin
        mus, bins = np.histogram(self.log_erec, bins=bins, weights=mc_weights)
        return mus


    def generate_pseudo_data(self, mus):
        '''
        arguments
        mus := vector of poisson mean in each bin (model expectation)
        '''
        # generating pseudo data (counts) for a binned experiment is easy.
        # one can simply draw from a poisson distribution with mean
        # given by the model expectation
        
        return np.random.poisson(mus)


    def get_logllh(self, bins, data, pars):
        '''
        arguments
        bins := vector of bin edges ( len(bins) = number_of_bins + 1 )
        data := vector of observed counts in each bin
        pars := vector {conv_norm, astro_norm, gamma}
        '''

        # get poisson means for set of model parameters
        mus = self.get_expectations(bins, pars)
        
        # in principle one would do this
        # logl = poisson.logpmf(k = data, mu = mus)

        # however we can omit the factorial in the poisson
        # by calculationg the logpmf manually instead

        # ensure that expectation is non-zero in all bins
        # (we only have finite MC statistics to compute this expectation)
        # (in the limit of inifinite MC events no bin would ever be empty)
        eps = 1.e-10
        idx = np.where(mus<eps)[0]
        mus[idx] = np.ones(len(mus[idx])) * eps

        logl = data * np.log(mus) - mus

        return np.sum(logl)

    def get_full_fit(self, bins, data, seed):
        '''
        arguments
        bins := vector of bin edges ( len(bins) = number_of_bins + 1 )
        data := vector of observed counts in each bin
        seed := vector of parameter values used as seed (conv_norm, astro_norm, astro_index)    
        '''
        # we minimize the negative log-likelihood function
        # this is done simultaenously for all parameters
        fmin = lambda pars: -1.0 * self.get_logllh(bins, data, pars)
        result = minimize(fmin, seed,bounds=((0, None), (0, None), (0, None)))

        # return bestfit parameters and value of -logL at the minimum
        return result.x, result.fun

    def get_restricted_fit(self, bins, data, seed, pars_H0):
        '''
        arguments
        bins := vector of bin edges ( len(bins) = number_of_bins + 1 )
        data := vector of observed counts in each bin
        seed := vector (length 1) of parameter values used as seed (conv_norm)  
        pars_H0 := vector (length 2) of constant parameters (astro_norm, astro_gamma)
        '''

        # to calculate the restricted fit
        # we only perform the minimization in one parameter (here conv_norm)
        
        # but to define the hypothesis we also need the values of the constant parameters
        astro_norm_H0, astro_gamma_H0 = pars_H0         
    
        def fmin(pars):
            # here pars is a vector of length 1 only
            conv_norm = pars[0]

            # need to extend by the constant parameter values to obtain logllh function
            pars = [conv_norm, astro_norm_H0, astro_gamma_H0]
            return -1.0 * self.get_logllh(bins, data, pars)

        result = minimize(fmin, seed, bounds=[(0, None)])   
        return result.x, result.fun

    def get_test_statistic(self, bins, data, seed_full, seed_restricted, pars_H0):
        '''
        arguments
        bins := vector of bin edges ( len(bins) = number_of_bins + 1 )
        data := vector of observed counts in each bin
        seed_full := vector of parameter values used as seed (conv_norm, astro_norm, astro_index) 
        seed_restricted := vector (length 1) of parameter values used as seed (conv_norm) 
        pars_H0 := vector (length 2) of constant parameters (astro_norm, astro_gamma)
        '''
        # perform both fits (full set of parameters and restricted set of parameters) to calculate the TS
        # TS = -2 log (L0 /L1)
        best_fit, best_neglogl = self.get_full_fit(bins, data, seed_full)
        restricted_fit, restricted_neglogl = self.get_restricted_fit(bins, data, seed_restricted, pars_H0)
        return best_fit, restricted_fit.tolist()+pars_H0, 2 * (restricted_neglogl - best_neglogl)


    def get_distributions_from_pseudo_data(self, bins, seed_full, seed_restricted, pars_H0, sim_pars, nsamples):
        '''
        arguments
        bins := vector of bin edges ( len(bins) = number_of_bins + 1 )
        data := vector of observed counts in each bin
        seed_full := vector of parameter values used as seed (conv_norm, astro_norm, astro_index) 
        seed_restricted := vector (length 1) of parameter values used as seed (conv_norm) 
        pars_H0 := vector (length 2) of constant parameters (astro_norm, astro_gamma)
        sim_pars := vector of parameters used to generate pseudo data
        nsamples := number of samples to generate (integer)
        '''
        # repeatedly perform TS calculation for random datasets and store results
        bestfit_results = []
        restricted_fit_results = []
        ts_values = []
        for i in range(nsamples):   
            mus_H0 = self.get_expectations(bins, sim_pars)
            data_H0 = self.generate_pseudo_data(mus_H0)

            x, y, z = self.get_test_statistic(bins, data_H0, seed_full, seed_restricted, pars_H0)
            bestfit_results.append(x)
            restricted_fit_results.append(y)
            ts_values.append(z)
            if (i%10)==0:
                print "analyzed", i, "/", nsamples, "datasets."

        return bestfit_results, restricted_fit_results, ts_values
            
            
            


    def get_expectations_conv(self, bins, pars):
        conv_norm = pars
        conv_flux =  conv_norm * self.df['conv_flux'].values
        mc_weights = self.gen_weight * self.ltime * conv_flux
        mus, bins = np.histogram(self.log_erec, bins=bins, weights=mc_weights)
        return mus


    def get_expectations_astro(self, bins, pars):
        astro_norm, astro_gamma = pars
        astro_flux =  astro_norm * self.get_astro_flux(self.df['prim_energy'].values, astro_gamma)
        mc_weights = self.gen_weight * self.ltime * astro_flux
        mus, bins = np.histogram(self.log_erec, bins=bins, weights=mc_weights)
        return mus


    def get_poisson_confidence_intervals(self, counts):
        feldmann_cousins = { 0:(0.00, 1.29),
                             1:(0.37, 2.75),
                             2:(0.74, 4.25),
                             3:(1.10, 5.30),
                             4:(2.34, 6.78),
                             5:(2.75, 7.81),
                             6:(3.82, 9.28),
                             7:(4.25, 10.30),
                             8:(5.30, 11.32),
                             9:(6.33, 12.79),
                             10:(6.78, 13.81),
                             11:(7.81, 14.82),
                             12:(8.83, 16.29),
                             13:(9.28, 17.30),
                             14:(10.30, 18.32),
                             15:(11.32, 19.32),
                             16:(12.33, 20.80),
                             17:(12.79, 21.81),
                             18:(13.81, 22.82),
                             19:(14.82, 23.82),
                             20:(15.83, 25.30) }
        
        data_vals = counts
        data_errs=[]
        for vals in data_vals:
            val = vals
            if val == 0:
                data_errs.append( (0.0, 0.0) )
            elif (val > 0 and val < 21):
                #print feldmann_cousins[int(val)]
                data_errs.append( feldmann_cousins[int(val)] )
            else:
                data_errs.append( (val - np.sqrt(val), val + np.sqrt(val) ) )
        
        data_elow, data_eup = zip(*data_errs)
        data_elow = np.asarray(data_elow)
        data_eup = np.asarray(data_eup)
    
        data_elow = data_vals- data_elow
        data_eup = data_eup - data_vals

        return (data_elow, data_eup)




    
