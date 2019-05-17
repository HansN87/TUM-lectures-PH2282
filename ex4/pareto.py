#!/usr/bin/env python

import numpy as np

class truncated_pareto(object):
    def __init__(self, gamma, xmin, xmax):
        self.gamma = gamma
        self.xmin = xmin
        self.xmax = xmax

        # calculate normalization and other useful terms
        if self.gamma != 1.0:
            self.int_gamma = 1.-self.gamma
            self.norm = 1./self.int_gamma*(self.xmax**self.int_gamma-self.xmin**self.int_gamma)
            self.norm = 1./self.norm

            self.cdf_factor = self.norm / self.int_gamma
            self.cdf_const = self.cdf_factor * (-self.xmin**self.int_gamma)

            self.inv_cdf_factor = self.norm**(-1)*self.int_gamma
            self.inv_cdf_const = self.xmin**self.int_gamma
            self.inv_cdf_gamma = 1./self.int_gamma

        else:
            self.norm = 1./np.log(self.xmax/self.xmin)
        
        
        

    def pdf(self, x):
        val = np.power(x, -self.gamma) * self.norm 

        if not isinstance(x, np.ndarray):
            if x<self.xmin or x>self.xmax:
                return 0.0 
            else:
                return val
        
        else:
            idx = np.logical_or(x<self.xmin, x>self.xmax)
            val[idx] = np.zeros(len(val[idx])) 
            return val


    def cdf(self, x):
        if self.gamma==1:
            val = self.norm * np.log(x/self.xmin)
        else:
            val = self.cdf_factor * np.power(x, self.int_gamma) + self.cdf_const
    
        if not isinstance(x, np.ndarray):
            if x<self.xmin:
                return 0.0
            if x>self.xmax:
                return 1.0
            else:
                return val


        else:
            idx = x<self.xmin
            val[idx] = np.zeros(len(val[idx]))
            idx = x>self.xmax
            val[idx] = np.ones(len(val[idx])) 
            return val 


    def inv_cdf(self, x):
        if self.gamma==1:
            return self.xmin * np.exp(x/self.norm)
        else:
            return np.power((x * self.inv_cdf_factor)+self.inv_cdf_const, self.inv_cdf_gamma)


    def samples(self, nsamples):
        u = np.random.uniform(0,1,nsamples)
        return self.inv_cdf(u)
        
        
        
        
        

    
    
        

        
