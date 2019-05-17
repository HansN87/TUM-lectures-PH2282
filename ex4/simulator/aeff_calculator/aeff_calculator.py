#!/usr/bin/env python

from aeff_info import aeff_info_cascades, aeff_info_tracks
from scipy.interpolate import RectBivariateSpline 
import numpy as np

class effective_area(object):
    def __init__(self, interaction, event):
        '''
        interaction: "flavor_channel", e.g. "nue_nc" or "numu_cc"
        event: "cascades" or "tracks"
        '''

        if event == 'cascades':
            d = aeff_info_cascades()
        elif event == 'tracks':
            d = aeff_info_tracks()
        else:
            raise ValueError('Event type not recognized:', event)
        self.infile = d.infiles[interaction]
        self.smoothing = d.smoothing[interaction]
        self.interaction = interaction

        # define lower bound on effective area
        self.lower_bound_lE = -5.0
        self.lower_bound_hE = -3.0

        # will set effective area to zero outside (negative infinity)
        self.lE_limit_low = 3.0
        self.lE_limit_high = 7.0

        self.spline = None

        self.lE_vals = None
        self.ct_vals = None
        self.aeff_vals = None

        self.__create_spline__()

    def eval(self, lE, cos_zenith):
        '''
        return log10(Aeff) in m
        '''
        if not isinstance(lE, np.ndarray):
            if lE < self.lE_limit_low or lE > self.lE_limit_high:
                print "energy", lE, "outside bounds. returning neg. infty."
                return np.NINF

            elif cos_zenith < -1.0 or cos_zenith > 1.0:
                print "cos zenith", cos_zenith, "outside bounds. returning neg. infty."
                return np.NINF

            else:
                return self.spline(lE, cos_zenith)


        else:
            # evaluate on grid. don't check bounds.
            return self.spline(lE, cos_zenith)


    def __create_spline__(self):

        table = np.loadtxt(self.infile, skiprows=2)

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
        idx = xvals < self.lE_limit_low
        k = len(xvals[idx])-2
        xvals = xvals[k:]
        idx = xvals < self.lE_limit_high
        k = len(xvals[idx])+2
        xvals = xvals[:k] 
        yvals = np.unique(table[:,-1])
		

		# collect effective area from all observable bins
        for logE in xvals:
            for ct in yvals:
                idx = np.logical_and(table[:,-1]==ct, table[:,-2]==logE)
                asum = np.sum(table[idx][:,-4])
                if asum:
                    aeff = np.log10(np.sum(table[idx][:,-4]))
                else:
                    aeff = np.NINF
                
                if np.isinf(aeff):
                    if logE < 5.0:
                        aeff=self.lower_bound_lE
                    else:
                        aeff=self.lower_bound_hE
            
               
                aeff_arr.append(aeff)
                logE_arr.append(logE)
                ct_arr.append(ct)

        self.lE_vals = np.array(logE_arr)
        self.ct_vals = np.array(ct_arr)
        self.aeff_arr = np.array(aeff_arr)

        self.aeff_vals = self.aeff_arr
       


		# now we average over bins. i.e. map 2 bins into 1
        xvals_avg = (0.5*(xvals[1:]+xvals[:-1]))[::2] # only every second
        aeff_smooth = np.ones(len(xvals_avg)*len(yvals))
        aeff_smooth = aeff_smooth.reshape((len(xvals_avg), len(yvals)))
	
        
		# average along energy axis
        aeff_arr_shaped = np.asarray(aeff_arr).reshape((len(xvals), len(yvals)))
        for column_idx in range(len(yvals)):
            aeff_avg = aeff_arr_shaped[:, column_idx]
            aeff_avg = (0.5*(aeff_avg[1:]+aeff_avg[:-1]))[::2]
            aeff_smooth[:, column_idx]=np.array(aeff_avg)
        
        self.spline = RectBivariateSpline(xvals_avg, yvals, aeff_smooth, s=self.smoothing) 
        print "... creating effective area spline for interaction", self.interaction, ", done!"

		
		
		
		
		
 
        


        

       
