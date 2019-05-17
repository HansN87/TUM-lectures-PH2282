'''
Statistical and Machine Learning Methods in Particle and Astrophysics

TUM - summer term 2019
M. Agostini <matteo.agostini@tum.de> and Hans Niederhausen <hans.niederhausen@tum.de>

Thanks to the following contributors:
    Martin Ha Minh (significant contributions to "IcecubeSim" a pseudo event generator for educational purposes)
'''

from simulator.icecube_sim import IcecubeSim


def generate_pseudoMC(nsamples, precomputed=False):

    # this method generates Monte Carlo datasets for the following combinations of neutrino and interaction types:
    #   electron neutrino CC DIS, electron neutrino NC DIS, muon neutrino NC DIS
    # all of them produce particle showers in a detector like IceCube
    #
    # the sole purpose of this code is to introduce the concept of predicting observable distributions
    #   from weighted MC events. This allows to account for complicated detector effects for which analytical, closed-form
    #   representations do not exist. When used for fitting (for example maximum likelihood) this is sometimes referred to as
    #   "forward folding".

    nsamples = int(nsamples)

    outfile_dict = {}
    for ptype in ["e", "mu"]:
        for pint in ["NC", "CC"]:
            # ignore numu CC events:
            if ptype == "mu" and pint == "CC":
                continue

            # create a dictionary with simulation arguments
            pars = {}
            pars['particle_type'] = 'nu'
            pars['flavor'] = ptype
            pars['interaction_type'] = pint
            
            # events are simulated with neutrino energies in the range from 1 TeV to 10 PeV 
            # ( log10( 1 TeV / 1 GeV) = 3.0 and log10 ( 10 PeV / 1 GeV) = 10.0
            pars['lEmin'] = 3. 
            pars['lEmax'] = 7.

            # primary energies are generated from a powerlaw with a "hard" spectral index gamma
            # f(E) ~ E^{-gamma} 
            pars['gamma'] = 1.1
            
            pars['neutrino_type'] = pars['particle_type'] + pars['flavor'] 
                
            outfile = './output/'+pars['neutrino_type']+'_'+pars['interaction_type']+'_simulation_gamma_'+str(pars['gamma'])+'_lEmin_'+str(pars['lEmin'])+'_lEmax_'+str(pars['lEmax'])+'_nsamples_'+str(nsamples)+'.h5'
       
            # run simulation unless it already exists
            # (events are stored in pandas tables)
           
            if not precomputed:
                sim = IcecubeSim(pars)
                sim.run_simulation(nsamples=nsamples, outfile=outfile)

            outfile_dict[pars['neutrino_type']+'_'+pars['interaction_type']]=outfile

    return outfile_dict
