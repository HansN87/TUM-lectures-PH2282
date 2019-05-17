import numpy as np
import pandas as pd
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from pareto_sampler import pareto
from inelasticity_sampler import inelasticity_sampler
from aeff_calculator.aeff_calculator import effective_area
from direction_sampler import uniform_from_sphere, transform_to_cartesian, transform_to_spherical
from hadronic_factor import sample_hadronic_reduction_factor
from energy_dependencies import get_eres_exinterpolator, get_angres_exinterpolator
from vMF_sampler import VonMisesFisher

# get conventional flux weights
from conv_flux.atmo_nu import conv_flux


class IcecubeSim(object):
    def __init__(self, pars):
        self.particle_type = pars['particle_type'] # nu, nubar
        self.neutrino_type = pars['neutrino_type']
        self.flavor = pars['flavor']
        self.interaction_type = pars['interaction_type'] # NC, CC
        self.lEmin = pars['lEmin']
        self.lEmax = pars['lEmax']
        self.gamma = pars['gamma']

        self._p_sampler = pareto(gamma=self.gamma, lower=10**self.lEmin, upper=10**self.lEmax)
        inter = str(self.particle_type) + str(self.flavor) + '_' + str(self.interaction_type)
        self.aeff = effective_area(inter, 'cascades')
        self._inel = inelasticity_sampler()

        self._energy_resolution = get_eres_exinterpolator()
        self._angular_resolution = get_angres_exinterpolator()
        self._vMF = VonMisesFisher()


    def run_simulation(self, outfile, nsamples):
        # Sample primary information
        logging.info("Sampling primary information")

        prim_types = {'nue':12, 'nuebar':-12, 'numu':14, 'numubar':-14, 'nutau':16, 'nutaubar':-16}
        prim_type = prim_types[self.neutrino_type]

        interaction_types = {'NC':2.0, 'CC':1.0}
        int_type = interaction_types[self.interaction_type]

        neutrino_energies = self._p_sampler.sample_pareto(size=nsamples)
        neutrino_energies_log = np.log10(neutrino_energies)

        neutrino_directions = uniform_from_sphere(nsamples)
        neutrino_directions_cartesian = np.asarray(transform_to_cartesian(neutrino_directions))
        neutrino_cos_thetas = np.asarray(neutrino_directions[:,0])

        # Calculate weights
        logging.info("Calculating weights")
        effective_areas = np.power(10.,
                                   np.asarray([self.aeff.eval(lE, ct)
                                               for lE, ct in zip(neutrino_energies_log, neutrino_cos_thetas)])).flatten()
        energy_probabilities = self._p_sampler.pdf(neutrino_energies)
        weights = 1./nsamples * effective_areas / energy_probabilities * 4. * np.pi * 10.**4

        # Calculate bjorken y for inelasticties
        logging.info("Calculating inelasticities")
        interaction = str(self.particle_type) + '_' + str(self.interaction_type)
        bjorken_y = np.asarray([self._inel.draw_inelasticity(energy=E, interaction=interaction)
                                for E in neutrino_energies])

        # Intial hadronic cascade
        logging.info("Calculating deposited energies")
        e_had = bjorken_y * neutrino_energies
        hadr_factor = np.vectorize(sample_hadronic_reduction_factor)(e_had)
        deposited_energies = hadr_factor * e_had  
        
        if self.interaction_type == 'CC':
            deposited_energies += (np.ones(nsamples) - bjorken_y) * neutrino_energies
        else: # NC
            pass

        # Reconstructted quanitities
        logging.info("Sampling reconstructed quantities")
        reconstructed_energies = np.vectorize(self._get_reconstructed_energy)(deposited_energies)

        # get reconstructed directions
        kappas_old = self._get_kappa(deposited_energies)
        kappas = []
        for tk in kappas_old:
            if type(tk)==np.array:
                kappas.append(tk[0])
            else:
                kappas.append(tk)

        reconstructed_directions = []
        for obj in zip(neutrino_directions_cartesian.tolist(), kappas):
            mu, kappa = obj
            tdis = self._vMF.randomize_about_point(np.asarray([mu]), kappa=kappa, num_samples=1)
            reconstructed_directions.append(transform_to_spherical(tdis)[0])
        dirs = np.vstack(reconstructed_directions)

        # Save to dataframe
        to_store={}
        to_store["prim_energy"]=np.asarray(neutrino_energies)
        to_store["prim_coszenith"]=np.asarray(neutrino_directions[:,0])
        to_store["prim_azimuth"]=np.asarray(neutrino_directions[:,1])
        to_store["rec_energy"]=np.asarray(reconstructed_energies)
        to_store["rec_coszenith"]=np.asarray(dirs[:,0])
        to_store["rec_azimuth"]=np.asarray(dirs[:,1])
        to_store["generation_weight"]=np.asarray(weights)
        to_store["dep_energy"]=np.asarray(deposited_energies)
        to_store["bjorken_y"]=np.asarray(bjorken_y)
        to_store["prim_type"]=np.ones(len(neutrino_energies)) * prim_type
        to_store["interaction_type"]=np.ones(len(neutrino_energies)) * int_type

        names = to_store.keys()
        X = np.hstack(tuple([to_store[key].reshape(to_store[key].shape[0], -1) for key in names]))
        df = pd.DataFrame(X, columns=names)

        # now add conventional flux weights
        cf = conv_flux()
        cf.add_flux(df)

        # write to file
        store = pd.HDFStore(outfile)
        sim_component = str(self.particle_type) + '_' + str(self.interaction_type)
        store[sim_component]=df
        store.close()
        logging.info("Completed")

        return

    def _get_reconstructed_energy(self, edep):
        erec = -1.
        sigma = self._energy_resolution(edep)
        while erec < 0:
            erec = np.random.normal(edep, sigma)
        return erec

    def _get_kappa(self, edep):
        # angular error as function of energy deposit
        # deterministic
        err = self._angular_resolution(edep) / 180. * np.pi
        var = err**2
        kappa = 2.3 / var
        return kappa
