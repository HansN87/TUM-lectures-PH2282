import stan_utility
import numpy as np
import pandas as pd


jla_data_set = pd.read_table('jla_lcparams.txt')


n_model = 200
z_model = np.linspace(0,1.4,n_model)



data = {}
data['N_sn'] = len(jla_data_set)
data['h0'] = 70.
data['zcmb'] = np.array(jla_data_set['zcmb'])

data['c_obs'] = np.array(jla_data_set['color'])
data['c_sigma'] = np.array(jla_data_set['dcolor'])

data['x1_obs'] = np.array(jla_data_set['x1'])
data['x1_sigma'] = np.array(jla_data_set['dx1'])

data['m_obs'] = np.array(jla_data_set['mb'])
data['m_sigma'] = np.array(jla_data_set['dmb'])
data['N_model'] = n_model
data['z_model'] = z_model


n_warmup=1000
n_samp = 250
iter = n_warmup + n_samp
n_chain = 4


model = stan_utility.compile_model('supernove_model.stan','sn_cosmo');


fit = model.sampling(
    data=data,
    iter=iter,
    warmup=n_warmup,
    chains=n_chain,
    n_jobs=n_chain,
    thin=1,
    seed=1234,
    control=dict(max_treedepth=13, adapt_delta=0.95))

stan_utility.stanfit_to_hdf5(fit, 'sncosmo_fit.h5')
