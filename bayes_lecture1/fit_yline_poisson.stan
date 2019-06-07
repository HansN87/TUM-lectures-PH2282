data {

  int<lower=0> N; // number of data points
  vector[N] x_obs; // x observations
  int y_obs[N]; // y observations  

  int<lower=0> N_model; // number of data points for line
  vector[N_model] x_model; //where to evaluate the model

}






parameters {
  
  real m; //slope of the line
  real b; //intercept of the line

}

transformed parameters {

  // latent y values not obscured by measurement error
  vector[N] y_true = m* x_obs + b;

}

model {

  // weakly informative priors

  m ~ normal(0,5);
  b ~ normal(0,5);

  // likelihood

  y_obs~ poisson(y_true);

  

}

generated quantities {

  
  vector[N] ppc;
  
  vector[N_model] line;

  
  // generate the posterior of the
  // fitted line
  line = m * x_model + b;

  // create posterior samples for PPC
  for (n in 1:N) {
    
    ppc[n] = poisson_rng(y_true[n]);

  }
  

}
