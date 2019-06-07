data {

  int<lower=0> N; // number of data points
  vector[N] x_obs; // x observations
  vector[N] y_obs; // y observations  
  int<lower=0> N_model; // number of data points for line
  vector[N_model] x_model; //where to evaluate the model

}


transformed data {
  vector[N] x_std;
  vector[N] y_std;

  real x_mean = mean(x_obs);
  real x_sd = sd(x_obs);

  real y_mean = mean(y_obs);
  real y_sd = sd(y_obs);

  
  
  x_std = (x_obs - x_mean) / x_sd;
  y_std = (y_obs - y_mean) / y_sd;
  
}




parameters {
  
  real m_std; //slope of the line
  real b_std; //intercept of the line
  real<lower=0> sigma_std;
  
}

transformed parameters {

  // latent y values not obscured by measurement error
  vector[N] y_std_true = m_std * x_std + b_std;

}

model {

  // weakly informative priors

  m_std ~ normal(0,5);
  b_std ~ normal(0,5);
  sigma_std ~ cauchy(0, 5.);

  
  // likelihood

  y_std ~ normal(y_std_true, sigma_std);

  

}

generated quantities {

  real m;
  real b;
  real sigma;
  
  vector[N] ppc;
  vector[N] y_true;
  vector[N_model] line;

  sigma = y_sd * sigma_std;
  
  m = m_std * y_sd / x_sd;
  b = y_sd * (b_std - m_std * x_mean / x_sd)
    + y_mean;



  y_true = m * x_obs +b; 
  
  // generate the posterior of the
  // fitted line
  line = m * x_model + b;

  // create posterior samples for PPC
  for (n in 1:N) {
    
    ppc[n] = normal_rng(m * x_obs[n] + b, sigma);

  }
  

}
