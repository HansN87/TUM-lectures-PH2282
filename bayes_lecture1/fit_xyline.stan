data {

  int<lower=0> N; // number of data points
  vector[N] x_obs; // x observations
  vector[N] y_obs; // y observations  
  real<lower=0> sigma_x; // homoskedastic measurement error
  real<lower=0> sigma_y; // homoskedastic measurement error

  
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
  real sigma_y_std = sigma_y / y_sd;
  real sigma_x_std = sigma_x / x_sd;
  
  
  x_std = (x_obs - x_mean) / x_sd;
  y_std = (y_obs - y_mean) / y_sd;
  
}




parameters {
  
  real m_std; //slope of the line
  real b_std; //intercept of the line
  vector[N] x_latent_std;
  
}

transformed parameters {

  // latent y values not obscured by measurement error
  vector[N] y_std_true = m_std * x_latent_std + b_std;

}

model {

  // weakly informative priors

  m_std ~ normal(0,1);
  b_std ~ normal(0,1);
  x_latent_std ~ normal(0,1);

  // likelihood

  y_std ~ normal(y_std_true, sigma_y_std);
  x_std ~ normal(x_latent_std, sigma_x_std);

  

}

generated quantities {

  real m;
  real b;
  
  
  vector[N] ppc_x;
  vector[N] ppc_y;
  
  vector[N_model] line;

  m = m_std * y_sd / x_sd;
  b = y_sd * (b_std - m_std * x_mean / x_sd)  + y_mean;



    // generate the posterior of the
  // fitted line
  line = m * x_model + b;

  // create posterior samples for PPC
  for (n in 1:N) {

    real x_std_new = normal_rng(x_latent_std[n], sigma_x_std);
    real y_std_new = normal_rng( m_std * x_latent_std[n] + b_std , sigma_y_std);
    
    ppc_y[n] = sigma_y * y_std_new + y_mean;
    ppc_x[n] = sigma_x * x_std_new + x_mean;

  }
  

}
