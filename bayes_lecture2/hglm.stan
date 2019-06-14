data {

  int N; // number of observations
  int N_groups; // number of groups
  int groups[N]; // group index

  vector[N] x1; // x data
  vector[N] y; // y data

  real<lower=0> sigma; // error

  int N_model; // for plotting

  vector[N_model] x_model; // for plotting
  
  

  

}

parameters {

  vector[N_groups] beta0; // group level intercept
  real beta1; // population level slope

  real mu_beta; // mean of group intercept distribtuion
  real<lower=0> sigma_beta; // spread in group level distribution
  
}


transformed parameters {
  vector[N] mu; // likelihood mean


  mu = beta0[groups] + beta1 * x1;
  
}


model {

  beta1 ~ normal(0, 5); 


  // hyper priors for intercept
  mu_beta ~ normal(0, 10);
  sigma_beta ~ cauchy(0, 2.5);

  // group level prior
  beta0 ~ normal(mu_beta, sigma_beta);

  // likelihood
  y ~ normal(mu,sigma);

}


generated quantities {

  vector[N_model] lines[N_groups];

  for (n in 1:N_groups) {


    lines[n] = beta0[n] + beta1 * x_model;
    

  }


  

}
