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
  vector[N_groups] beta1; // group level slope

  // hyper parameters
  real mu_beta0;
  real<lower=0> sigma_beta0;

  real mu_beta1;
  real<lower=0> sigma_beta1;

  
}


transformed parameters {
  vector[N] mu; // likelihood mean


  mu = beta0[groups] + beta1[groups] .* x1;
  
}


model {


  // hyoer priors
  mu_beta0 ~ normal(0, 10);
  sigma_beta0 ~ cauchy(0, 2.5);

  
  mu_beta1 ~ normal(0, 10);
  sigma_beta1 ~ cauchy(0, 2.5);

  
  beta0 ~ normal(mu_beta0, sigma_beta0);

  beta1 ~ normal(mu_beta1, sigma_beta1);
 
  y ~ normal(mu,sigma);

}


generated quantities {

  vector[N_model] lines[N_groups];

  for (n in 1:N_groups) {



    lines[n] = beta0[n] + beta1[n] * x_model;
    

  }


  

}
