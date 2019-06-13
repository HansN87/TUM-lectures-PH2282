data {

  int N;
  int N_groups;
  int groups[N];

  vector[N] x1;
  vector[N] y;

  real<lower=0> sigma;

  int N_model;

  vector[N_model] x_model;
  
  

  

}

parameters {

  vector[N_groups] beta0;
  real beta1;

  real mu_beta;
  real<lower=0> sigma_beta;
  
}


transformed parameters {
  vector[N] mu;


  mu = beta0[groups] + beta1 * x1;
  
}


model {

  beta1 ~ normal(0, 5);


  mu_beta ~ normal(0, 10);
  sigma_beta ~ cauchy(0, 2.5);

  beta0 ~ normal(mu_beta, sigma_beta);

 
  y ~ normal(mu,sigma);

}


generated quantities {

  vector[N_groups] lines[N_model];

  for (n in 1:N_groups) {


    lines[n][:] = beta0[n] + beta1 * x_model;
    

  }


  

}
