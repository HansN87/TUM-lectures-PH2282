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
  vector[N_groups] beta1;

  real mu_beta0;
  real<lower=0> sigma_beta0;

  real mu_beta1;
  real<lower=0> sigma_beta1;

  
}


transformed parameters {
  vector[N] mu;


  mu = beta0[groups] + beta1[groups] .* x1;
  
}


model {



  mu_beta0 ~ normal(0, 10);
  sigma_beta0 ~ cauchy(0, 2.5);

  beta0 ~ normal(mu_beta0, sigma_beta0);


  mu_beta1 ~ normal(0, 10);
  sigma_beta1 ~ cauchy(0, 2.5);

  beta1 ~ normal(mu_beta1, sigma_beta1);

 
  y ~ normal(mu,sigma);

}


generated quantities {

  vector[N_model] lines[N_groups];

  for (n in 1:N_groups) {



    lines[n] = beta0[n] + beta1[n] * x_model;
    

  }


  

}
