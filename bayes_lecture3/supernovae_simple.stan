
functions {

  real integrand(real z, real Om, real Ode, real wp1){
    /*
      Intergrand for flat cosmology.


     */
    real zp1 = z+1;
    real Hsquare = Om * pow(zp1,3) + Ode *pow(zp1, 3*wp1);
    
    return  inv_sqrt(Hsquare);


  }


  real distance_modulus(real z, real Om, real Ode, real wp1, real h0){
    /*
      The distance modulus in magnitude


     */


    // set up for integral
    int N_itr = 10; 
    real N_div = 10.;
    real z0 = 0.; 
    real h= (z-z0)/N_div; 
    real s_sum=0.5 * integrand(z0, Om, Ode, wp1);

    real dl; // the luminosity distance

    
    // simple trapazoid rule
    for(n in 1:N_itr)
      {
        s_sum +=  integrand(z0 +n*h, Om, Ode, wp1);
      }
    
    s_sum += 0.5*integrand(z, Om, Ode, wp1);

    s_sum *= h;

    // compute the luminosity distance in Mpc
    dl = (1+z) * (299792458.0*1E-3/ h0) * s_sum;


    // convert to mag
    return 5*log10(dl)+25;

  }


}



data{
    
  
  int<lower=0> N_sn;    // number of supernovae
  real<lower=0> h0;     // the hubble constant
  vector[N_sn] zcmb;    //  redshifts
  vector[N_sn] m_obs;   // observed m
  vector[N_sn] m_sigma; // observed m err


  int N_model;      // number of model evaluations
  vector[N_model] z_model; // redshift for generating curves

  
}    
   

parameters {

    
  real<lower=0, upper=1> Om; // yes, I do not have to explain
  real<lower=-20.,upper=-18.> M0; // intrinsic brightness
  real<lower=-3., upper=-0.> tau_log; // the log of tau
  real<lower=-2, upper=-0.0> w; // the dark energy parameter 
    
        
  vector<lower=-10, upper=10>[N_sn] x1_latent;      // latent stretch
  vector<lower=-5, upper=5>[N_sn] c_latent;      // latent color shift
    
  vector[N_sn] mb; // latent mb
    
   
  
}



transformed parameters {
    

  real tau;  // brightness spread 
  real Ode = 1-Om; // dark matter
  real wp1 = w+1; // precompute for speed

  vector[N_sn] dist_mods_latent; // latent distance moduli
  vector[N_sn] m_latent; // latent mp
  
  tau = pow(10.,tau_log);

  // compute the latent distance moduli
  // and vectorize later
  
  for(n in 1:N_sn){

    dist_mods_latent[n] = distance_modulus(zcmb[n], Om, Ode, wp1, h0);
 
  }

  // compute the latent mb
  
  m_latent = M0 +  dist_mods_latent;
    
}




model {

  M0 ~ normal(-20, 5);
  
  // intrinsic scatter
  
  mb ~ normal( m_latent, tau);

  // observation model
  
  m_obs ~ normal(mb, m_sigma);

} 

generated quantities {

  vector[N_model] mb_curve;
  vector[N_model] mu_curve;


  // compute qauntites for plots
  
  for (n in 1:N_model) {

    mu_curve[n] = distance_modulus(z_model[n], Om, Ode, wp1, h0);


  }

  mb_curve = M0 + mu_curve;


}



