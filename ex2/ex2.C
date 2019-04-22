/*****************************************************************************
 * Statistical and Machine Learning Methods in Particle and Astrophysics
 *
 * TUM - summer term 2019
 * M. Agostini <matteo.agostini@tum.de>
 *
 * Ex 2, conceptual steps:
 *   1) build a model 
 *   2) build a data set by sampling the model and storing the events in a hist
 *   3) plot model and pseudo-data set for different background and signals rates
 *
 ****************************************************************************/


// This function builds a model as a TF1 and plot it
// sgnCts -> expectation for the signal cts
// bkgCts -> expectation for the bkg cts
TF1 BuildModel (const double bkgCts, const double sgnCts) {

   /*************************************************************************
    * Initialize the model as
    *    - a gaussian distribution for the signal 
    *    - a flat backgroud distribution
    *************************************************************************/
   // define parameters for the signal distribution, i.e. mean and sigma of the
   // guassian
   const double mean       =   0;
   const double sigma      =   1;

   // define parameters for the background distribution, i.e. the window edges
   const double rangeMin   = -10; 
   const double rangeMax   =  10;
   const double rangeWidth = rangeMax - rangeMin;

   // define binning
   const int    binNum     = 100;
   const double binWidth   = rangeWidth / binNum;

   // Coding-style note: all variables up to here are const, including the 
   // arguments parsed to the function -> this makes clear that these values 
   // cannot be changed and avoid unintentional modification of these variables

   // Build model 
   TF1 model ("modelGen", "([0]*(1/[2]) + [1]*(TMath::Gaus(x,[3],[4],1)))*[5]", rangeMin, rangeMax);
   model.SetParNames ("bkgCts","sgnCts", "rangeWidth", "mean","sigma", "binWidth");
   model.FixParameter(0, bkgCts);
   model.FixParameter(1, sgnCts);
   model.FixParameter(2, rangeWidth);
   model.FixParameter(3, mean);
   model.FixParameter(4, sigma);
   model.FixParameter(5, binWidth);

   // Set Properties of the TF1s: 
   // increase the number of points used for interpolation 
   model.SetNpx(1e4); 
   model.SetNpx(1e4);

   return model;
}

// This function builds a binned data set (histogram) given a model and
// count expectation 
// model -> model built using the BuildModel
// cts   -> expectation for the number of signal + background events 
//          (the ratio is specify by the model)
TH1D BuildDataset(TF1& model, const int cts) {

   /*************************************************************************
    * Initialize and generate data sets.
    * Here we use gRandom, which in the new ROOT version by default is a 
    * TRandom3 object.  TRandom3 is the recommended random number generator. 
    * For more info see:  https://root.cern.ch/doc/master/classTRandom.html
    *************************************************************************/
   const double xmin = model.GetXmin();
   const double xmax = model.GetXmax();

   // Initalize data set histogram
   TH1D dataset ("","",100,xmin,xmax);

   // Fill histogram
   // Randomly define the number of cts of the data set
   const int rndCts = gRandom->Poisson(cts);

   for (int i = 0; i < rndCts; i++) dataset.Fill(model.GetRandom(xmin,xmax));

   return dataset;
}


// Initialize a global object - i.e. the data set - to be used within the likelihood
TH1D globalDataset;

// Negative Log Likelikhood function. The arg of the function is fixed because
// of the minimizer. par[0] is our bkgCts and par[1] our sgnCts
double nll  (const double *par) {
   // initialize local model 
   // the static keyword means that the initialization is done only the first
   // time this line is executed
   static TF1 model = BuildModel(0,0);
   // define parameters of the model according to the value the Likelihood is computed for 
   model.FixParameter(0, par[0]);
   model.FixParameter(1, par[1]);

   // compute likelihood
   double sum = 0;
   for (int i = 1; i <= globalDataset.GetNbinsX(); i++) {
      const double energy = globalDataset.GetBinCenter(i);
      sum += -TMath::Log(TMath::Poisson(globalDataset.GetBinContent(i), model.Eval(energy)));
   }
   return sum;
}

// create data set and find MLE for background and signal expectations
void ex2_single_dataset (int numberCts = 1000) {

   // create minimizer and set its properties
   ROOT::Minuit2::Minuit2Minimizer minimizer (ROOT::Minuit2::kMigrad);
 
   minimizer.SetMaxFunctionCalls(1000000);
   minimizer.SetMaxIterations(100000);
   minimizer.SetTolerance(0.001);

   minimizer.SetVariable(0,"bkgCts",3./4.*numberCts, 1);
   minimizer.SetVariable(1,"sgnCts",1./4.*numberCts, 1);

   minimizer.SetVariableLowerLimit(0,1e-15);
   minimizer.SetVariableLowerLimit(1,1e-15);
 
   ROOT::Math::Functor f(&nll,2); 
   minimizer.SetFunction(f);


   // create model with a defined background-to-signal ratio
   TF1 model  = BuildModel(3,1);

   globalDataset = BuildDataset(model, numberCts);
   minimizer.Minimize(); 

   const double *par = minimizer.X();
   cout << "MLE: bkgCts = " << par[0] << " - sgnCts = " << par[1] << endl;
 
   return;

}

// create multiple datasets and find MLE for background and signal expectations
void ex2 (int numberCts = 1000) {

   // create minimizer and set its properties
   ROOT::Minuit2::Minuit2Minimizer minimizer (ROOT::Minuit2::kMigrad);
 
   minimizer.SetMaxFunctionCalls(1000000);
   minimizer.SetMaxIterations(100000);
   minimizer.SetTolerance(0.001);

   minimizer.SetVariable(0,"bkgCts",3./4.*numberCts, 1);
   minimizer.SetVariable(1,"sgnCts",1./4.*numberCts, 1);

   minimizer.SetVariableLowerLimit(0,1e-15);
   minimizer.SetVariableLowerLimit(1,1e-15);
 
   ROOT::Math::Functor f(&nll,2); 
   minimizer.SetFunction(f);


   // create model with a defined background-to-signal ratio
   TF1 model  = BuildModel(3,1);

   // create histograms where to store the MLE estimators
   TH1D sgnCtsMLE ("sgnCtsMLE","sgnCtsMLE", 100, 0, 1000);
   TH1D bkgCtsMLE ("bkgCtsMLE","bkgCtsMLE", 100, 0, 1000);
   for (int i = 0; i < 1e3; i++) {
      globalDataset = BuildDataset(model, numberCts);
      minimizer.Minimize(); 

      const double *par = minimizer.X();
      sgnCtsMLE.Fill(par[0]);
      bkgCtsMLE.Fill(par[1]);
   }

   // draw objects
   TCanvas cc;
   sgnCtsMLE.DrawClone();
   cc.DrawClone();
   bkgCtsMLE.SetLineColor(kRed);
   bkgCtsMLE.DrawClone();
   cc.DrawClone();
 
   return;

}
