/*****************************************************************************
 * Statistical and Machine Learning Methods in Particle and Astrophysics
 *
 * TUM - summer term 2019
 * M. Agostini <matteo.agostini@tum.de>
 *
 * Ex 1, conceptual steps:
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
   const double mean       =  10;
   const double sigma      =   1;

   // define parameters for the background distribution, i.e. the window edges
   const double rangeMin   =   0; 
   const double rangeMax   =  20;
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

   // Initialize data set histogram
   TH1D dataset ("","",100,xmin,xmax);

   // Fill histogram
   // Randomly define the number of cts of the data set
   const int rndCts = gRandom->Poisson(cts);

   for (int i = 0; i < rndCts; i++) dataset.Fill(model.GetRandom(xmin,xmax));

   return dataset;
}

// create objects and plot them
void ex1 () {
   // these commands can also be run directly using the interpreter after
   // loading the macro as a library
   //  .L ex1.C
   TCanvas cc;
   cc.Divide(2,1);
   cc.cd(1);
   auto myModel = BuildModel(1,1);
   myModel.DrawCopy();
   cc.cd(2);
   auto myDataset = BuildDataset(myModel, 2000);
   myDataset.DrawCopy();
   cc.DrawClone();
}
