import cPickle as pickle
import numpy as np

class conv_flux(object):
    def __init__(self):

        splinedir = './simulator/conv_flux/splines/'

        spline_file = splinedir + 'QGSJETII04_MSIS00_IC_SouthPole_January_finebins_corrdetsurface.dat'        
        inf = open(spline_file, 'r')
        self.data_MSI_QGS_Jan = pickle.load(inf)
        inf.close()

        spline_file = splinedir + 'QGSJETII04_MSIS00_IC_SouthPole_June_finebins_corrdetsurface.dat'           
        inf = open(spline_file, 'r')
        self.data_MSI_QGS_June = pickle.load(inf)
        inf.close()

    def add_flux(self, df):
        prim_type = {12:'NuE', -12:'NuEBar', 14:'NuMu', -14:'NuMuBar', 16:'NuTau', -16:'NuTauBar'}
    
        def calc_weight(row, spline, flux_type, wcorr=False):                                             
            #flux_type = 'conv', 'prompt'
            p_id = int(df['prim_type'].values[0]) 
            penergy = row['prim_energy']
            logpenergy = np.log10(row['prim_energy'])                                     
            pzenith = row['prim_coszenith']                                                                                 
    
            comp = prim_type[p_id]+'_'+flux_type
            if not comp in spline.keys():
                flux = 0.0
            else:
                flux = spline[comp](pzenith, logpenergy, grid=False)                                      
    
            return flux / penergy**3
    
        def MSI_QGS_JUN_conv(row):
            return calc_weight(row, self.data_MSI_QGS_June, 'conv')                                            
        vals_june =df.apply(MSI_QGS_JUN_conv, axis=1)  
        
        def MSI_QGS_JAN_conv(row):
            return calc_weight(row, self.data_MSI_QGS_Jan, 'conv')                                             
        vals_jan=df.apply(MSI_QGS_JAN_conv, axis=1) 

        df['conv_flux'] = 0.5 * (vals_june + vals_jan) 
