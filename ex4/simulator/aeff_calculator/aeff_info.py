#!/usr/bin/env python

class aeff_info_cascades(object):
    def __init__(self):

        self.nue_cc = './simulator/aeff_input/effective_area.per_bin.nu_e.cc.cascade.txt'
        self.nue_cc_smoothing = 1

        self.nue_nc = './simulator/aeff_input/effective_area.per_bin.nu_e.nc.cascade.txt'
        self.nue_nc_smoothing = 5

        self.nue_gr = './aeff_input/effective_area.per_bin.nu_e_bar.gr.cascade.txt'
        self.nue_gr_smoothing = 15

        self.numu_cc = './aeff_input/effective_area.per_bin.nu_mu.cc.cascade.txt'
        self.numu_cc_smoothing = 150

        self.numu_nc = './simulator/aeff_input/effective_area.per_bin.nu_mu.nc.cascade.txt'
        self.numu_nc_smoothing = 300

        self.nutau_cc = './aeff_input/effective_area.per_bin.nu_tau.cc.cascade.txt'
        self.nutau_cc_smoothing = 5

        self.nutau_nc = './aeff_input/effective_area.per_bin.nu_tau.nc.cascade.txt'
        self.nutau_nc_smoothing = 30


        self.nuebar_cc = './aeff_input/effective_area.per_bin.nu_e_bar.cc.cascade.txt'
        self.nuebar_cc_smoothing = 1

        self.nuebar_nc = './aeff_input/effective_area.per_bin.nu_e_bar.nc.cascade.txt'
        self.nuebar_nc_smoothing = 5

        self.nuebar_gr = './aeff_input/effective_area.per_bin.nu_e_bar_bar.gr.cascade.txt'
        self.nuebar_gr_smoothing = 15

        self.numubar_cc = './aeff_input/effective_area.per_bin.nu_mu_bar.cc.cascade.txt'
        self.numubar_cc_smoothing = 150

        self.numubar_nc = './aeff_input/effective_area.per_bin.nu_mu_bar.nc.cascade.txt'
        self.numubar_nc_smoothing = 300

        self.nutaubar_cc = './aeff_input/effective_area.per_bin.nu_tau_bar.cc.cascade.txt'
        self.nutaubar_cc_smoothing = 5

        self.nutaubar_nc = './aeff_input/effective_area.per_bin.nu_tau_bar.nc.cascade.txt'
        self.nutaubar_nc_smoothing = 30

        '''
        self.nue_cc = '../aeff_input/effective_area.per_bin.nu_e.cc.cascade.txt'
        self.nue_cc_smoothing = 1

        self.nue_nc = '../aeff_input/effective_area.per_bin.nu_e.nc.cascade.txt'
        self.nue_nc_smoothing = 5

        self.nue_gr = '../aeff_input/effective_area.per_bin.nu_e_bar.gr.cascade.txt'
        self.nue_gr_smoothing = 15

        self.numu_cc = '../aeff_input/effective_area.per_bin.nu_mu.cc.cascade.txt'
        self.numu_cc_smoothing = 150

        self.numu_nc = '../aeff_input/effective_area.per_bin.nu_mu.nc.cascade.txt'
        self.numu_nc_smoothing = 300

        self.nutau_cc = '../aeff_input/effective_area.per_bin.nu_tau.cc.cascade.txt'
        self.nutau_cc_smoothing = 5

        self.nutau_nc = '../aeff_input/effective_area.per_bin.nu_tau.nc.cascade.txt'
        self.nutau_nc_smoothing = 30
        '''

        self.infiles = {'nue_NC': self.nue_nc, 'nue_CC': self.nue_cc, 'numu_NC': self.numu_nc, 'numu_CC': self.numu_cc, 'nutau_NC': self.nutau_nc, 'nutau_CC': self.nutau_cc, 'nue_GR': self.nue_gr,
                'nuebar_NC': self.nuebar_nc, 'nuebar_CC': self.nuebar_cc, 'numubar_NC': self.numubar_nc, 'numubar_CC': self.numubar_cc, 'nutaubar_NC': self.nutaubar_nc, 'nutaubar_CC': self.nutaubar_cc, 'nuebar_GR': self.nuebar_gr}
        self.smoothing = {'nue_NC': self.nue_nc_smoothing, 'nue_CC': self.nue_cc_smoothing, 'numu_NC': self.numu_nc_smoothing, 'numu_CC': self.numu_cc_smoothing, 'nutau_NC': self.nutau_nc_smoothing, 'nutau_CC': self.nutau_cc_smoothing, 'nue_GR': self.nue_gr_smoothing,
                'nuebar_NC': self.nuebar_nc_smoothing, 'nuebar_CC': self.nuebar_cc_smoothing, 'numubar_NC': self.numubar_nc_smoothing, 'numubar_CC': self.numubar_cc_smoothing, 'nutaubar_NC': self.nutaubar_nc_smoothing, 'nutaubar_CC': self.nutaubar_cc_smoothing, 'nuebar_GR': self.nuebar_gr_smoothing}


class aeff_info_tracks(object):
    def __init__(self):

        self.nue_cc = './aeff_input/effective_area.per_bin.nu_e.cc.track.txt'
        self.nue_cc_smoothing = 100

        self.nue_nc = './aeff_input/effective_area.per_bin.nu_e.nc.track.txt'
        self.nue_nc_smoothing = 50

        self.nue_gr = './aeff_input/effective_area.per_bin.nu_e_bar.gr.track.txt'
        self.nue_gr_smoothing = 15

        self.numu_cc = './aeff_input/effective_area.per_bin.nu_mu.cc.track.txt'
        self.numu_cc_smoothing = 100

        self.numu_nc = './aeff_input/effective_area.per_bin.nu_mu.nc.track.txt'
        self.numu_nc_smoothing = 100

        self.nutau_cc = './aeff_input/effective_area.per_bin.nu_tau.cc.track.txt'
        self.nutau_cc_smoothing = 150

        self.nutau_nc = './aeff_input/effective_area.per_bin.nu_tau.nc.track.txt'
        self.nutau_nc_smoothing = 100


        self.nuebar_cc = './aeff_input/effective_area.per_bin.nu_e_bar.cc.track.txt'
        self.nuebar_cc_smoothing = 100

        self.nuebar_nc = './aeff_input/effective_area.per_bin.nu_e_bar.nc.track.txt'
        self.nuebar_nc_smoothing = 100

        self.nuebar_gr = './aeff_input/effective_area.per_bin.nu_e_bar_bar.gr.track.txt'
        self.nuebar_gr_smoothing = 15

        self.numubar_cc = './aeff_input/effective_area.per_bin.nu_mu_bar.cc.track.txt'
        self.numubar_cc_smoothing = 100

        self.numubar_nc = './aeff_input/effective_area.per_bin.nu_mu_bar.nc.track.txt'
        self.numubar_nc_smoothing = 100

        self.nutaubar_cc = './aeff_input/effective_area.per_bin.nu_tau_bar.cc.track.txt'
        self.nutaubar_cc_smoothing = 150

        self.nutaubar_nc = './aeff_input/effective_area.per_bin.nu_tau_bar.nc.track.txt'
        self.nutaubar_nc_smoothing = 100

        '''
        self.nue_cc = '../aeff_input/effective_area.per_bin.nu_e.cc.track.txt'
        self.nue_cc_smoothing = 1

        self.nue_nc = '../aeff_input/effective_area.per_bin.nu_e.nc.track.txt'
        self.nue_nc_smoothing = 5

        self.nue_gr = '../aeff_input/effective_area.per_bin.nu_e_bar.gr.track.txt'
        self.nue_gr_smoothing = 15

        self.numu_cc = '../aeff_input/effective_area.per_bin.nu_mu.cc.track.txt'
        self.numu_cc_smoothing = 150

        self.numu_nc = '../aeff_input/effective_area.per_bin.nu_mu.nc.track.txt'
        self.numu_nc_smoothing = 300

        self.nutau_cc = '../aeff_input/effective_area.per_bin.nu_tau.cc.track.txt'
        self.nutau_cc_smoothing = 5

        self.nutau_nc = '../aeff_input/effective_area.per_bin.nu_tau.nc.track.txt'
        self.nutau_nc_smoothing = 30
        '''

        self.infiles = {'nue_NC': self.nue_nc, 'nue_CC': self.nue_cc, 'numu_NC': self.numu_nc, 'numu_CC': self.numu_cc, 'nutau_NC': self.nutau_nc, 'nutau_CC': self.nutau_cc, 'nue_GR': self.nue_gr,
                'nuebar_NC': self.nuebar_nc, 'nuebar_CC': self.nuebar_cc, 'numubar_NC': self.numubar_nc, 'numubar_CC': self.numubar_cc, 'nutaubar_NC': self.nutaubar_nc, 'nutaubar_CC': self.nutaubar_cc, 'nuebar_GR': self.nuebar_gr}
        self.smoothing = {'nue_NC': self.nue_nc_smoothing, 'nue_CC': self.nue_cc_smoothing, 'numu_NC': self.numu_nc_smoothing, 'numu_CC': self.numu_cc_smoothing, 'nutau_NC': self.nutau_nc_smoothing, 'nutau_CC': self.nutau_cc_smoothing, 'nue_GR': self.nue_gr_smoothing,
                'nuebar_NC': self.nuebar_nc_smoothing, 'nuebar_CC': self.nuebar_cc_smoothing, 'numubar_NC': self.numubar_nc_smoothing, 'numubar_CC': self.numubar_cc_smoothing, 'nutaubar_NC': self.nutaubar_nc_smoothing, 'nutaubar_CC': self.nutaubar_cc_smoothing, 'nuebar_GR': self.nuebar_gr_smoothing}
