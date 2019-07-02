# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:26:44 2019

@author: slee
"""
import logging

class DAQ:
    """ base class for digital acquisition system
        segment-based data acquistion
        
    """
    
    def __init__(self,params): 
        
        try:            
            self.scan_rate = params['scan_rate']        
            self.mode = params['mode'] # continuous, trigger, finite                
            self.ai_ch = params['ai_ch'] # ai channel
        except (KeyError, ValueError) as err:
            logging.error(err,exc_info=True)
        
        self.dev_id = None
        self.data =[] # current segment data
        self.t =[]  # acquisition relative time for segment
        
        
        self.acq_dur = 0 # acquistion duration for each segment for 'continuous'
                         #                     for entire recording time for 'finite'  
        self.acq_counter = 0 # segment counter
        self.total_nsample_perchannel =0  # total number of samples per channel        
        self.data_acqtime ={} #relative time of data-segment acquisition
        self.data_len = {} # sample number per each segment

        
       
        
    def init_daq(self):        
        """ 
        def init_daq(self):        
            initialize daq
        """

        
    def record_cont(self):
        """
        def record_cont(self):
        recording continously while scan_status is running
        """
        
       
                
    def record_N_sample(self):
        """
        def record_N_sample(self):
            read definite N samples 
        """
        
       
            
    
    def wait_for_trigger(self):
        """
        def wait_for_trigger(self):
            wait_for_trigger        
        """

                
    def acq_start(self):
        """
        def acq_start(self):
            acqusition start
            within this function, record_X should be called
        """
        
            
    def acq_stop(self):        
        """
        def acq_stop(self):
            acqusition stop
        """
        