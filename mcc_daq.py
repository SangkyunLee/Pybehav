# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:56:24 2019

@author: slee
"""
import sys
sys.path.insert(1,'/home/pi/dev/Pybehav/ref')

from time import sleep

import platform
node = platform.uname()[1]
if node == 'raspberrypi':
    from daqhats import mcc118, OptionFlags,TriggerModes, HatIDs, HatError
    from daqhats_utils import select_hat_device, enum_mask_to_string, \
    chan_list_to_mask
    

from Timer import Timer
from Threadworker import *
from Daq import *

import numpy as np
import pdb
import logging

#logging.basicConfig(level=logging.DEBUG)

READ_ALL_AVAILABLE = -1
class MCC118(DAQ):
    """ mcc118"""

    def __init__(self, params):     
        
        
        #super(DAQ,self).__init__(params)
        super().__init__(params)
        try:                                 
            self.timeout = params['timeout']
            self.acq_dur = params['acq_dur'] # acquistion segment duration
            
        except KeyError:
            raise
            
        if 'timer' in params:
            self.timer = params['timer']            
        else:
            self.timer = Timer()            
            
        self.read_request_size =[] # will be converted from self.acq_dur from init_daq
        self.hat = []  
              
        self.worker = Threadworker(self.acq_start)
        self.worker.setName('DAQthreader_ch'+str(self.ai_ch))
       
        
        
        self.init_daq()
        
    def init_daq(self):
        try:            
            address = select_hat_device(HatIDs.MCC_118)
            self.hat = mcc118(address)
            
            num_channels = len(self.ai_ch)             
            self.scan_rate = self.hat.a_in_scan_actual_rate(num_channels, self.scan_rate)        
            self.read_request_size = int(self.scan_rate*self.acq_dur)
        
            
            
        except (NameError, SyntaxError):
            pass
        
        
    def reset_timer(self):
        """
        def reset_timer(self):
            reset timer
        """
        self.timer.start()
        
    def record_cont(self):
        """
        def record_cont(self):
        recording continously while scan_status is running
        """
        
        nch  = len(self.ai_ch)            
            
        scan_status = self.hat.a_in_scan_status()    
        while self.worker.running() & scan_status.running : 
            
            scan_status = self.hat.a_in_scan_status()
            nsample =scan_status.samples_available
            
            if nsample>= self.read_request_size:                
                read_result = self.hat.a_in_scan_read_numpy(READ_ALL_AVAILABLE, self.timeout)       
                self.data_acqtime[self.acq_counter] = self.timer.elapsed_time()
                nsample = int(len(read_result.data) / nch) # 
                self.data_len[self.acq_counter] =nsample
                
                # Check for an overrun error
                if read_result.hardware_overrun:
                    print('\n\nHardware overrun\n')                
                elif read_result.buffer_overrun:
                    print('\n\nBuffer overrun\n')                
                    
                self.data = np.reshape(read_result.data,(nsample,nch))
                timeoff = self.total_nsample_perchannel/self.scan_rate
                self.t = timeoff + np.array(range(0,nsample))/self.scan_rate
                
                workername = self.worker.getName()                
                #logging.debug('{}: counter:{}, nsample:{}, abstime:{}'.format(workername,self.acq_counter, nsample, self.data_acqtime[self.acq_counter]))
                
                self.worker.set_datflag()
                
                self.total_nsample_perchannel += nsample
                self.acq_counter +=1
                sleep(self.acq_dur*0.9)
            else:                
                sleep(0.05)
                
                
    def record_N_sample(self):
        """
        def record_N_sample(self):
        read N samples 
        """
        
        nch  = len(self.ai_ch)                        
        scan_status = self.hat.a_in_scan_status()    
        total_samples_read =0
        segment_size = int(self.scan_rate* 0.1) #set segment size to 100msec
        N = self.read_request_size
        
        if self.worker.running() & scan_status.running :                         
            while total_samples_read <N:                        
                read_result = self.hat.a_in_scan_read_numpy(segment_size, self.timeout)       
                self.data_acqtime[self.acq_counter] = self.timer.elapsed_time()
                nsample = int(len(read_result.data) / nch) #                 
                # Check for an overrun error
                if read_result.hardware_overrun:
                    print('\n\nHardware overrun\n')                
                elif read_result.buffer_overrun:
                    print('\n\nBuffer overrun\n')                
                    
                dataseg = np.reshape(read_result.data,(nsample,nch))
                timeoff = self.total_nsample_perchannel/self.scan_rate
                tseg = timeoff + np.array(range(0,nsample))/self.scan_rate
                self.t = np.hstack((self.t,tseg))
                self.data = np.vstack((self.data,dataseg))
                self.data_len[self.acq_counter] =nsample
                
                workername = self.worker.getName()                
                #logging.debug('{}: counter:{}, nsample:{}, abstime:{}'.format(workername,self.acq_counter, nsample, self.data_acqtime[self.acq_counter]))                                                
                
                self.total_nsample_perchannel += nsample
                self.acq_counter +=1
                sleep(segment_size*0.9)
            self.worker.set_datflag() # set data_ready_flag
            
    
    def wait_for_trigger(self):
        """
        Monitor the status of the specified HAT device in a loop until the
        triggered status is True or the running status is False.        
        """
        # Read the status only to determine when the trigger occurs.
        is_running = True
        is_triggered = False
        while is_running and not is_triggered:
            status = self.hat.a_in_scan_status()
            is_running = status.running
            is_triggered = status.triggered
            
            
                
    def record_withtrigger(self):                                
        while self.worker.running():             
            self.wait_for_trigger()
            self.record_N_sample()         
            
                
    def acq_start(self):
        """
        def acq_start(self):
            acqusition start
        """
        
        
        channel_mask = chan_list_to_mask(self.ai_ch)
        
        if self.mode =='continuous':
            samples_per_channel = 0
            options = OptionFlags.CONTINUOUS    
                        
            self.hat.a_in_scan_start(channel_mask, samples_per_channel, self.scan_rate,options)                 
            self.record_cont()
            
            
        elif self.mode=='trigger':
            samples_per_channel = self.read_request_size
            options = OptionFlags.EXTTRIGGER
            trigger_mode = TriggerModes.RISING_EDGE
            
            self.hat.trigger_mode(trigger_mode)
            
            self.hat.a_in_scan_start(channel_mask, samples_per_channel, self.scan_rate,options) 
            self.record_withtrigger()
            
        elif self.mode =='finite':
            samples_per_channel = self.read_request_size
            options = OptionFlags.DEFAULT
            self.hat.a_in_scan_start(channel_mask, samples_per_channel, self.scan_rate,options) 
            self.record_N_sample()
            
            
        else:
            print('not implmented\n')
            raise      
        self.worker.clear_datflag()
            
    def acq_stop(self):
        self.hat.a_in_scan_stop()
        self.worker.stop()
    def acq_cleanup(self):
        self.hat.a_in_scan_cleanup()



#READ_ALL_AVAILABLE = -1
#class DAQ:
#    """ DAQ"""
#
#    def __init__(self, params):     
#        try:      
#            self.ch = params['channels']
#            self.scan_rate = params['scan_rate']
#            self.mode = params['mode'] # continuous, trigger, finite        
#            self.timeout = params['timeout']
#            self.acq_dur = params['acq_dur'] # acquistion segment duration
#        except KeyError:
#            raise
#            
#        self.read_request_size =[]
#        self.hat = [] 
#        
#        self.data =[] # segment data
#        self.t =[]  # acquisition relative time for segment
#        self.acq_counter = 0 # segment counter
#        self.total_nsample_perchannel =0  # total number of samples per channel
#        
#        if 'timer' in params:
#            self.timer = params['timer']            
#        else:
#            self.timer = Timer()            
#            
#        self.data_acqtime ={} #relative time of data-segment acquisition
#        self.data_len = {} # sample number per each segment
#        #pdb.set_trace()        
#        self.worker = Threadworker(self.acq_start)
#        self.worker.setName('DAQthreader_ch'+str(self.ch))
#        #self.worker = Processworker(self.acq_start)
#        
#        
#        self.init_daq()
#        
#    def init_daq(self):
#        try:            
#            address = select_hat_device(HatIDs.MCC_118)
#            self.hat = mcc118(address)
#            #pdb.set_trace()
#            num_channels = len(self.ch)
#            self.read_request_size = int(self.scan_rate*self.acq_dur) 
#            self.scan_rate = self.hat.a_in_scan_actual_rate(num_channels, self.scan_rate)        
#        
#            
#            
#        except (NameError, SyntaxError):
#            pass
#        
#        
#    def reset_timer(self):
#        """
#        def reset_timer(self):
#            reset timer
#        """
#        self.timer.start()
#        
#    def record_cont(self):
#        """
#        def record_cont(self):
#        recording continously while scan_status is running
#        """
#        
#        nch  = len(self.ch)            
#            
#        scan_status = self.hat.a_in_scan_status()    
#        while self.worker.running() & scan_status.running : 
#            
#            scan_status = self.hat.a_in_scan_status()
#            nsample =scan_status.samples_available
#            
#            if nsample>= self.read_request_size:                
#                read_result = self.hat.a_in_scan_read_numpy(READ_ALL_AVAILABLE, self.timeout)       
#                self.data_acqtime[self.acq_counter] = self.timer.elapsed_time()
#                nsample = int(len(read_result.data) / nch) # 
#                self.data_len[self.acq_counter] =nsample
#                
#                # Check for an overrun error
#                if read_result.hardware_overrun:
#                    print('\n\nHardware overrun\n')                
#                elif read_result.buffer_overrun:
#                    print('\n\nBuffer overrun\n')                
#                    
#                self.data = np.reshape(read_result.data,(nsample,nch))
#                timeoff = self.total_nsample_perchannel/self.scan_rate
#                self.t = timeoff + np.array(range(0,nsample))/self.scan_rate
#                
#                workername = self.worker.getName()                
#                #logging.debug('{}: counter:{}, nsample:{}, abstime:{}'.format(workername,self.acq_counter, nsample, self.data_acqtime[self.acq_counter]))
#                
#                self.worker.set_datflag()
#                
#                self.total_nsample_perchannel += nsample
#                self.acq_counter +=1
#                sleep(self.acq_dur*0.9)
#            else:                
#                sleep(0.05)
#                
#                
#    def record_N_sample(self):
#        """
#        def record_N_sample(self):
#        read N samples 
#        """
#        
#        nch  = len(self.ch)                        
#        scan_status = self.hat.a_in_scan_status()    
#        total_samples_read =0
#        segment_size = int(self.scan_rate* 0.1) #set segment size to 100msec
#        N = self.read_request_size
#        
#        if self.worker.running() & scan_status.running :                         
#            while total_samples_read <N:                        
#                read_result = self.hat.a_in_scan_read_numpy(segment_size, self.timeout)       
#                nsample = int(len(read_result.data) / nch) #                 
#                # Check for an overrun error
#                if read_result.hardware_overrun:
#                    print('\n\nHardware overrun\n')                
#                elif read_result.buffer_overrun:
#                    print('\n\nBuffer overrun\n')                
#                    
#                dataseg = np.reshape(read_result.data,(nsample,nch))
#                timeoff = self.total_nsample_perchannel/self.scan_rate
#                tseg = timeoff + np.array(range(0,nsample))/self.scan_rate
#                self.t = np.hstack((self.t,tseg))
#                self.data = np.vstack((self.data,dataseg))
#                self.data_len[self.acq_counter] =nsample
#                
#                workername = self.worker.getName()                
#                #logging.debug('{}: counter:{}, nsample:{}, abstime:{}'.format(workername,self.acq_counter, nsample, self.data_acqtime[self.acq_counter]))                                                
#                
#                self.total_nsample_perchannel += nsample
#                self.acq_counter +=1
#                sleep(segment_size*0.9)
#            self.worker.set_datflag() # set data_ready_flag
#            
#    
#    def wait_for_trigger(self):
#        """
#        Monitor the status of the specified HAT device in a loop until the
#        triggered status is True or the running status is False.        
#        """
#        # Read the status only to determine when the trigger occurs.
#        is_running = True
#        is_triggered = False
#        while is_running and not is_triggered:
#            status = self.hat.a_in_scan_status()
#            is_running = status.running
#            is_triggered = status.triggered
#            
#            
#                
#    def record_withtrigger(self):                                
#        while self.worker.running():             
#            self.wait_for_trigger()
#            self.record_N_sample()         
#            
#                
#    def acq_start(self):
#        """
#        def acq_start(self):
#            acqusition start
#        """
#        
#        
#        channel_mask = chan_list_to_mask(self.ch)
#        
#        if self.mode =='continuous':
#            samples_per_channel = 0
#            options = OptionFlags.CONTINUOUS    
#                        
#            self.hat.a_in_scan_start(channel_mask, samples_per_channel, self.scan_rate,options)                 
#            self.record_cont()
#            
#            
#        elif self.mode=='trigger':
#            samples_per_channel = self.read_request_size
#            options = OptionFlags.EXTTRIGGER
#            trigger_mode = TriggerModes.RISING_EDGE
#            
#            self.hat.trigger_mode(trigger_mode)
#            
#            self.hat.a_in_scan_start(channel_mask, samples_per_channel, self.scan_rate,options) 
#            self.record_withtrigger()
#            
#        elif self.mode =='finite':
#            samples_per_channel = self.read_request_size
#            options = OptionFlags.DEFAULT
#            self.hat.a_in_scan_start(channel_mask, samples_per_channel, self.scan_rate,options) 
#            self.record_N_sample()
#            
#            
#        else:
#            print('not implmented\n')
#            raise      
#        self.worker.clear_datflag()
#            
#    def acq_stop(self):
#        self.hat.a_in_scan_stop()
#        self.worker.stop()
#    def acq_cleanup(self):
#        self.hat.a_in_scan_cleanup()                