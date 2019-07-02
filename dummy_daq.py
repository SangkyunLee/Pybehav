# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:56:24 2019

@author: slee
"""
import sys
sys.path.insert(1,'Z:\Pybehav')

from Wheel import *
from time import sleep
import pandas as pd
from Timer import Timer
from Threadworker import *
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

import pdb
import logging

logging.basicConfig(level=logging.DEBUG)


class fakeDAQ:
    """ DAQ"""

    def __init__(self, params):     
        try:			  
            self.rawdata = params['rawdata']	             
            self.scan_rate = params['scan_rate']           
            self.acq_dur = params['acq_dur'] # acquistion segment duration
        except KeyError:
            raise            
        self.ch = list(range(0,self.rawdata.shape[1])) 
        self.read_request_size =int(self.scan_rate*self.acq_dur)
       
        
        self.data =[] # segment data
        self.t =[]  # acquisition relative time for segment
        self.acq_counter = 0 # segment counter
        self.total_nsample_perchannel =0  # total number of samples per channel
        
        if 'timer' in params:
            self.timer = params['timer']            
        else:
            self.timer = Timer()            
            
        self.data_acqtime ={} #relative time of data-segment acquisition
        self.data_len = {} # sample number per each segment
   
        self.worker = Threadworker(self.acq_start)
       
       
        
        

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
        
        nsample = int(self.scan_rate*self.acq_dur)
        nch  = len(self.ch)            
        datalen = self.rawdata.shape[0]    
        
        while self.worker.running() : 
            try:
                
                sleep(self.acq_dur)
                #sleep(0.001)
                if self.total_nsample_perchannel>datalen:
                    break
             
                self.data_acqtime[self.acq_counter] = self.timer.elapsed_time()
               
                inx = range(self.acq_counter*nsample,(self.acq_counter+1)*nsample)
                self.data = self.rawdata[inx,:]
                timeoff = self.total_nsample_perchannel/self.scan_rate
                self.t = timeoff + np.array(range(0,nsample))/self.scan_rate
                self.data_len[self.acq_counter] =nsample
            
                
                workername = 'fakeDAQ'              
                #logging.debug('{}: counter:{}, nsample:{}, abstime:{}'.format(workername,self.acq_counter, self.total_nsample_perchannel, self.data_acqtime[self.acq_counter]))
                
                self.worker.set_datflag()
                
                self.total_nsample_perchannel += nsample
                self.acq_counter +=1
                
            except KeyboardInterrupt:
                logging.info('\nExit from DAQ\n')                                
                break
            
        self.acq_stop()
                
    def acq_start(self):
        """
        def acq_start(self):
            acqusition start
        """   
        self.record_cont()
        self.worker.clear_datflag()
            
    def acq_stop(self):       
        self.worker.stop()
        
        
##################################        

#def main():
if sys.platform =='win32':
    #fname = 'Z:\\Pybehav\\VoltageRecording-04042019-1055-001_Cycle00001_VoltageRecording_001.csv'
    fname = 'Z:\\Pybehav\\VoltageRecording-04042019-1055-002_Cycle00001_VoltageRecording_001.csv'
else:
    fname = '/home/slee/data/Pybehav/VoltageRecording-04042019-1055-001_Cycle00001_VoltageRecording_001.csv'
    
   
D = pd.read_csv(fname, delimiter = ',')
dty = D.dtypes
#X = D.to_numpy()
X = D.values
X1 = X[:,1:]
timer = Timer()
params = {'rawdata':X1, 'scan_rate':5000, 'acq_dur':0.1,\
          'timer':timer}
daq = fakeDAQ(params)
whparam ={'channels':[2,3], 'daq':daq,'timer':timer}
wheel =  Wheel(whparam)
daq.worker.start()  
sleep(0.005)          
wheel.worker.start() 
    
    
    
    
    
#    plt.plot(wheel.whspeed)
#    t= X[:,0]
#    t = t[:,np.newaxis]/1000
#    t = t.flatten()
#    sig=X1[:,2:]
#
#    speed_t,speed= get_whspeed(sig,t)

if __name__=='__main__':
    main()

       
