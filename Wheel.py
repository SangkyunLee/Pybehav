# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 14:26:03 2019

@author: slee
"""
import pandas as pd
import matplotlib.pyplot as plt
import logging
import pdb
import sys
                
import numpy as np
from Threadworker import *
from time import sleep
from Timer import Timer




def get_whspeed(sig,t, twin=0.1, ROTAROADSIZE_RAD =8, ENCODER_RESOLUTION = 2500):
    
    """" 
    get_whspeed(sig,t, twin=0.1, ROTAROADSIZE_RAD =8, ENCODER_RESOLUTION = 2500)
    ROTAROADSIZE_RAD = 8cm in radius (default)
    ENCODER_RESOLUTION = 2500 cycles/revolution
    twin = 0.1(default) in second
    The following codes take into account for counts/revolution
    1 cycle consists of 4 counts as follows:
        ROT1 = (0, 0, 0, 1, 1, 1, 1, 0) # clockwise
        ROT2 = (1, 0, 1, 1, 0, 1, 0, 0) # counter clockwise   
    """
    # rotation pattern (clockwise or counter clockwise)    
    ROT1 = (0, 0, 0, 1, 1, 1, 1, 0)
    ROT2 = (1, 0, 1, 1, 0, 1, 0, 0)    
    
#    A=sig[:,0]-np.mean(sig[:,0])
#    maxA = np.max(np.abs(A))
#    B=sig[:,1]-np.mean(sig[:,1]);
#    maxB = np.max(np.abs(B))
#    if maxA==0 or maxB==0 :        
#        return speed
#    A=(np.sign(A/maxA)+1)/2    
#    B=(np.sign(B/maxB)+1)/2;
    
    sig[sig<1]=0
    sig[sig>3]=5
    A = sig[:,0]/5
    B = sig[:,1]/5
        
    
    speed = np.zeros(t.shape)
    
    
    
    statechange_inxA = np.nonzero(np.diff(A)!=0)
    statechange_inxB = np.nonzero(np.diff(B)!=0)
    statechange_inx = np.concatenate((statechange_inxA[0], statechange_inxB[0]))
    statechange_inx = np.unique(statechange_inx)

    
    #logging.debug('statechange: {}, tlen:{}, Alen:{}'.format(len(statechange_inx),len(t),len(A)))
    Time_statechange = t[statechange_inx]
    
    M=np.array([A[statechange_inx], B[statechange_inx]]) 
    Maug =np.concatenate((M[:,:-1],M[:,1:]),axis=0)    
    lenstate = Maug.shape[1]    
    
    
    # rotation distance / count
    distseg = (ROTAROADSIZE_RAD*2*np.pi)/(ENCODER_RESOLUTION*4)
    
 
   
    #c = time()
    FWD = np.zeros(Maug.shape) # checking forward movement
    BWD = np.zeros(Maug.shape) # checking backward movement
    shifts = list(range(0,8,2))
    for i in range(0,4):
        ishift = shifts[i]
        ROT1sh = np.roll(ROT1,ishift)
        ROT2sh = np.roll(ROT2,ishift)            
        ROT1sh = ROT1sh[:4,np.newaxis]            
        ROT2sh = ROT2sh[:4,np.newaxis]       
        FWD[i,:]=  np.sum(np.abs(Maug - ROT1sh),0)==4
        BWD[i,:]=  np.sum(np.abs(Maug - ROT2sh),0)==4
    FWD = np.sum(FWD,0)>0
    BWD = np.sum(BWD,0)>0
    statetimedur = Time_statechange[1:]-Time_statechange[:-1]    
    statetimedur[BWD] = -1*statetimedur[BWD]  
    statetimedur[np.logical_not(np.logical_or(FWD,BWD))]=np.inf
    for i in range(lenstate):
        ist = statechange_inx[i]
        ied = statechange_inx[i+1]
        speed[ist:ied+1] = distseg/statetimedur[i]    
   

    tperiod = t[1]-t[0]
    winsize = int(twin/tperiod)
    
    twindow = np.ones(winsize)
    speed_t=np.convolve(speed,twindow)/winsize
    
    offset = int(winsize/2)
    speed_t = speed_t[range(offset,offset+len(t))]
    
#    plt.plot(t,speed)
#    plt.plot(t,speed_t)
    #print('totaltime:' + str(time()-c))
    return speed_t,speed


        
        
class Wheel:
    """ Class for wheel rotation info """
    
    def __init__(self,params):        
        try:
            self.wh_daqch= params['channels'] # channel index from daq card        
            self.daq = params['daq']        
        except KeyError:
            raise            
        
        self.ROTAROADSIZE_RAD = 8 # wheel size in cm
        self.ENCODER_RESOLUTION = 2500 # cycles/revolution
           
        self.alldata=np.array([])
        self.allt = np.array([])
        
        self.whspeed =[]        
        self.whsig =None
        self.t =None
        self.wheel_counter = 0
        self.calc_requested = False 
        
        if 'timer' in params:
            self.timer = params['timer']            
        else:
            self.timer = None
            
                
        self.worker = Threadworker(self.calc_speed)
        self.worker.setName('Wheelthreader')
        #self.worker = Processworker(self.calc_speed)
        
    def get_datinx(self):
        """ get channel index from acquired daq data"""
        map =np.zeros(8) # suppose maximum channel < 8
        ch1 = list(self.daq.ai_ch)
        map[ch1] = list(range(0,len(ch1) ))        
        inx = map[list(self.wh_daqch)]
        return inx.astype(int)
    
    def get_latestdaq(self):        
        """
        def get_latestdaq(self):
            get lastest segment daq data
        """
        chinx = self.get_datinx()        
        self.whsig = self.daq.data[:,chinx] # wheel signal Samples x ch        
        self.t = self.daq.t
            
        if self.alldata.size==0:
            self.alldata = self.whsig
            self.allt = self.t
        else:
            self.alldata = np.concatenate((self.alldata,self.whsig),axis=0)
            self.allt = np.concatenate((self.allt,self.t),axis=0)
        
    def enable_calc_speed(self):
        self.calc_requested = True
        return self.wheel_counter
    
    def get_wheel_counter(self):
        return self.wheel_counter
    
    def disable_calc_speed(self):
        self.calc_requested = False
        return self.wheel_counter
    
    def is_active(self,thr,start,end=-1):
        #pdb.set_trace()
       
        if start>=len(self.whspeed):
            return False
        
        if end == -1:
            spd = self.whspeed[start:]            
        else:
            spd = self.whspeed[start:end]
        #print('len:{}, start:{}'.format(len(self.whspeed),start))
        #pdb.set_trace()    
        #print('len:{}, start:{}, spdlen:{}'.format(len(self.whspeed),start,len(spd)))
        if np.nanmean(spd)>thr:
            return True
        else:
            return False
        
    def calc_speed(self):               
        logging.info('wheel_thread started!!!\n')
        
        
        while self.daq.worker.running():              
            daq_counter = self.daq.acq_counter
            dataready = self.daq.worker.check_datflag() # checking daq data is ready
            if self.daq.acq_dur<0.05:
                logging.warning('DAQ segment {} is too small for Wheel objects'.format(self.daq.acq_dur))
                
            if dataready:                       
                if True: #self.calc_requested:
                    self.get_latestdaq()
                    spd = get_whspeed(self.whsig,self.t, twin=0.1)
                    spd = round(np.nanmean(spd))
                #else:
                #    spd = float('nan')
                self.whspeed.insert(self.wheel_counter,spd)
                #self.whspeed.append(spd)
                
                workername = self.worker.getName() 
                logging.debug('{}: daq_counter:{}, wheel_counter:{},speed:{}, abstime:{}'\
                              .format(workername,daq_counter,self.wheel_counter,\
                                      self.whspeed[self.wheel_counter],self.timer.elapsed_time()))
                
                
                self.wheel_counter +=1
                self.daq.worker.clear_datflag()
            
        
             
            sleep(0.01)
        
        self.worker.stop()
        if not self.worker.running():
            workername = self.worker.getName()
            logging.info('{} STOPPED'.format(workername))
                
                
                
#def main():
#    if sys.platform =='win32':
#        fname = 'Z:\\Pybehav\\VoltageRecording-04042019-1055-001_Cycle00001_VoltageRecording_001.csv'
#    else:
#        fname = '/home/slee/data/Pybehav/VoltageRecording-04042019-1055-001_Cycle00001_VoltageRecording_001.csv'
#        
#       
#    D = pd.read_csv(fname, delimiter = ',')
#    dty = D.dtypes
#    #X = D.to_numpy()
#    X = D.values
#    X1 = X[:,1:]
#    
#    t= X[:,0]
#    t = t[:,np.newaxis]/1000
#    t = t.flatten()
#    sig=X1[:,2:]
#    speed= get_whspeed(sig,t)
#    plt.plot(t,speed)
#
#if __name__=='__main__':
#    main()            
# 


























######################################################################

#def get_whspeed(sig,t, twin=0.1, ROTAROADSIZE_RAD =8, ENCODER_RESOLUTION = 2500):
#    
#    """" 
#    get_whspeed(sig,t, twin=0.1, ROTAROADSIZE_RAD =8, ENCODER_RESOLUTION = 2500)
#    ROTAROADSIZE_RAD = 8cm in radius (default)
#    ENCODER_RESOLUTION = 2500 cycles/revolution
#    twin = 0.1(default) in second
#    The following codes take into account for counts/revolution
#    1 cycle consists of 4 counts as follows:
#        ROT1 = (0, 0, 0, 1, 1, 1, 1, 0) # clockwise
#        ROT2 = (1, 0, 1, 1, 0, 1, 0, 0) # counter clockwise   
#    """
#    # rotation pattern (clockwise or counter clockwise)    
#    ROT1 = (0, 0, 0, 1, 1, 1, 1, 0)
#    ROT2 = (1, 0, 1, 1, 0, 1, 0, 0)    
#    
#    A=sig[:,0]-np.mean(sig[:,0]);
#    A=(np.sign(A/np.max(np.abs(A)))+1)/2;
#    
#   
#    B=sig[:,1]-np.mean(sig[:,1]);
#    B=(np.sign(B/np.max(np.abs(B)))+1)/2;
#    
#    statechange_inxA = np.nonzero(np.diff(A)!=0)
#    statechange_inxB = np.nonzero(np.diff(B)!=0)
#    statechange_inx = np.concatenate((statechange_inxA[0], statechange_inxB[0]))
#    statechange_inx = np.unique(statechange_inx)
#    
#    Time_statechange = t[statechange_inx]
#    M=np.array([A[statechange_inx], B[statechange_inx]])
# 
#    Maug =np.concatenate((M[:,:-1],M[:,1:]),axis=0)
#    
#    lenstate = Maug.shape[1]    
#    
#    speed = np.zeros(t.shape)
#    # rotation distance / count
#    distseg = (ROTAROADSIZE_RAD*2*np.pi)/(ENCODER_RESOLUTION*4)
#    
# 
#   
#    #c = time()
#    FWD = np.zeros(Maug.shape) # checking forward movement
#    BWD = np.zeros(Maug.shape) # checking backward movement
#    shifts = list(range(0,8,2))
#    for i in range(0,4):
#        ishift = shifts[i]
#        ROT1sh = np.roll(ROT1,ishift)
#        ROT2sh = np.roll(ROT2,ishift)            
#        ROT1sh = ROT1sh[:4,np.newaxis]            
#        ROT2sh = ROT2sh[:4,np.newaxis]       
#        FWD[i,:]=  np.sum(np.abs(Maug - ROT1sh),0)==4
#        BWD[i,:]=  np.sum(np.abs(Maug - ROT2sh),0)==4
#    FWD = np.sum(FWD,0)>0
#    BWD = np.sum(BWD,0)>0
#    statetimedur = Time_statechange[1:]-Time_statechange[:-1]    
#    statetimedur[BWD] = -1*statetimedur[BWD]  
#    statetimedur[np.logical_not(np.logical_or(FWD,BWD))]=np.inf
#    for i in range(lenstate):
#        ist = statechange_inx[i]
#        ied = statechange_inx[i+1]
#        speed[ist:ied+1] = distseg/statetimedur[i]    
#   
#
#    tperiod = t[1]-t[0]
#    winsize = int(twin/tperiod)
#    
#    twindow = np.ones(winsize)
#    
#    speed_t=np.convolve(speed,twindow)/winsize
#    
#    offset = int(winsize/2)
#    speed_t = speed_t[range(offset,offset+len(t))]
#    
##    plt.plot(t,speed)
##    plt.plot(t,speed_t)
#    #print('totaltime:' + str(time()-c))
#    return speed_t