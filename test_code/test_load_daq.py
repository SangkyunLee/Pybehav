# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:14:57 2019

@author: slee
"""

import datajoint as dj
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt 
from time import time

from 


pl_pr = dj.create_virtual_module('pl_pr','pl_pr')
com = dj.create_virtual_module('common','common')

key ={'animal_id': 'SL_thy1_0206_MB',
 'fov': 'FOV1',
 'scan_id': 7,
 'session_id': '0425_D8'}

ch1 = (com.Daq()&key).fetch1('wheel_ch1')
ch2 = (com.Daq()&key).fetch1('wheel_ch2')



 
fn = (com.Daq()&key).fetch(as_dict=True)
fpath = fn[0]['data_path']
fpath =fpath.replace('/media/sde_WD6T/','V:/')

fname = fn[0]['data_fn']
fullfn = fpath + '/' + fname


# load a daq file and select wheel signals 
import pandas as pd
df = pd.read_csv(fullfn,sep=', ')
t = np.array(df.loc[:,'Time(ms)'])
wh = np.array(df.loc[:,['Input ' + str(ch1), 'Input ' + str(ch2)]])

plt.plot(t,wh)
sig = wh



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
    
    A=sig[:,0]-np.mean(sig[:,0]);
    A=(np.sign(A/np.max(np.abs(A)))+1)/2;
    
   
    B=sig[:,1]-np.mean(sig[:,1]);
    B=(np.sign(B/np.max(np.abs(B)))+1)/2;
    
    statechange_inxA = np.nonzero(np.diff(A)!=0)
    statechange_inxB = np.nonzero(np.diff(B)!=0)
    statechange_inx = np.concatenate((statechange_inxA[0], statechange_inxB[0]))
    statechange_inx = np.unique(statechange_inx)
    
    Time_statechange = t[statechange_inx]
    M=np.array([A[statechange_inx], B[statechange_inx]])
 
    Maug =np.concatenate((M[:,:-1],M[:,1:]),axis=0)
    
    lenstate = Maug.shape[1]    
    
    RotTime = np.Inf*np.ones(lenstate)   
    speed = np.zeros(t.shape)
    # rotation distance / count
    distseg = (ROTAROADSIZE_RAD*2*np.pi)/(ENCODER_RESOLUTION*4)
    
    error_time = list([])
    error_inx = list([])
   
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
    return speed_t    

                    
seltrange = (81,82)
tsec = t[::]/1000;
sig0= sig[::,:]
#tsec = t/1000;
#sig1 = sig

inx= np.logical_and(tsec>seltrange[0] , tsec<seltrange[1])
sig1 = sig0[inx,:]
tsec1 = tsec[inx]
twin =0.1
c = time()
speed_t = get_whspeed(sig1,tsec1,twin)
lap = (time()- c)


plt.plot(tsec1,speed_t)
plt.title(str(lap))


import scipy.io as sio
sio.savemat('Wheel.mat',{'sig':sig1,'tsec':tsec1})
                
    
            
###

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
#    RotTime = np.Inf*np.ones(lenstate)   
#    speed = np.zeros(t.shape)
#    # rotation distance / count
#    distseg = (ROTAROADSIZE_RAD*2*np.pi)/(ENCODER_RESOLUTION*4)
#    
#    error_time = list([])
#    error_inx = list([])
#   
#    c = time()
#    for i in range(lenstate):
#        
#        Mi = Maug[:,i]
#        statetime = Time_statechange[i+1] - Time_statechange[i]
#        ist = statechange_inx[i]
#        ied = statechange_inx[i+1]
#        
#        for ishift in range(0,8,2):
#            ROT1sh = np.roll(ROT1,ishift)
#            ROT2sh = np.roll(ROT2,ishift)
#            ROT1sh = ROT1sh[:4]
#            ROT2sh = ROT2sh[:4]
#            
#            
#            if np.all(Mi==ROT1sh):
#                direction = 1                
#                RotTime[i]=statetime
#                speed[ist:ied+1] = direction*distseg/statetime
#                break
#            elif np.all(Mi==ROT2sh):
#                direction = -1
#                RotTime[i]=statetime;
#                speed[ist:ied+1] = direction*distseg/statetime
#                break
#            else:
#                if ishift == 6:
#                    #print(" not matching state : " +str(i) )
#                    error_time.append(statetime)
#                    error_inx.append(i)
#        
#                    
#                    
#   
#    missinfo = {'statechange_inx':statechange_inx, 'error_inx':error_inx,
#                 'error_time':error_time}
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
#    print('totaltime:' + str(time()-c))
#    return speed_t 

                
                
            
    
   
    
    
    
    
    
    
