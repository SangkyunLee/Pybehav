# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:54:00 2019

@author: slee
"""
import pandas as pd

import logging
import math
#import logging
import numpy as np
import matplotlib.pyplot as plt




#logging.basicConfig(level=logging.DEBUG)


class daqdat:
    
    def __init__(self,fname, chlist):
        
        try:            
            pdat = pd.read_csv(fname)
            if not chlist:
                self.pdat = pd.DataFrame(pdat.to_numpy(),index=pdat['time'])
            else:
                self.pdat = pd.DataFrame(pdat[chlist].to_numpy(),index=pdat['time'])                       
            
        except(KeyError,ValueError) as err:
            logging.error(err,exc_info=True)


    def getchsig(self, cols):
        """
        def getchsig(self, cols):
            extract chanel singlas to numpy    
        """    
        data = self.pdat[cols].to_numpy
        return data
    
    def gettimestamp(self):
        
        t=self.pdat.index.to_numpy()
        return t
        


def cal_motorspeed(s, t, twin = 0.1, wheel_raidus=8, nocyc_per_revol = 699.2):
    """    
     def cal_motorspeed(s,t, twin = 0.1, wheel_raidus=8,nocyc_per_revol = 699.2) :         
         calculate motorspeed from motor encoder                  
    """
     
    
    N = int(np.ceil(np.max(t)/twin))
    
    nocyc_per_revol = 699.2
    wheel_radius = 8
    
    s[s<2] = 0
    s[s>3] = 5
    
    
    tstamps = np.zeros(N)
    speed  = np.zeros(N)
    for i in range(N):
        inxst = twin*i+t[0]
        inxed = twin*(i+1)+t[0]
        
        if i==N:
            inxed = t[-1]
        inx=np.nonzero(np.bitwise_and(t>=inxst, t<inxed))
        a=s[inx]
        tstamps[i] = np.mean(t[inx])
        ncyc= len(np.nonzero(np.diff(a)>0)[0])
        norevpersec = ncyc/twin/nocyc_per_revol 
        speed[i] = norevpersec*2*math.pi*wheel_radius # in cm/second
#    if not fignum:
#        plt.figure()
#    else:
#        plt.figure(fignum)
#    plt.plot(tstamps,speed)
#    
    return speed, tstamps


def get_designspeed(t,params):
    dur = params['dur']
    spd = params['speedlist']
    predur = params['prerotdur']
    tdf = np.diff(t[:2])[0]
    spd_tseries = np.zeros(t.shape)
    st = predur
    for i in range(len(spd)):
        spdi = spd[i]
        duri = dur[i]
        et = st+duri[0] 
        inx = np.nonzero(np.bitwise_and(t>=st, t<et))[0]
        if spdi[0] == spdi[1]:
            spd_tseries[inx] = spdi[0]
        else:
            sz =inx.size
            spd_tseries[inx]=np.linspace(spdi[0],spdi[1],sz)
        
        #intblock
        st = et+duri[1]+tdf
    return spd_tseries
    


def plot_daqsig(fname,chlist, mode='all'):
    """
    def plot_daqsig(fname,chlist):
        To confirm whether the data were saved properly,
        plot tseries of the data
    """
    
    dc=daqdat(fname,chlist)    
    nc =len(dc.pdat.columns)
   
    plt.figure()
    if mode=='sep':
        for i in range(nc):    
            plt.subplot(nc,1,i+1)
            dc.pdat[i].plot()    
    else:
        dc.pdat.plot()
    plt.show()










def main():
    
    fname = './data/test-diff10.csv'

    chlist =['0','1']
    dc=daqdat(fname,chlist)
    
    plot_daqsig(fname,chlist, mode='sep')
    
    fbkch =0
    s= dc.pdat[fbkch].to_numpy() 
    t = dc.pdat.index.to_numpy()
    t = t-t[0]
        
    cal_motorspeed(s,t)

if __name__=='__main__':
    main()    
