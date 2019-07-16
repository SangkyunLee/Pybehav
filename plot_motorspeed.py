# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:07:07 2019

@author: slee
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:47:57 2019

@author: slee
"""
import numpy as np
from util import datutil
import math
import matplotlib.pyplot as plt
import json
import os

loc ='./data/mouseC'

flist = []
for fname in os.listdir(loc):
    jsonf = fname.split('.')[0]+'.json'
    if fname.endswith('.csv') and os.path.exists(loc+'/'+jsonf):
        flist.append(fname)

flist = sorted(flist)



    

        
fbkch =0        
chlist =['0','1']
#fignum=1
#plt.figure(fignum)
for fcsv in flist:
    fcsv1 = loc +'/'+fcsv
    dc=datutil.daqdat(fcsv1,chlist)
    fjson = loc + '/' +fcsv.split('.')[0]+'.json'    
    with open(fjson) as envf:
            params=json.load(envf) 
    
    s= dc.pdat[fbkch].to_numpy() 
    t = dc.pdat.index.to_numpy()
    t = t-t[0]    
    actspd, tstamps = datutil.cal_motorspeed(s,t)
    spd_tseries = datutil.get_designspeed(t,params)
    plt.figure()
    plt.plot(t,spd_tseries)
    plt.plot(tstamps,actspd)
    plt.title(fcsv1)
    plt.savefig(loc+'/'+fcsv.split('.')[0]+'.png')
    