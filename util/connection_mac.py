# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 07:48:05 2019

@author: slee
"""

"""
ip-address for mac-vstim: \\172.22.92.141
sudo mount -t cifs //172.22.92.141/stimulation-1 ./test -o username=sslab,vers=1.0
"""
"""
connecting to slee-ss for the test
sudo mount -t cifs //172.23.152.99/stimulation ./test -o username=slee,vers=2.0
"""

import h5py
import scipy.io as sio
import os
from datetime import date, datetime
target_dir = '/home/pi/dev/Pybehav/test/StimulationData/SANG'
stim_clsname ='GratingExperiment_TRG' 

#today = date.today().strftime('%Y-%m-%d')
today = datetime(2019, 3, 21).strftime('%Y-%m-%d')
#datestr=("%d-%d-%d"%(today.year,today.month,today.day))

fdir_date = os.path.join(target_dir,stim_clsname,today)
fdir_time = os.listdir(fdir_date)
fdir = os.path.join(fdir_date,fdir_time[-1])    

fullfn = os.path.join(fdir,stim_clsname+'.mat')    
f = sio.loadmat(fullfn)
a=f['stim']['pams'][0,0]['trials'][0,0]['orientation'][0,0]ra


fullfn ='GratingExperiment_TRG.mat'
os.path.exists(fullfn)
f = sio.loadmat(fullfn)



except IndexError:
    

