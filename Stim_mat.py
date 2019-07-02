# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 05:20:06 2019

@author: slee
"""

import scipy.io as sio
import os
from datetime import date, datetime
 

class vstim_mat:
    def __init__(self,**kwargs):
        
        self.fullfn = self.get_fullfn(kwargs)
        self.mat = sio.loadmat(self.fullfn)
        
    def get_fullfn(self, kwargs):
        if 'fullfn' in kwargs:
            return kwargs['fullfn']
        else:
            try:                
                tdir = kwargs['target_dir']
                clsname = kwargs['stim_clsname']
                
                 
                if type(kwargs['date'])==str :
                    dateinfo = kwargs['date']
                elif type(kwargs['date']) == datetime :
                    dateinfo = kwargs['date'].strftime('%Y-%m-%d')
                else:
                    raise ValueError
                
                fdir_date = os.path.join(tdir,clsname,dateinfo)
                fdir_time = os.listdir(fdir_date)
                fdir = os.path.join(fdir_date,fdir_time[-1]) 

                fullfn = os.path.join(fdir,clsname+'.mat')
                return fullfn
            except (KeyError, IOError,OSError) as err:
                print(err)
    def print_fileloc(self):
        print('\nVSTIM_FILE:{}\n'.format(self.fullfn))
                
    def get_params(self,paramkey,paramtype='trials'):
        outvalue = self.mat['stim']['params'][0,0][paramtype][0,0][paramkey][0,0]
        return outvalue.flatten()
            

#VSTIM_DIR='/home/pi/dev/Pybehav/test/StimulationData/SANG'
#datestr='2019-03-21'                    
#vstim_locdict ={'target_dir':VSTIM_DIR,'date':datestr,\
#                 'stim_clsname':'GratingExperiment_TRG',\
#                 'stimcond':'orientation','target_stimcond':0}
#     
#A=vstim_mat(**vstim_locdict)       
##b = A.get_params('orientation')     