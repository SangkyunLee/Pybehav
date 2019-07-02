# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 05:02:22 2019

@author: slee
"""

WHEEL_CH = [0,1] # DAQ channels for wheel signal
FRAME_TRG_CH =0
VSTIM_OUTTRG_CH = 6
ELECSTIM_OUTTRG_CH = 7

TP_FRAME_DUR = 0.1 # 2p-frame duration in second

VSTIM_DIR = '/home/pi/dev/Pybehav/test/StimulationData/SANG'

dioparam ={'di_ch':FRAME_TRG_CH, 'do_ch':[VSTIM_OUTTRG_CH, ELECSTIM_OUTTRG_CH]}
daqparam={'channels':WHEEL_CH, 'scan_rate':5000, 'mode':'continuous',\
       'timeout':5,'acq_dur':0.1}
    
# visual stimulus matfile info and target condition
 
vstimparam ={'target_dir':VSTIM_DIR,\
                 'stim_clsname':'GratingExperiment_TRG',\
                 'stimcond':'orientation','target_stimcond':0}

# electric stimulation parameters

elecparam ={'delaytime':1,\
            'elecstim_trigger_ch':ELECSTIM_OUTTRG_CH,\
            'pulsedur':0.5}
# triger parameter
trparam ={
      'Ntrial':100, 'nframe_delay':10, 'nframe_pretrial':10,'nframe_trial':50,\
      'wheel_ch':WHEEL_CH, 'Whspeedthr':10,\
      'vstim_trigger_ch':VSTIM_OUTTRG_CH, \
      }
      

data={}
data['dioparam']=dioparam
data['daqparam']=daqparam
data['vstimparam']=vstimparam
data['elecparam']=elecparam
data['trparam']=trparam



import json
filename='Sang.json'

with open(filename,'wt') as envf:    
    j=json.dumps(data,indent=4)
    envf.write(j)
        

#with open(filename) as envf: 
#    d1=json.load(envf)        
#    
    
    