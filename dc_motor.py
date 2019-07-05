# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:01:08 2019

@author: slee
"""
import nidaqmx.system
import nidaqmx.errors

import logging
from Daq import *
import datetime


import json
import sys
from time import sleep
from Timer import Timer
from Threadworker import *
import numpy as np
from scipy import stats 
import math

#import pdb
import pandas as pd
import matplotlib.pyplot as plt


#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

class usb6009(DAQ):
    """ NI USB-6009 DAQ: for analog input and output"""
    
    
    def __init__(self, params):
        #super(usb6009, self).__init__(params)
        DAQ.__init__(self,params)
        self.device_name = 'USB-6009'
        try:            
            self.dev_id = params['dev_id']
            self.ai_fdb_chan = params['ai_feedback_ch']
            if self.mode == 'trigger':
                self.trg_chan = params['ai_trg_ch']
                self.trg_mode= params['trg_mode'] #'rising_edge' or 'falling_edge'
            
            self.ao_ch = params['ao_ch']
            self.acq_dur = params['acq_dur']
            self.ai_buffersize= 5100
            self.timeout =10
        except(KeyError, ValueError) as err:
            logging.error(err,exc_info=True)
        
        if 'timer' in params:
            self.timer = params['timer']            
        else:
            self.timer = Timer()
        
        self.init_daq()
        self.aiworker = Threadworker(self.acq_start)
        self.aiworker.setName('DAQthreader_ch'+str(self.ai_ch))
        self.aiworker_live = True
       
            
    def init_daq(self):
        """ configure USB-6009"""
        system = nidaqmx.system.System.local()
        
        dev = system.devices[self.dev_id]
        #dev = system.devices['Dev3']
        try:
            if not (dev.product_type==self.device_name):
                logging.error('not proper DAQ{} selected!\n',dev.product_type)
            
            self.task_ao = nidaqmx.Task()            
            self.task_ai = nidaqmx.Task()
            
            for i in self.ao_ch:
                ao_ch = self.get_channame('ao', i)
                self.task_ao.ao_channels.add_ao_voltage_chan(ao_ch,min_val=0.0,max_val=5.0)
                logging.info('ao_ch{} added'.format(ao_ch))
            
            for i in self.ai_ch:
                ai_ch = self.get_channame('ai', i)
                self.task_ai.ai_channels.add_ai_voltage_chan(ai_ch)
                logging.info('ai_ch{} added'.format(ai_ch))
            logging.info('\n')
            #ai_ch = self.get_channame('ai', self.ai_fdb_chan)
            #self.task_ai.ai_channels.add_ai_voltage_chan(ai_ch)            
            #trg_ch = self.get_channame('ai', self.trg_chan)                
            #self.task_ai.ai_channels.add_ai_voltage_chan(trg_ch)
            
            
            self.task_ai.timing.cfg_samp_clk_timing(self.scan_rate,samps_per_chan=self.ai_buffersize, \
                                                    sample_mode= nidaqmx.constants.AcquisitionType.CONTINUOUS)

            
            self.scan_rate = self.task_ai.timing.samp_clk_rate # actual sample rate set in the USB-6009
            self.read_request_size = int(self.scan_rate*self.acq_dur) # requested sample number
            
        except(nidaqmx.errors.DaqError,KeyError, ValueError) as err:
            logging.error(err,exc_info=True)
    
    
    def get_channame(self,chtype, ch):
        """
        def get_channame(self, chtype,ch):
            chtype: 'ao', 'ai'
            ch: channel number
            return device dependent channel names
        """
        return self.dev_id + "/" + chtype + str(ch)
    
    def get_ai_ch_inx(self, ch):
        """ def get_ai_ch_inx(self,chtype, ch):
        """
        chname = self.get_channame('ai',ch)
        chs = self.task_ai.channel_names
        return chs.index(chname)
        
    
        
    def record_N_sample(self):
        """
        def record_N_sample(self):
            read definite N samples 
        """
                       
        
        #self.total_samples_read =0
        self.acq_counter = 0
        segment_size = nidaqmx.constants.READ_ALL_AVAILABLE #set segment size to 100msec
        N = self.read_request_size
                              
        
        segdur=1#self.ai_buffersize/self.scan_rate
        while self.total_nsample_perchannel <N:     
            
            if not self.aiworker.running():
                break
                                 
            data = self.task_ai.read(segment_size, self.timeout)         
            self.data_acqtime[self.acq_counter] = self.timer.elapsed_time()
            nsample = len(data[0]) #                 
                      
                
            dataseg = np.array(data)
            timeoff = self.total_nsample_perchannel/self.scan_rate
            tseg = timeoff + np.array(range(0,nsample))/self.scan_rate
            
            
            if self.acq_counter ==0:
                self.data = dataseg
                self.t = tseg
            else:                
                self.data = np.hstack((self.data,dataseg))
                self.t = np.hstack((self.t,tseg))
                
                
            self.data_len[self.acq_counter] =nsample
            
            workername = self.aiworker.getName()                
            logging.debug('{}: counter:{}, nsample:{}, abstime:{}'.format(workername,self.acq_counter, nsample, self.data_acqtime[self.acq_counter]))                                                
            
            self.total_nsample_perchannel += nsample
            self.acq_counter +=1
            sleep(segdur*0.95)
            
            
        
       
            
    
    def wait_for_trigger(self):
        """
        def wait_for_trigger(self):
            wait_for_trigger        
        """
        inxch = self.get_ai_ch_inx(self.trg_chan)
        while True:
            value = self.task_ai.read()
            if self.trg_mode=='rising_edge' and np.mean(value[inxch])>2.5:
                logging.info('Triggered')
                break
            elif self.trg_mode =='falling_edge' and np.mean(value[inxch])<2.5:
                logging.info('Triggered')
                break
            sleep(0.001)
    
            
    def write_volt(self, vals):
        """
        def write_volt(self, val):
            vals: list for multiple channels
        """
        nch = len(self.task_ao.channel_names)        
        if nch ==len(vals):
            self.task_ao.write(vals)
        else:
            logging.error('No. ch:{}, No. val:{}'.format(nch, len(vals)))

                
    def acq_start(self):
        """
        def acq_start(self):
            acqusition start, this function is called from thread.start()
            within this function, record_X should be called
        """
        
        
        while True:
            if not self.aiworker_live:
                break
            
            if self.aiworker.running():        
                if self.mode == 'trigger':
                    self.wait_for_trigger()
                self.task_ai.start()
                self.record_N_sample()
                
            
            sleep(0.1)
            
    def acq_resume(self):
        """
        def acq_resume(self):
            start ai acquistion
        """
        
        # if thread was not started
        if not self.aiworker.is_alive():
            self.aiworker.start()
            logging.info("aiworker is started")
            
        # if thread was started already
        if not self.aiworker.running():
            self.aiworker.resume()
            logging.info("aiworker resumes working")
        
            
    def acq_stop(self):        
        """
        def acq_stop(self):
            acqusition stop
        """
        self.task_ai.stop()   
        self.task_ao.stop()   
        self.aiworker.stop()
        logging.info("aiworker is stopped")
    
    def cleanup(self):
        self.aiworker.stop()
        self.aiworker_live = False
        logging.info("aiworker is cleaned up")

        
        
        
        
class dc_motor_control:
    """ DC motor controller"""
    
    def __init__(self, params):
        
        try:
            if 'daqparam' in params:
                self.daq = usb6009(params['daqparam'])
            
            if 'speed_to_volt' in params:
                self.speed_to_volt = params['speed_to_volt']
            else:
                self.speed_to_volt ={}
            
            self.wheel_rad = params['wheel_radius']
            
        except(KeyError, ValueError) as err:
            logging.error(err,exc_info=True)
        
    def get_feedback_data(self):
        """def get_feedback_data(self):
            get signal from feedback channel
        """
        chinx =  self.daq.get_ai_ch_inx(self.daq.ai_fdb_chan)
        dat = self.daq.data[chinx]
        t = self.daq.t
        return dat, t
    
    def get_alldaqdata(self):
        dat = self.daq.data
        t = self.daq.t
        return dat, t
        

    def write_voltseq(self, dur, volt_list,prestimdur=1, poststimdur = 5, stop_volt=2.3):        
        
        block_dur = dur[0]        
        intblock_dur = dur[1]
        
        self.daq.acq_resume()
        sleep(prestimdur)
        self.daq.write_volt([stop_volt])        
        sleep(intblock_dur)
        for volt in volt_list:
            self.daq.write_volt([volt])
            sleep(block_dur)
            self.daq.write_volt([stop_volt])
            sleep(intblock_dur)
        sleep(poststimdur)    
        self.daq.acq_stop()
        
    def cal_volt_to_speed(self, volt_list, blockdur, stopvolt=2.3, offtime = 0.5, nocyc_per_revol = 699.2):
        """
        def cal_volt_to_speed(self, volt_list, blockdur, stopvolt=2.3, offtime = 0.5, nocyc_per_revol = 699.2):
            The default value (699.2) is from RB-Dfr-444, 64 counts resolution,
            gear ratio: 43.7 : 1, which corresponds to 2797 counts per revolution of the gearbox's output shaft
            
        """ 
        if type(volt_list)==list:
            volt_list = np.array(volt_list)
            
        y,t = self.get_feedback_data()
        y[y<2] = 0
        y[y>3] = 5
        
        Redge_inx = np.nonzero(np.diff(y)>0)[0] # only detect risingedge
        
        fs = 1/np.mean(np.diff(t))
        inx0 = np.nonzero(np.diff(Redge_inx)>(blockdur*0.5*fs))[0]
        
        inx1 = inx0+1
        inx1 = np.insert(inx1,0,0)
        inx_block_start = Redge_inx[inx1]
        
        inx2 = np.append(inx0,Redge_inx.shape[0]-1)
        inx_block_end = Redge_inx[inx2]
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t,y)
        y1 = np.ones(y[inx_block_start].shape)
        plt.plot(t[inx_block_start],y1,'.')
        y1 = 2*np.ones(y[inx_block_end].shape)
        plt.plot(t[inx_block_end],y1,'.')
        
        offsample = int(fs*offtime)
        
        nlist = inx_block_start.size
        encoder_cycle = np.zeros(nlist)
        tdur_block = np.zeros(nlist)
        for i in range(0,nlist):
            inx_start = inx_block_start[i]+offsample
            inx_end = inx_block_end[i]-offsample
            inx_sel = np.nonzero((Redge_inx>inx_start) & (Redge_inx<inx_end))[0]
            encoder_cycle[i] = inx_sel.size    
            tdur_block[i] = t[inx_end]-t[inx_start]
            
        norevpersec = encoder_cycle/tdur_block/nocyc_per_revol 
        plt.subplot(2,1,2)  
        
        
        
        nvoltinput = volt_list.size
            
        if nvoltinput ==nlist: 
            plt.plot(volt_list,norevpersec,'.')  
        elif nvoltinput> nlist:
            inx = np.argsort(volt_list-stopvolt)
            nzero = nvoltinput - nlist
            inx = np.setdiff1d(inx,list(range(0,nzero)))
            volt_list = volt_list[inx]
            plt.plot(volt_list,norevpersec,'.')
        else:
            logging.error('No blocks detected is greater than No voltageinput blocks' )
        
        return norevpersec, volt_list
    
      
    def get_function_of_speed(self, norevpersec, volts,  revmax=2.8):
        """
        def get_function_of_speed(self, norevpersec, volts,revmax):
            get slope and intercept of a linear function
            revmax: revolution max in the linear domain
            
        """
        
        speed = norevpersec*2*math.pi*self.wheel_rad # in cm/second
        a, b, r_value, p_value, std_err = stats.linregress(speed,volts)        
        plt.figure()
        plt.plot(speed,volts,'.')
        plt.plot(speed,speed*a+b)
        plt.show()
        speed_to_volt = {'a':a,'b':b,'speed_max': 2*math.pi*revmax*self.wheel_rad}
        
        self.speed_to_volt = speed_to_volt
        return speed_to_volt
        
    def convert_speed_volt(self,x):
        if x>=0 and x<self.speed_to_volt['speed_max']:
            a = self.speed_to_volt['a']
            b = self.speed_to_volt['b']
            return a*x+b
        else:
            logging.error('speed({}) cannot be greater than the maximum({})',a*x+b, self.speed_to_volt['speed_max'])
            
        
        
    def rotate_motor_steady(self, speedlist, dur,prerotdur=20, postrotdur = 10):
        """
        def rotate_motor_block(self, speedlist, dur ,prerotdur=1, postrotdur = 10):
            
            dur = [blocklen, intblocklen]
            speedlist = sequence of speed in cm/sec
            
        """
        
        
        block_dur = dur[0]        
        intblock_dur = dur[1]
        stop_volt = self.convert_speed_volt(0) 
        
        self.daq.acq_resume()        
        self.daq.write_volt([stop_volt])        
        sleep(prerotdur)
        
        for spd in speedlist:
            volt = self.convert_speed_volt(spd)
            self.daq.write_volt([volt])
            sleep(block_dur)
            self.daq.write_volt([stop_volt])
            sleep(intblock_dur)
        
        sleep(postrotdur)    
        self.daq.acq_stop()
    
    def rotate_motor_ramp(self,  speedlist, durlist,prerotdur=20, postrotdur = 10,nstep=100):
        """
        def rotate_motor_ramp(self,  speedlist, durlist,prerotdur=20, postrotdur = 10,nstep=100):
            
            
            durlist = [[blocklen, intblocklen],[blocklen, intblocklen]]
            
            speedlist = sequence of speed in cm/sec
            speedlist = [[10,20],[20,30]]; 1st block: ramp from 10 to 20, 2nd block: ramp from 20 to 30 
            nstep=20: within each block, #speed bin
                                                
            
        """
   		
        stop_volt = self.convert_speed_volt(0) 
        
        self.daq.acq_resume()
        
        self.daq.write_volt([stop_volt])        
        sleep(prerotdur)
        for spd,dur in zip(speedlist,durlist):
            spd1 = float(spd[0])
            spd2 = float(spd[1])
            block_dur = float(dur[0])        
            intblock_dur = float(dur[1])
   
            stepsize = float((spd2-spd1)/nstep)
            bindur = block_dur/(nstep+1)
            for i in range(nstep+1):
                spdbin = spd1+i*stepsize                                
                volt = self.convert_speed_volt(spdbin)
                self.daq.write_volt([volt])
                sleep(bindur)           
                logging.debug('SPblock{},{};{}:{},stepsize:{}'.format(spd1,spd2,spdbin,bindur,stepsize))
                
            if intblock_dur>0:
                self.daq.write_volt([stop_volt])
                sleep(intblock_dur)
        self.daq.write_volt([stop_volt])
        sleep(postrotdur)    
        self.daq.acq_stop()
        
    def exit(self):
        self.daq.cleanup()
        
        
        
    
def plot_daqsig(fname):
    """
    def plot_daqsig(fname):
        To confirm whether the data were saved properly,
        plot tseries of the data
    """
    
    data = pd.read_csv(fname)
    
    t= data['time'].values
    x1 =data['0'].values
    x2 =data['1'].values
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(t,x1)
    plt.subplot(2,1,2)
    plt.plot(t,x2)
    plt.show()
    
        
        
def main():
    
    

    if sys.argv[1] == 'Cal':
        # Meausre of speed to volt
        logging.info('Calibration mode')
        envfile ='DCmotor_Cal.json'
        with open(envfile) as envf:
            params=json.load(envf)       
        dc = dc_motor_control(params)
        #volt_list =np.arange(2.32,2.8,0.02)
        #volt_list =[2.32, 2.4]#np.arange(2.32,2.4,0.02)
        volt_list = params['volt_list']
        #stopvolt = 2.3
        stopvolt = params['stop_volt']
        dc.write_voltseq([2,2],volt_list,stop_volt=stopvolt)
        norevpersec, volts = dc.cal_volt_to_speed(volt_list,2,stopvolt=stopvolt)
        speed_to_volt = dc.get_function_of_speed(norevpersec, volts,  revmax=2.8)
        print('a:{}, b:{}'.format(speed_to_volt['a'],speed_to_volt['b']))
        try:
            while True:
                sleep(0.1)
        except KeyboardInterrupt:
            dc.exit()

        
    elif sys.argv[1] == 'Run':
        # Rotate motor
        logging.info('Run mode')
        envfile ='DCmotor_RUN.json'
        x = datetime.datetime.now()        
        datfname = x.strftime("%y%m%d")+'_'+x.strftime("%X")        
        file_name = './data/'+datfname.replace(':','')+'.csv'
        
        
        with open(envfile) as envf:
            params=json.load(envf)   
        dc = dc_motor_control(params)
        
        speedlist = params['speedlist']
        dur= params['dur'] # block length:20sec, interblock=0sec
        prerotdur=params['prerotdur']
        postrotdur =params['postrotdur']
        # block design with steady speed
        #dc.rotate_motor_steady(speedlist,dur, prerotdur, postrotdur) 
        # block design with steady speed
        dc.rotate_motor_ramp(speedlist,dur, prerotdur, postrotdur) 
        
        dc.exit()
        
        dat1,t = dc.get_alldaqdata()    
        df = pd.DataFrame({'time':t[:], '0':dat1[0,:],'1':dat1[1,:]})            
        df.to_csv(file_name, sep=',')
        print('Data are saved in '+file_name+'.\n')
        
        plot_daqsig(file_name)
        
        
        
        
        
    else:
        logging.error('{}: Not specified'.format(sys.argv[1]))
        
    sys.exit(0)
    #quit()

if __name__=='__main__':
    main()
    
 

        
        
        
        
    
#import json
#filename='DCmotor_RUN.json'
#
#with open(filename,'wt') as envf:    
#    j=json.dumps(params,indent=4)
#    envf.write(j)            
#        
    
        
        
        
        
        
###################################################            
#        
#            
#daqparam={'dev_id':'Dev3',\
#        'mode':'finite',\
#        'ai_ch':[0,1],\
#        'ai_feedback_ch':0,\
#        'ao_ch': 0,\
#        'ai_trg_ch':1,\
#        'trg_mode':'rising_edge',\
#        'scan_rate':5000,\
#        'acq_dur':500}
#
#
#
#params= {'daqparam':daqparam}
#
#dc = dc_motor_control(params)
##volt_list =np.arange(2.35,3.35,0.1)
#
##volt_list =np.arange(2.55,2.85,0.05)
##volt_list =np.arange(2,1.7,-0.05)
##volt_list =np.arange(2.31,3,0.02)
##volt_list =np.arange(2.31,2.35,0.01)
#
#volt_list =np.arange(2.32,2.8,0.02)
#dc.write_voltseq([2,2],volt_list,stop_volt=2.3)
#
#
#dat,t = dc.get_feedback_data()
#
#
#plt.plot(t,dat)
#
#dat1 = dat
#dat1[dat1<2] = 0
#dat1[dat1>3] = 5
#y=dat1
#
#
##plt.plot(t,dat1)
##plt.show()
#
#
#import pandas as pd
#import numpy as np
#
#
#df = pd.DataFrame({'time':t[:], 'feedback':dat1[:]})
#file_name='feedback.csv'
#df.to_csv(file_name, sep=',')
#
#
##### calculating motor speed
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#
#
#file_name='feedback.csv'
#intblock_len = 2
#
#offtime = 0.5
#
#df =pd.read_csv(file_name)
#dty = df.dtypes
##X = D.to_numpy()
#X = df.values
#
#t = df['time'].values
#y = df['feedback'].values
#
#Redge_inx = np.nonzero(np.diff(y)>0)[0] # only detect risingedge
##tedge = t[Redge_inx]
##plt.plot(t,y)
##plt.plot(tedge[:-1],np.diff(tedge),'.')
#
##inx0 = np.nonzero(np.diff(tedge)>(intblock_len*0.5))[0]
##inx_block = Redge_inx[inx0]
###inx_block = Redge_inx[np.insert(inx0,0,0)]
##plt.plot(t,y)
##plt.plot(t[inx_block],y[inx_block],'.')
#
#
#fs = 1/np.mean(np.diff(t))
#inx0 = np.nonzero(np.diff(Redge_inx)>(intblock_len*0.5*fs))[0]
#
#inx1 = inx0+1
#inx1 = np.insert(inx1,0,0)
#inx_block_start = Redge_inx[inx1]
#
#inx2 = np.append(inx0,Redge_inx.shape[0]-1)
#inx_block_end = Redge_inx[inx2]
#
#plt.figure()
#plt.subplot(2,1,1)
#plt.plot(t,y)
#y1 = np.ones(y[inx_block_start].shape)
#plt.plot(t[inx_block_start],y1,'.')
#y1 = 2*np.ones(y[inx_block_end].shape)
#plt.plot(t[inx_block_end],y1,'.')
#
#offsample = int(fs*offtime)
#
#nlist = inx_block_start.size
#encoder_cycle = np.zeros(nlist)
#tdur_block = np.zeros(nlist)
#for i in range(0,nlist):
#    inx_start = inx_block_start[i]+offsample
#    inx_end = inx_block_end[i]-offsample
#    inx_sel = np.nonzero((Redge_inx>inx_start) & (Redge_inx<inx_end))[0]
#    #plt.plot(t,y)
#    #plt.plot(t[Redge_inx[inx_sel]],y[Redge_inx[inx_sel]],'.')
#    encoder_cycle[i] = inx_sel.size    
#    tdur_block[i] = t[inx_end]-t[inx_start]
#    
#volt_list1 = np.setdiff1d(volt_list,np.array([2.3, 2.31]))  
#plt.subplot(2,1,2)  
#plt.plot(volt_list,encoder_cycle/tdur_block,'.')    
##plt.plot(encoder_cycle/tdur_block,'.')
#
#####################################3#     
##a= dc.data_acqtime
##a =np.array(list(a.items()))
##
##b=dc.data_len
##b =np.array(list(b.items()))
##
##
##
###value = dc.task_ai.read(5000) 
###plt.plot(value[1])        
##            
##        
##        
###data = np.zeros((2,1000))                
###reader =nidaqmx.stream_readers.AnalogMultiChannelReader
###reader.read_many_sample(*data)     
