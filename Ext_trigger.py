# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:11:59 2019

@author: slee
"""


import logging

import json
import sys, os
from time import sleep
from Timer import Timer
from datetime import date, datetime

#from Daq import *
from mcc_daq import *
from Wheel import *
from mcc_dio import *
from Stim_mat import *
import pdb


logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)
class Trigger:
    """ generic class for tirgger generation"""
    def __init__(self,dio, params):
        
        if isinstance(dio,MCC152_DIO):
            self.dio = dio 
        else:
            raise ValueError('not proper dio object assigned')
            
        if 'timer' in params:
            self.timer = params['timer']
        else:
            self.timer = Timer()
        # setting a common timer in dio     
        self.dio.timer = self.timer
        
        self.init_dio()
        
        
    def init_dio(self): 
        # start digital input acquisition i.e., 2p-frame triggers
        self.dio.di_acqstart()
        
    def clear_dio(self):
        self.dio.di_acqstop()
    
    def elapsed_time(self,start=0):
        return self.timer.elapsed_time()-start
        
        
    def get_frame_counter(self):
        # 2p-frame counter
        return self.dio.get_interrupt_counter()        

    def wait_delayframe(self, nframe_delay, sleeptime=0.01):        
        """
        def wait_delayframe(self, nframe_delay, sleeptime=0.01):        
            wait until nframe_delay
        """ 
        try:
            while self.dio.get_interrupt_counter()<nframe_delay:
            
                #logging.debug('DIO_interrupt_counter:{}'.\
                #              format(self.dio.get_interrupt_counter()))
                sleep(sleeptime)   
        except KeyboardInterrupt:            
            print('\nKeyboard Interruption\n')
                
            
        timestamp = self.timer.elapsed_time()
        return timestamp
        
    def write_trigger(self, ch, timedur = 0):
        """ write output trigger"""
        
        timestamp = self.timer.elapsed_time()
        self.write_1(ch)
        if timedur>0:
            sleep(timedur)
        self.write_0(ch)
        
        return timestamp
    
    def toggle_state(self,ch):
        """ flip digital output value from the current state"""
                
        outputvalue = self.dio.read_outputvalue(ch)
        if outputvalue ==1:
            self.write_0(ch)
        else:
            self.write_1(ch)
        
        timestamp = self.timer.elapsed_time()
        return timestamp
    
    def write_1(self,ch):        
        self.dio.write_outputvalue(ch,1)
    def write_0(self,ch):
        self.dio.write_outputvalue(ch,0)
        
    def reset_dio_output(self):        
        for ich in self.dio.do_ch:
            self.dio.write_0(ich)
            
    __write_trigger = write_trigger
    
    
#Elecstim_trigger    
class Elecstim_trigger(Trigger):
    """ elecstim_trigger: generate trigger for electric stimulation
    """
    def __init__(self, dio, params):
        super(Elecstim_trigger, self).__init__(dio, params)
        
        
        self.elecstim_trigger_ch = params['elecstim_trigger_ch']
        self.delaytime_insec = params['delaytime']
        self.pulsedur_insec = params['pulsedur']       
        self.threadworker = Threadworker(self.gen_elecstim)
        self.threadworker.setName('ELEC_threader_ch'+str(self.elecstim_trigger_ch))
        self.elecstim_on = []
        self.current_trial = -1 # variabl to check synchronization with other threads
        self.increase_trialnum() # initiate trial number
        
    def set_delaytime(self,timeinsec):
        self.delaytime_insec = timeinsec
            
    def wait_delaytime(self):        
        sleep(self.delaytime_insec)
    
    def increase_trialnum(self):
        self.current_trial +=1
        self.elecstim_on.insert(self.get_trialnum(),False)
        
    def get_trialnum(self):
        return self.current_trial
        
    def is_elecstim_on(self):
        return self.elecstim_on[self.current_trial]
    
    def is_worker_running(self):
        return self.threadworker.running()
    
    def is_worker_ready(self):
        return self.threadworker.check_datflag()
    
        
    def gen_elecstim(self):
        
        while self.is_worker_running():
            if self.is_worker_ready():
                self.wait_delaytime()
                if self.is_elecstim_on():
                       
                    workername = self.threadworker.getName()
                    logging.info('{}: tr{}, abstime:{}'.format(workername,self.current_trial, self.timer.elapsed_time()))
                    self.write_trigger(self.elecstim_trigger_ch,self.pulsedur_insec)
                self.increase_trialnum()
                self.pause_worker()
            else:
                sleep(0.01)
                
                
    def pause_worker(self):            
        """def pause_worker(self):
            pause threadworker by clearing only data flags (cf. see stop function)
        """
        if self.threadworker.running():
            self.threadworker.clear_datflag()
            
            
    def resume_worker(self):
        """def resume_worker(self):
            pause threadworker by setting data flags, it is a pair of pause
        """
        if self.threadworker.running():
            self.threadworker.set_datflag()
            
    def start(self):
        self.threadworker.start()
            
    def stop(self):
        """def stop(self):
            stop threadworker by clearing running and data flags
        """
        if self.threadworker.running():
            self.threadworker.stop()
        self.clear_dio()
    
            
                
    
    
        
    
       
## Vstimbehav_trigger
class Vstimbehav_trigger(Trigger):
    """ Vstimbehav_trigger: generate triggers synchronized to 2p-scan frames
    for instance, generating a trigger every 10 2p-scan frames
    
    """
    def __init__(self,dio,params):
        super(Vstimbehav_trigger, self).__init__(dio, params)
        
        if 'daq' in params:
            self.daq = params['daq']
        else:
            self.daq = None
        
        try:
            if isinstance(self.daq,DAQ):
                self.daq.timer = self.timer            
                whparam ={'channels':params['wheel_ch'], 'daq':self.daq}
                self.wheel =  Wheel(whparam)
                self.wheel.timer = self.timer
                self.bwheel = True 
                
                # Wheel_speed threshold
                self.wheelspeedthr = params['Whspeedthr']
                self.elecstim_worker = params['elecstim']
                self.elecstim_worker.timer = self.timer
            else :            
                self.bwheel = False
                self.wheel = None
                self.elecstim_worker = None
            
            self.vstim_trigger_ch = params['vstim_trigger_ch']
            self.nframe_pretrial = params['nframe_pretrial']
            self.nframe_trial = params['nframe_trial']
            self.nframe_delay = params['nframe_delay'] # delay 2p-frames
            self.Ntrial = params['Ntrial'] # -1 vs positive number; for continuous mode, -1
            
            
            # visual stimulation matlab file loaded            
            self.stimcond = self.load_vstimparam(params['vstimparam'])
            self.target_stimcond = params['vstimparam']['target_stimcond']
            if self.stimcond.size<self.Ntrial:
                raise ValueError
                
        except (KeyError, ValueError) as err:
            logging.error(err,exc_info=True)
        
        
        if self.Ntrial ==-1:
            if not isinstance(self.nframe_trial,int):
                print('continous mode needs a single "nframe_trial"')
                sys.exit()
                
        # threads start        
        self.start_workers()
            
        



    def load_vstimparam(self, vstimparam):            
        self.vstim_mat = vstim_mat(**vstimparam)   
        self.vstim_mat.print_fileloc()
        target_param = vstimparam['stimcond']
        return self.vstim_mat.get_params(target_param)
       
    def start_workers(self): 
        if self.bwheel:
            self.daq.worker.start()            
            self.wheel.worker.start()
            self.elecstim_worker.start()
        
    def stop_workers(self):
        if self.bwheel:
            self.elecstim_worker.stop()
            self.daq.acq_stop()
            self.wheel.worker.stop()
        self.clear_dio()
            
            
    def get_nframe_trial(self,trial):
        """
        def get_nframe_trial(self,trial):
            get number of 2p frames /trial
        """        
        if isinstance(self.nframe_trial,int):
            Nframe = int(self.nframe_trial)
        else:
            Nframe = int(self.nframe_trial[trial])
        return Nframe
        
            
    def get_trial_num(self):
        if self.Ntrial==-1:
            return float("inf")
        else:
            return self.Ntrial
        
    def istarget_stim(self,itr):
        """ 
        def istarget_stim(self,itr):
        check whether the target stimuli are presented at 'itr' 
        """        
        if self.stimcond[itr]==self.target_stimcond:
            return True
        else:
            return False
        
    def set_elecstim_on(self, itr):
        itr_elecstim = self.get_elecstim_trial()
        if itr == itr_elecstim:
            self.elecstim_worker.elecstim_on[itr] = True
#        else:
#            raise Exception('Current trial {} for Elecstim does not match to input trial{}'.\
#                  format(itr_elecstim,itr))
        
    def set_elecstim_off(self, itr):
        itr_elecstim = self.get_elecstim_trial()
        if itr == itr_elecstim:
            self.elecstim_worker.elecstim_on[itr] = False
#        else:
#            raise Exception('Current trial {} for Elecstim does not match to input trial{}'.\
#                  format(itr_elecstim,itr))
            
            
    def set_elecstim_active(self):
        """ set elecstim_worker to be active """                
        self.elecstim_worker.resume_worker()
        
    def get_elecstim_trial(self):
        """ get elecstim trial number"""
        return self.elecstim_worker.get_trialnum()
        

        
        
        

    def gen_triggers(self):
        """ gen_triggers:
            
            wait for nframe_delay
            while Trials
                wait for pre-trial delay
                presenting visual stim
                while waiting for a trial completion
                    presenting elec stim with (delay)
        """
        
        
        Ntrial = self.get_trial_num()
        # wait until N frame_triggers occurs
        self.wait_delayframe(self.nframe_delay, 0.005)
        
        logging.info('\n========Vstimbehave_trigger started!!!===========\n')
        itr = 0
        error =False
        while itr < Ntrial and not error:
            
            try: 
                #enable rotating speed calculation
                # return daq acq_dur segment number of the calculation start    
                if self.bwheel:                    
                    whc1 = self.wheel.enable_calc_speed()      
                    self.set_elecstim_off(itr)
                
                # wait for pre-trial delay frames
                self.wait_delayframe(self.nframe_pretrial, 0.005)
                
                # if wheel is rotating before stimulus presentation from whc1 to now, 
                if self.bwheel and self.wheel.is_active(self.wheelspeedthr,whc1):
                    self.set_elecstim_on(itr)
                
                # create a pulse to trigger visual stimulation
                tstamp = self.write_trigger(self.vstim_trigger_ch)
                
                if self.bwheel:
                    whc2 = self.wheel.get_wheel_counter()
                    self.set_elecstim_active() # set elecstim-worker to be active
                
                
                frame1 = self.get_frame_counter()           
                #logging.debug('TRG:tr{},fr{}, tstamp:{}'.format(itr,frame1, tstamp))
                logging.info('TRG:tr{},fr{}, tstamp:{}'.format(itr,frame1, tstamp))
                
                # wait for next trial until 2p-frame counter reach N-frame 
                Nframe = self.get_nframe_trial(itr)
                elapsed_frame =0
                while  elapsed_frame <Nframe and not error:                                        
                    try:                        
                        if self.bwheel:      

                            # True when animal isnot running for target stimuli                                
                            if not self.wheel.is_active(self.wheelspeedthr,whc2) and\
                            self.istarget_stim(itr):                        
                                self.set_elecstim_on(itr)
                            
                            #S for test only
                            if not self.wheel.is_active(self.wheelspeedthr,whc2): 
                                self.set_elecstim_on(itr)
                            
                            
                            if self.get_elecstim_trial()>itr:                                
                                self.wheel.disable_calc_speed()  
                                
                        sleep(0.01)    
                        elapsed_frame =self.get_frame_counter() - frame1                    
                        #logging.info('elapased_frame:{}'.format(elapsed_frame))

                    except KeyboardInterrupt:
                        logging.info('\nExit from trial {}\n'.format(itr))
                        error = True
                    
                
                # increase number of trial number
                itr += 1                            
            except (ValueError,TypeError,RuntimeError) as err:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                #print(exc_type, fname, exc_tb.tb_lineno)
                logging.info('filename:{}'.format(fname),exc_info=True)
            
        self.stop_workers()
        logging.info('\n=========Vstimbehave_trigger stopped!!!========\n')
                
    


  
      
##############
def main():    
    """
    This function run mcc118 and mcc152 simultaneously
    """
    
    envfile ='Sang.json'
    with open(envfile) as envf:
        data=json.load(envf)
    
    
    
    dioparam = data['dioparam']    
    DIO = MCC152_DIO(dioparam)
    
    daqparam = data['daqparam']    
    vstimparam = data['vstimparam']
    #datestr = date.today().strftime('%Y-%m-%d')  
    datestr='2019-03-27'
    vstimparam['date']=datestr
    

    elecparam = data['elecparam']

    
    trparam = data['trparam']    
    trparam['daq']=MCC118(daqparam)
    trparam['elecstim'] = Elecstim_trigger(DIO,elecparam)
    trparam['vstimparam']= vstimparam
    trparam['timer']= Timer() # internally reset all the timers as this timer
    

    
    TR = Vstimbehav_trigger(DIO,trparam)
    TR.gen_triggers()



    
    

if __name__=='__main__':
    main()



