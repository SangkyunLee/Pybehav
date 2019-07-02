# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:49:25 2019

@author: slee
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:56:24 2019

@author: slee
"""
import sys
#sys.path.insert(1,'/home/pi/dev/Pybehav/ref')

from time import sleep
from daqhats import mcc118, OptionFlags,TriggerModes, HatIDs, HatError, \
    mcc152, DIOConfigItem, \
    interrupt_callback_enable, HatCallback, interrupt_callback_disable 


from daqhats_utils import select_hat_device, enum_mask_to_string, \
    chan_list_to_mask
 


from Threadworker import *
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

READ_ALL_AVAILABLE = -1
class DAQ:
    """ mcc118 DAQ"""

    def __init__(self, params):     
        try:      
            self.ch = params['channels']
            self.scan_rate = params['scan_rate']          
            self.timeout = params['timeout']
            self.acq_segdur = params['acq_segdur'] # acquistion segment duration
        except KeyError:
            raise
            
        self.read_request_size =None
        self.hat = None 
        
    
        self.acq_counter = 0 # segment counter
        self.total_nsample_perchannel =0  # total number of samples per channel
        
        # self.acq_start is given to threadworker to run daq acquisition in a separate thread.
        self.worker = Threadworker(self.acq_start)
        self.worker.setName('DAQthreader_ch'+str(self.ch))

        
        
        self.init_daq()
        
    def init_daq(self):
        try:            
            address = select_hat_device(HatIDs.MCC_118)
            self.hat = mcc118(address)
            #pdb.set_trace()
            num_channels = len(self.ch)
            self.read_request_size = int(self.scan_rate*self.acq_segdur) 
            self.scan_rate = self.hat.a_in_scan_actual_rate(num_channels, self.scan_rate)        
        
            
            
        except (NameError, SyntaxError):
            pass
        
        
 
        
    def record_cont(self):
        """
        def record_cont(self):
        recording continously while scan_status is running
        """
        
        nch  = len(self.ch)            
            
        scan_status = self.hat.a_in_scan_status()    
        while self.worker.running() & scan_status.running : 
            
            scan_status = self.hat.a_in_scan_status()
            nsample =scan_status.samples_available
            
            if nsample>= self.read_request_size:                
                read_result = self.hat.a_in_scan_read_numpy(READ_ALL_AVAILABLE, self.timeout)       
                nsample = int(len(read_result.data) / nch) # 
                  
                # Check for an overrun error
                if read_result.hardware_overrun:
                    print('\n\nHardware overrun\n')                
                elif read_result.buffer_overrun:
                    print('\n\nBuffer overrun\n')                
                    

                workername = self.worker.getName()
                logging.debug('{}: counter:{}, nsample:{}'.format(workername,self.acq_counter, nsample))
                

                self.acq_counter +=1
                sleep(self.acq_segdur*0.9)
            else:                
                sleep(0.05)
                
                
                
    def acq_start(self):    
        channel_mask = chan_list_to_mask(self.ch)    
        self.hat.a_in_scan_start(channel_mask, 0, self.scan_rate,OptionFlags.CONTINUOUS)                 
        self.record_cont()   
            
       
            
    def acq_stop(self):
        self.hat.a_in_scan_stop()
        self.worker.pause()



"""
MCC152 module definition
"""
class MCC152_DIO:
    def __init__(self,params):
        try:
            self.di_ch = params['di_ch']   # digital input channels             
            self.do_ch = params['do_ch']   # digital output channels       
          
        except KeyError:
            raise
            
        self.HAT = None
        # data for input interruption
        self.interrupt_counter = [0]
        self.init_dev()
        self.callback = HatCallback(self.interrupt_callback)

    def init_dev(self):
        address = select_hat_device(HatIDs.MCC_152)
        self.HAT = mcc152(address)
        
        # Reset the DIO to defaults (all channels input, pull-up resistors
        # enabled).
        self.HAT.dio_reset()
        # Read the initial input values so we don't trigger an interrupt when
        # we enable them.
        self.HAT.dio_input_read_port()
        
        
        # set digital ouptput channels
        for ch in self.do_ch:
            try:
                self.HAT.dio_config_write_bit(ch, DIOConfigItem.DIRECTION, 0)
            except (HatError, ValueError):
                print('could not configure the channel{} as output'.format(ch))
                sys.exit()
                
        # set digital iput channels as latched input        
        for ch in self.di_ch:
            try:
                self.HAT.dio_config_write_bit(ch, DIOConfigItem.DIRECTION, 1)
                # Enable latched inputs so we know that a value changed even if it changes
                # back to the original value before the interrupt callback.
                self.HAT.dio_config_write_bit(ch,DIOConfigItem.INPUT_LATCH, 1)
                # interrupt enabled
                self.HAT.dio_config_write_bit(ch, DIOConfigItem.INT_MASK, 0)
                
            except (HatError, ValueError):
                print('could not configure the channel{} as output'.format(ch))
                sys.exit()
        
        
        
        
    def interrupt_callback(self,userdat):
        print('counter:{}'.format(userdat[0]))
        userdat[0] +=1          
        interrupt_ch = list([])        
        status = self.HAT.dio_int_status_read_port()
        if status !=0:            
            for i in self.di_ch:
                if (status & (1 << i)) != 0:                    
                    interrupt_ch.append(i)

            # Read the inputs to clear the active interrupt.
            dio_input_value = self.HAT.dio_input_read_port()
          
            logging.debug("counter{}-Ch:{}, port value: 0x{:02X}"\
                  .format(userdat[0],interrupt_ch, dio_input_value))
     
        return

        
    
    
    def di_acqstart(self):        
        interrupt_callback_enable(self.callback,self.interrupt_counter)       
        logging.debug('DI acqusition started')
        

            
    
    def di_acqstop(self):        
        self.HAT.dio_reset()
        interrupt_callback_disable()
        logging.debug('DI acqusition stopped')

    

# #################
def main():    
    """
    This function run mcc118 and mcc152 simultaneously
    """
    mcc152param ={'di_ch':[0], 'do_ch':[6,7]}
    mcc118param={'channels':[0,1], 'scan_rate':5000,  'timeout':5,'acq_segdur':0.1}

    dio = MCC152_DIO(mcc152param)
    daq = DAQ(mcc118param)
    
    # please comment out daq start line if you want to run mcc152 alone
    daq.worker.start()
    
    dio.di_acqstart()
    
    input("")
    daq.worker.pause()
    dio.di_acqstop()

if __name__=='__main__':
    main()



                    