# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:49:22 2019

@author: slee
"""
from sys import version_info
from daqhats_utils import select_hat_device
from daqhats import mcc152, HatIDs, HatError, DIOConfigItem, \
    interrupt_callback_enable, HatCallback, interrupt_callback_disable


from Timer import Timer
from Threadworker import *
import logging
import pdb

class MCC152_DIO:
    def __init__(self,params):
        try:
            self.di_ch = params['di_ch']
            # convert int to tuple
            if isinstance(self.di_ch,int):
                self.di_ch = (self.di_ch,)
                
            self.do_ch = params['do_ch']
            # convert int to tuple
            if isinstance(self.do_ch,int):
                self.do_ch = (self.do_ch,)
        except KeyError:
            logging.error(KeyError, exc_info=True)
            
        self.HAT = None
        self.callback = None
        if 'timer' in params:
            self.timer = params['timer']            
        else:
            self.timer = Timer()
            
            
        
        if 'timestamping' in params:
            self.di_tstamp =[]
            self.di_series = []
            self.timestamping = True
        else:
            self.timestamping = False
            
        # data for input interruption
        # interrupt_counter: frame_counter        
        self.interrupt_counter = 0 
        self.interrupt_time = 0
        
        # since mc152 interrupt time is too slow to detect fast interrupts
        # between a millisecond, I set a sleep time that allows to get 
        # interrupt signals that occurs only longer than the min interrupt interval set below
        if 'min_interrupt_interval' in params:            
            self.min_interrupt_interval =params['min_interrupt_interval']
        else:
            self.min_interrupt_interval = 10  # millsecond
        
    
        
        self.init_dev()

        
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
                # since default output register value is 1,
                # reset to 0
                self.write_outputvalue(ch,0)
                
            except (HatError, ValueError):
                logging.error('could not configure the channel{} as output'.format(ch))
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
                logging.error('could not configure the channel{} as output'.format(ch))
                sys.exit()
        
        self.callback = HatCallback(self.interrupt_callback)
        
        
    def interrupt_callback(self,userdat):
        """
        def interrupt_callback(self,userdat):
            interrupt callback function
            This function is only for interrupt counter            
        """
        
        interrupt_ch = list([])        
        status = self.HAT.dio_int_status_read_port()
        if status !=0:            
            for i in self.di_ch:
                if (status & (1 << i)) != 0:                    
                    interrupt_ch.append(i)

            # Read the inputs to clear the active interrupt.
            dio_input_value = self.HAT.dio_input_read_port()
            if self.timestamping:
                self.di_tstamp.insert(self.interrupt_counter,self.timer.elapsed_time())
                self.di_series.insert(self.interrupt_counter,dio_input_value)
                
            interrupt_timestamp = self.timer.elapsed_time()
            framedur = interrupt_timestamp-self.interrupt_time
            
            
            
            if framedur>self.min_interrupt_interval:                                 
                self.interrupt_counter +=1
                self.interrupt_time = interrupt_timestamp
                logging.debug("DIcounter:{}-Ch:{}, port value: 0x{:02X},{}-framedur:{}"\
                  .format(self.interrupt_counter,interrupt_ch,\
                          dio_input_value,dio_input_value,framedur))
                
               
                
            

        return
    
    def get_interrupt_counter(self):
        return self.interrupt_counter
        #return self.interrupt_counter[0]
    
    def reset_timer(self):
        """
        def reset_timer(self):
            reset timer
        """
        self.timer.start()
        
    
    
    def di_acqstart(self):        
        """
        def di_acqstart(self):        
            digital input acqusition start    
        """        
          
        interrupt_callback_enable(self.callback,[])
       
        logging.info('DI acqusition started')
        

            
    
    def di_acqstop(self):        
        """
        def di_acqstop(self):        
            digital input acqusition stop    
        """
        self.HAT.dio_reset()
        interrupt_callback_disable()
        logging.info('DI acqusition stopped')



    def write_outputvalue(self, ch, value):
        """
        def write_outputvalue(self, ch, value):
        output value at channel ch
        return: timestamp
        """
        
        if bool(self.do_ch) :
            try:                
                self.HAT.dio_output_write_bit(ch,value)
                timestamp = self.timer.elapsed_time()
                logging.debug('DO ch{} : {}-tstamp:{}'.format(ch,value,timestamp))
            except (HatError, ValueError):
                raise
        else:
            logging.warning('DO channel is empty.')
        
        return timestamp
            
    def read_outputvalue(self,ch):
        """
        def read_outputvalue(self, ch, value):
        read output value at channel ch
        return: timestamp
        """
        
        if bool(self.do_ch) :
            try:                
                outputvalue = self.HAT.dio_output_read_bit(ch)
                timestamp = self.timer.elapsed_time()
                logging.debug('DO ch{} : {}-tstamp:{}'.format(ch,value,timestamp))
            except (HatError, ValueError):
                raise
        else:
            logging.error('DO channel is empty.', exc_info=True)
        
        return outputvalue, timestamp
        
                
                

  
      
##############
def main():    
    """
    This function run mcc118 and mcc152 simultaneously
    """
    

    
    from time import sleep
    dioparam = {'di_ch':1,'do_ch':6}
    DIO = MCC152_DIO(dioparam)
    while True:
        try:
            sleep(4)
            DIO.write_outputvalue(6, 1)
            sleep(1)
            DIO.write_outputvalue(6, 0)
        except KeyboardInterrupt:
            logging.info('\nExit from DIO\n')                                
            break
        
    



    
    

if __name__=='__main__':
    main()



            
            

        
            
        
            