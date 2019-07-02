# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:40:24 2019

@author: slee
"""


import threading

class Threadworker(threading.Thread):
    """ Threadworker"""

    def __init__(self, pollfunc):     
        self.pollfunc = pollfunc
        threading.Thread.__init__(self)
        self.runflag = threading.Event()  # clear this to pause thread
        self.runflag.clear()
        self.dataflag = threading.Event() 
        self.dataflag.clear()
        

    def set_datflag(self):
        self.dataflag.set()
    def clear_datflag(self):
        self.dataflag.clear()
    
    def check_datflag(self):
        return self.dataflag.is_set()
    
    
        
    def run(self):
        self.runflag.set()
        self.pollfunc()

    def stop(self):
        self.runflag.clear()
        self.dataflag.clear()

    def resume(self):
        self.runflag.set()

    def running(self):
        return (self.runflag.is_set())

    def kill(self):
        print("WORKER END")
        #sys.stdout.flush()
        #  self._Thread__stop()
        

   

    



    
 


