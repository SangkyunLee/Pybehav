# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:58:07 2019

@author: slee
"""
import logging

from tkinter import *
import json

from os import path
from datetime import date, datetime
#from Daq import *
from mcc_daq import *
from mcc_dio import *
from Ext_trigger import *
from Timer import Timer

logging.basicConfig(level=logging.INFO)
    
class Entryframe(LabelFrame):
    def __init__(self, parent, data, framename=None):
        super().__init__(parent,text=framename)       
        self.hcomp = {}
        self.comp_dtype={}
        self.data =data
        self.framename = framename
        self.init_entrys()
        
    


    def init_entrys(self):    
        
            
        i=0
        for key,value in self.data.items():
            
            Label(self, text=key).grid(row=i)
            
            h = Entry(self)
            h.insert(0,value)
            h.grid(row=i, column=1)    
            self.hcomp[key]=h
            if type(value) == list:                
                self.comp_dtype[key]=[type(value),type(value[0])]
            else:
                self.comp_dtype[key]=[type(value)]
            i +=1  
        
        

 
    def update_par(self):
        """
        def update_par(self):
            update data structure from inputs
        """
        for key in self.data:                        
            h=self.hcomp[key]
            dtype = self.comp_dtype[key][0]
            value= h.get()
            if dtype ==list:
                dtype1 = self.comp_dtype[key][1]
                dtype_value = [dtype1(i) for i in value.split()]
                self.data[key]= dtype_value                
            else:
                self.data[key]=dtype(value)
            

            

            
            
    def entry_enable(self):
        for key in self.data:            
            h=self.hcomp[key]
            h.config(state='normal')
            
    def entry_disable(self):
        for key in self.data:            
            h=self.hcomp[key]
            h.config(state='disabled')
    
        
        


class StimTriggerGUI:
    def __init__(self, filename):        
        here = path.abspath(path.dirname(__file__))
        self.envfile = path.join(here,filename)
        with open(self.envfile) as envf:
            self.data=json.load(envf)
        
        self.hwin = Tk()
        self.hcomp ={}
        self.hframes={}
        
        self.init_frames()
        self.init_buttons()
            
    def init_frames(self):        
        
        i=0
        for key in self.data:
            
            dat1 = self.data[key]
            h = Entryframe(self.hwin,dat1,key)
            if key== 'vstimparam':
                for ename,ih in h.hcomp.items():
                    ih.config(width=50)                
                h.grid(row=i,columnspan=4, sticky=W+E, padx=5, pady=5)
            else:
                h.grid(row=i,columnspan=1, sticky=W, padx=5, pady=5)
            #h.pack( expand=1)
            self.hframes[key]=h
            i +=1
    
    def update_par(self):
        """
        def update_par(self):
            update data structure from inputs
        """
        for key,hframe in self.hframes.items():                        
            hframe.update_par()
            
            
    def entry_enable(self):
        for key,hframe in self.hframes.items():                        
            hframe.entry_enable()
            
    def entry_disable(self):
        for key,hframe in self.hframes.items():                        
            hframe.entry_disable()
        
        
    def init_buttons(self):
        nframe = len(self.data)
        h=Button(self.hwin, text='UPDATE', command=self.update_par).grid(row=nframe, column=0, sticky=W, pady=4)
        self.hcomp['update_b']=h
        h=Button(self.hwin, text='SAVE', command=self.save_env).grid(row=nframe, column=2, sticky=W, pady=4)
        self.hcomp['save_b']=h
        h=Button(self.hwin, text='Enable', command=self.entry_enable).grid(row=nframe, column=1, sticky=W, pady=4)
        self.hcomp['save_en']=h
        h=Button(self.hwin, text='RUN', command=self.run).grid(row=nframe, column=3, sticky=W, pady=4)
        self.hcomp['run']=h

    def save_env(self):
        """
        def save_env(self):
            update data structure from inputs
            and save to enviornment file
        """
        self.update_par()
        with open(self.envfile,'wt') as envf:
            j=json.dumps(self.data,indent=4)
            envf.write(j)
        self.entry_disable()   
        
    def run(self):
        
        self.save_env()
        
        trparam = self.data['trparam']
        dioparam =self.data['dioparam']
        dio = MCC152_DIO(dioparam)
        
        if 'vstimparam' in self.data:
            vstimparam = self.data['vstimparam']
            datestr = date.today().strftime('%Y-%m-%d')              
            datestr='2019-03-27'
            vstimparam['date']=datestr
            trparam['vstimparam'] = vstimparam
        
        
        if 'daqparam' in self.data:
            daqparam = self.data['daqparam']
            trparam['daq']=mcc118(daqparam)    
        
           
                    
        if 'elecparam' in self.data:
            elecparam = self.data['elecparam']
            trparam['elecstim'] = Elecstim_trigger(dio,elecparam)
        
        
        trparam['timer'] = Timer() # set timer        
        TR = Vstimbehav_trigger(dio, trparam)
        TR.gen_triggers()
        
        
            
##############
def main(): 
    #logging.basicConfig(level=logging.DEBUG)
    hf = StimTriggerGUI('Sang.json')
    mainloop( )
    
if __name__=='__main__':
    main()    
#
#envfile ='Sang.json'
##here = path.abspath(path.dirname(__file__))
##envfile = path.join(here,filename)
#with open(envfile) as envf:
#    data=json.load(envf)
#    
#    
#    
#def convert_type(data):
#    for key,value in data.items():
#        if type(data[key]) is dict:
#            data[key] = convert_type(data[key])
#        else:
#            data[key]
#            
#            
    