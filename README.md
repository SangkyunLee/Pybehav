# Pybehav
animal-behav via daq control
This program support controcl NI-DAQ card to control dc-motor by voltage output in a time-locked manner
To control NI-DAQ, specifically USB-6008 model, use dc_motor.py


To control MCC118 & MCC 156 to record two encoder signals from wheel, estimate the speed of wheel rotation, and generate trigger signals based on visual stimulus condition and wheel speed, 
use Ext_trigger.py
Currently, I am developing a GUI for this system ( test_code/gui_test.py in the branch 
