%% pulse generation
clear all
model = daq.getDevices;
daqses=daq.createSession('ni');
ch=addCounterOutputChannel(daqses,'Dev1', 0, 'PulseGeneration');
ch1=addCounterOutputChannel(daqses,'Dev1', 'ctr1', 'PulseGeneration');


ch = daqses.Channels(1);
ch.Frequency = 10;
ch.InitialDelay = 0;
ch.DutyCycle =0.99%0.99; % 0.001%

% ch1 = daqses.Channels(2);
% ch1.Frequency = 0.5;
% ch1.InitialDelay = 0.33;
% ch1.DutyCycle =0.008%0.99; % 0.001%

% 
daqses.Rate = 5000;
daqses.DurationInSeconds =600;
[data, time] = daqses.startForeground();
% 
% % StartForeground returns data for input channels only. The data variable
% % will contain one column of data.n
% figure
% plot(time, data);
%

ch = daqses.Channels(1);
ch.Frequency = 200;
ch.InitialDelay = 0;
ch.DutyCycle =1/5%0.99; % 0.001%

daqses.Rate = 5000;
daqses.DurationInSeconds =500;
[data, time] = daqses.startForeground();

%%
clear all
model = daq.getDevices;
daqses=daq.createSession('ni');
ch = addDigitalChannel(daqses,'Dev1','Port0/Line0:1','OutputOnly');
st=0.001
for i=1:10
    outputSingleScan(daqses,[1,0])
    pause(st)
    outputSingleScan(daqses,[0,0])
    pause(3-st)
end

%% recording
clear all
model = daq.getDevices;
daqses=daq.createSession('ni');
% ch=addCounterOutputChannel(daqses,'Dev1', 0, 'PulseGeneration');
daqses.addAnalogInputChannel('Dev1','ai0','Voltage');
% daqses.addAnalogInputChannel('Dev1','ai1','Voltage');
% 
% daqses.addAnalogInputChannel('Dev2','ai0','Voltage');
% % daqses.addAnalogInputChannel('Dev2','ai1','Voltage');

% ch = daqses.Channels(1);
% ch.Frequency = 1/2;
% ch.InitialDelay = 0;
% ch.DutyCycle = 0.99;


% 
daqses.Rate = 10000;
daqses.DurationInSeconds =100;

% 
% % StartForeground returns data for input channels only. The data variable
% % will contain one column of data.
[data, time] = daqses.startForeground();
figure
plot(time, data);



params.daqses.NotifyWhenDataAvailableExceeds=500;






