# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:12:44 2019

@author: slee
"""

import time
import numpy as np
import pandas as pd
import datetime as dt
x = np.array([1, 2, 4, 8, 10, 13, 16, 17, 18, 19, 21, 23, 25, 27])
y = np.exp(-x/30)

#d={'time':x, 'sig':y}
#df = pd.DataFrame(d)
## Convert to datetime, if necessary.
#df['time'] = pd.to_datetime(df['time'])
#
## Set the index and resample (using month start freq for compact output).
#df = df.set_index('time')
#df = df.resample('U').mean()

start = dt.datetime(year=2000,month=1,day=1)
floatseconds = map(float,x)
datetimes = map(lambda x:dt.timedelta(seconds=x)+start,floatseconds)

#construct the time series
t_s = pd.Series(y,index=datetimes)
ts={'y':t_s}
df= pd.DataFrame(ts)
out = t_s.resample('S').interpolate()
value =out.to_numpy()
t = out.index -start
t1 = [x.total_seconds() for x in t]

