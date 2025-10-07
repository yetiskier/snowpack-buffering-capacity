#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:31:16 2024

Load Alpine3D model results, find date of max swe for each year, claculate change in date with Modified bootstrap technique

@author: yeti
"""

import pandas as pd
import xarray as xr
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

doy = []
mag = []
doy_cc = []
year_trend = []
for yr in range(1951, 2023):
  print('Loading year ', yr)
  print('\n')
  ds = xr.open_dataset('ModelNETcdfResults/water_year_' + str(yr) + '_Alpine3Doutput.nc')
  days=list(range(1,366))
  dowy = range(0,(len(ds.timestamp)))
  b=ds.swe.mean(dim=['lat', 'lon'], skipna=True).argmax()
  cc_max_val=ds.swe.mean(dim=['lat', 'lon'], skipna=True).max()
  cc_max_day=ds.coldcontentsnow.mean(dim=['lat', 'lon'], skipna=True).argmin()
  plt.figure(1)
  plt.plot(yr,dowy[int(b)]/8,'b.')    
  plt.plot(yr,dowy[int(cc_max_day)]/8,'c.') 
  plt.figure(2)
  plt.plot(yr,cc_max_val,'k.')   
  doy.append(dowy[int(b)]/8)  
  doy_cc.append(dowy[int(cc_max_day)]/8)
  mag.append(cc_max_val)
  year_trend.append(yr)
  
intercepts = []
mse = []
r_vals = []
coefs = []
z = []  
doy = [x for x in doy if str(x) != 'nan']
year_trend = [x for x in year_trend if str(x) != 'nan']
for k in range(0,int(5000)):  
  X_train, X_test, y_train, y_test = train_test_split(year_trend, doy,test_size=0.2, random_state=None)
  regr = linear_model.LinearRegression(fit_intercept=True)  
  regr.fit(np.asarray(X_train).reshape(-1, 1),y_train)  
  y_pred = regr.predict(np.asarray(X_test).reshape(-1, 1))
  z.append(np.polyfit(X_test,y_pred,1))
  coefs.append(regr.coef_)
  r_vals.append(regr.score(np.asarray(year_trend).reshape(-1, 1), doy))
  intercepts.append(regr.intercept_)
  mse.append(np.sqrt(mean_squared_error(doy, regr.predict(np.asarray(year_trend).reshape(-1,1)))))
mons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.figure(4)
plt.clf()
for i in range(len(z)):
  if i == 1:
      plt.plot(range(1951,2023),(range(1951,2023)*z[i][0]+z[i][1]),'c',alpha=0.1,label='Fit')
  else:
      plt.plot(range(1951,2023),(range(1951,2023)*z[i][0]+z[i][1]),'c',alpha=0.01)
plt.plot(year_trend, doy,'r+',label='Mean SWE')
plt.title('Change in peak SWE \n Basin Average = %.4f days per year '% np.nanmean(coefs))
plt.xlabel('Date')
plt.ylabel('Day of Water Year')
plt.legend(loc='best') 
plt.grid()

sigma = np.std(coefs,0)
mu = np.mean(coefs,0)
plt.figure(5)
plt.clf()
n, bins, patches = plt.hist(np.asarray(coefs),125, edgecolor='black', density=True)
norm_y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
plt.plot(bins,norm_y,'--')
plt.grid()
plt.xlim(-0.3,0.3)
plt.title('Change in max SWE')
plt.xlabel('Days per year')
plt.ylabel('Probablilty density')


intercepts = []
mse = []
r_vals = []
coefs = []
z = []  
mag = [x for x in mag if str(x) != 'nan']
year_trend = [x for x in year_trend if str(x) != 'nan']
for k in range(0,int(5000)):  
  X_train, X_test, y_train, y_test = train_test_split(year_trend, mag,test_size=0.2, random_state=None)
  regr = linear_model.LinearRegression(fit_intercept=True)  
  regr.fit(np.asarray(X_train).reshape(-1, 1),y_train)  
  y_pred = regr.predict(np.asarray(X_test).reshape(-1, 1))
  z.append(np.polyfit(X_test,y_pred,1))
  coefs.append(regr.coef_)
  r_vals.append(regr.score(np.asarray(year_trend).reshape(-1, 1), mag))
  intercepts.append(regr.intercept_)
  mse.append(np.sqrt(mean_squared_error(mag, regr.predict(np.asarray(year_trend).reshape(-1,1)))))
mons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.figure(8)
plt.clf()
for i in range(len(z)):
  if i == 1:
      plt.plot(range(1951,2023),(range(1951,2023)*z[i][0]+z[i][1]),'c',alpha=0.1,label='Fit')
  else:
      plt.plot(range(1951,2023),(range(1951,2023)*z[i][0]+z[i][1]),'c',alpha=0.01)
plt.plot(year_trend, mag,'r+',label='Mean SWE')
plt.title('Change in peak SWE \n Basin Average = %.4f mm W.E. '% np.nanmean(coefs))
plt.ylim([0, 450])
plt.xlabel('Date')
plt.ylabel('SWE (mm W.E.)')
plt.legend(loc='best') 
plt.grid()

sigma = np.std(coefs,0)
mu = np.mean(coefs,0)
plt.figure(9)
plt.clf()
n, bins, patches = plt.hist(np.asarray(coefs),125, edgecolor='black', density=True)
norm_y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
plt.plot(bins,norm_y,'--')
plt.grid()
plt.xlim(-1.3,1.3)
plt.title('Change in max SWE')
plt.xlabel('SWE (mm W.E.)')
plt.ylabel('Probablilty density')



plt.show()
