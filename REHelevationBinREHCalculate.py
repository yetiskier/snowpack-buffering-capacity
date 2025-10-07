#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REHelevationBinREHCalculate.py
Created Jul 2024

Adapted from elevationBinColdContentCalculate.py

calculates daily Liquid Water Buffering Capacity (LWbc) (originally thought of as Residual Energy Hurdle) trend over 72 year model period
filters out data from outside elevation range called
creates all associated plots

Example command line call:
python REHelevationBinREHCalculate.py 900 2600

Note: relative folder locations and names are HARD CODED. If the paths are not correct or
do not exist, the files will not be written, the code will crash, or both.

@author: yeti
"""
import pandas as pd
import xarray as xr
import numpy as np
from sklearn import linear_model 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import sys
import os

def make_A3D_series(month,day):
  """
  make_A3D_series
  
  creates array for each day of the water year
  the array includes a 196 x 117 grid of data (nan for no data) for each year.
  This function takes a long time to run, therefore, the daily data is saved.
  If daily data already exists, this function loads and resizes the data.
  
  This function requires model NetCDF results from Alpine3D model output compiled with:
  consolidateA3Douput.py
  
  """

  if int(month) < 10 and int(day) < 10:
    fnam = 'REH_0'+ str(month) + '_0' + str(day) +'.npy'
  elif int(month) < 10 and int(day) >= 10:
    fnam = 'REH_0'+ str(month) + '_' + str(day) +'.npy'
  elif int(day) < 10 and int(month) >= 10 :
    fnam = 'REH_'+ str(month) + '_0' + str(day) +'.npy'
  else:
    fnam = 'REH_'+ str(month) + '_' + str(day) +'.npy'
  pathnam = os.path.join('REHByDate', fnam)
  try:
    c = np.load(pathnam)
    print('Resizing Series')
    print('\n')
    
    REH_norm = c.reshape((196,117,72))
  except:
    print('File for date does not exist.')
    print('We will construct file.')
    print('This will take several minutes.')
    REH_norm = np.zeros((196,117,72))
    count = 0 
    for yr in range(1951, 2023):
      try:
        print('Loading year ', yr)
        print('\n')
        ds = xr.open_dataset('ModelNETcdfResults/water_year_' + str(yr) + '_Alpine3Doutput.nc')
        if int(month) <10 and int(day) < 10:
          date = '0' + str(month) + '-0' + str(day) + '-' + str(yr) + 'T12:00:00.000000000'
        elif int(month)<10 and int(day)>=10:
          date = '0' + str(month) + '-' + str(day) + '-' + str(yr) + 'T12:00:00.000000000'
        elif int(month)>=10 and int(day)<10:
          date = str(month) + '-0' + str(day) + '-' + str(yr-1) + 'T12:00:00.000000000'
        else:
          date = str(month) + '-' + str(day) + '-' + str(yr-1) + 'T12:00:00.000000000'
        print('Building Array')
        print('\n')
        print('Calculating REH')
        
        
        s_w = ds.liquidwatercontent.where(ds.swe>0.001)/(ds.swe.where(ds.swe >0.001)+ds.liquidwatercontent.where(ds.swe >0.001))
        s_w_min = s_w.where(ds.coldcontentsnow==0).where(ds.runoff>0).where(ds.swe>0).where(ds.liquidwatercontent>0).min(dim='timestamp')
        s_w_min = s_w_min.values[:,:,np.newaxis]
        ddds = ds.swe.values * s_w_min
        dds = ds.swe.values * (np.ones(np.shape(s_w_min))*0.07)
        dds[dds==0]=np.nan
        aa = (ds['coldcontentsnow']/0.334) - (ddds - ds['liquidwatercontent'])
        
        
        
        
        
        c = (ds.coldcontentsnow.where(ds.swe>0.1).sel(timestamp=date)/.334) - ((ds.swe.where(ds.swe >0.001).sel(timestamp=date)*s_w_min[:,:,0])-ds.liquidwatercontent.where(ds.swe>0.001).sel(timestamp=date)) 
        print('Resizing Series')
        print('\n')
        REH_norm[:,:,count] = c
        print('Done Building Array')
        print('\n')
        count+=1
      except:
        count+=1
        pass
  
    np.save(pathnam,REH_norm.reshape(REH_norm.shape[0],-1), allow_pickle=False) 
  return REH_norm

def basinAverageREHTrend(month, day, top_ele, bottom_ele, NumIter=10000):
  """
  basinAverageREHTrend
  
  calculates elevation dependent trends in LWbc using a modified bootstrap linear calculated NumIter times
  each iteration fits a random 80% of the data, for the 72 years examined, there are roughly 20 Billion 80/20 sets of data
  This means that every time this code is run, the exact solution will vary slightly.
  This is OK since we are calculating the PROBABLILTY that (1) the trend exists, (2) the direction of the trend, and (3) the magnitude of the trend.
  
  The time this function takes to run is dependent on NumIter, default of 10k fits generally takes 5-10 minutes if daily array files
  from make_A3D_series already exist
  
  This function creates two figures for each day of the water year:
  (1) Average LWbc on that date for each year of model run with NumIter linear fits to the data
  (2) Histogram of the slopes (coefficients) with calculated gaussian overlain.
  
  Statistical values for coefficients are printed including mean, std, var, min, max, and intercept
  
  """
  
  
  REH_norm = make_A3D_series(month, day)
  REH_norm = REH_norm.reshape((196,117,72))
  REH_norm[REH_norm > 0] = 0 
  df = pd.read_csv('dem.csv',names=range(0,117))
  intercepts = []
  r_vals = []
  coefs = []
  z = []
  
  entire_basin_trend = []
  year_trend = []
  for ii in range(0,72):
    aa = REH_norm[:,:,ii]
    aa[df<bottom_ele]= np.nan
    aa[df>=top_ele] = np.nan
    if np.nanmean(aa) <=0:
      entire_basin_trend.append(np.nanmean(aa))
      year_trend.append(ii+1951)
    else:
      entire_basin_trend.append('nan')
      year_trend.append('nan')
  
  for k in range(0,int(NumIter)):  
    entire_basin_trend = [x for x in entire_basin_trend if str(x) != 'nan']
    year_trend = [x for x in year_trend if str(x) != 'nan']
    X_train, X_test, y_train, y_test = train_test_split(year_trend, entire_basin_trend,test_size=0.2, random_state=None)
    regr = linear_model.LinearRegression(fit_intercept=True)  
    regr.fit(np.asarray(X_train).reshape(-1, 1),y_train)  
    y_pred = regr.predict(np.asarray(X_test).reshape(-1, 1))
    z.append(np.polyfit(X_test,y_pred,1))
    coefs.append(regr.coef_)
    r_vals.append(regr.score(np.asarray(year_trend).reshape(-1, 1), entire_basin_trend))
    intercepts.append(regr.intercept_)

  mons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  fname = str(month) +'_' + str(mons[int(month)-1]) + '_' + str(day) + '_' + str(bottom_ele) + '_' + str(top_ele) + '.png'
  plt.figure(4)
  plt.clf()
  for i in range(len(z)):
    if i == 1:
        plt.plot(range(1951,2023),(range(1951,2023)*z[i][0]+z[i][1]),'c',alpha=0.5,label='Fit')
    else:
        plt.plot(range(1951,2023),(range(1951,2023)*z[i][0]+z[i][1]),'c',alpha=0.01)
  plt.plot(year_trend, entire_basin_trend,'r+',label='$LW_{bc}$')
  plt.title(str(mons[int(month)-1]) + ' ' + str(day) + ' ' + ' $LW_{bc}$ \n Mean magnitude $LW_{bc}$ = %.4f (mm WE)' % np.nanmean(entire_basin_trend))
  plt.xlabel('Year')
  plt.ylabel('$LW_{bc}$ (mm WE)')
  plt.legend(loc='best') 
  plt.grid()
  
  plt.savefig(os.path.join('Figures','REHFit',f'{bottom_ele}_to_{top_ele}', fname), dpi=300)    
  
  sigma = np.std(coefs,0)
  mu = np.mean(coefs,0)
  plt.figure(5)
  plt.clf()
  n, bins, patches = plt.hist(np.asarray(coefs),125, edgecolor='black', density=True)
  norm_y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
  plt.plot(bins,norm_y,'--')
  plt.grid()
  plt.xlim(-0.4,0.4)
  plt.title('Histogram of $LW_{bc}$ per year trend on ' + str(day) + ' ' + str(mons[int(month)-1]))
  plt.xlabel('mm WE/year')
  plt.ylabel('Probablilty density')
  
  plt.savefig(os.path.join('Figures','REHCoefficientHistograms',f'{bottom_ele}_to_{top_ele}', fname), dpi=300)
  
  print('Coefficients: ', np.mean(coefs,0))
  print('StDev of Coefficients: ', np.std(coefs,0))
  print('Varience of Coefficients: ', np.var(coefs,0))
  print('Max of Coefficients: ', np.max(coefs,0))
  print('Min of Coefficients: ', np.min(coefs,0))
  print('Intercepts:', np.mean(intercepts,0))
  print('StDev of intercepts:', np.std(intercepts,0))

        
  return [coefs, r_vals, REH_norm]

def water_year_coef_plot(top_ele=2600, bottom_ele=900):
  """
  water_year_coef_plot
  
  loads and plots statistical data created by basinAverageREHTrend
  
  
  if calc_seasonal = True, it also plots original LWbc curves and seasonal means of data for each year
  
  """


  try: 
    coef = np.load(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_coeffs.npy')
    stdcoef = np.load(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_stdcoeffs.npy')
    r_val = np.load(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_r_vals.npy') 
    stdr_val = np.load(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_stdr_vals.npy')
    mean_REH = np.load(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_values.npy')
  except:
  
    months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    long_months = [10, 12, 1, 3, 5, 7, 8]
    coef = []
    stdcoef = []
    r_val = []
    stdr_val = []
    mean_REH = []
    for ii in months:
      if ii in long_months:
         days = range(1,32)
      elif ii == 2:
         days = range(1,29)
      else:
         days = range(1,31)
      for jj in days:
         print("Month = ", ii, " Day = ", jj)
         try:
           [coefs, r_vals, REH_norm] = basinAverageREHTrend(ii, jj, top_ele, bottom_ele)
           mean_REH.append(np.nanmean(REH_norm))
           coef.append(float(np.mean(coefs,0)))
           stdcoef.append(float(np.std(coefs,0)))
           r_val.append(float(np.mean(r_vals,0)))
           stdr_val.append(float(np.std(r_vals,0)))
         except:
           coef.append(np.nan)
           stdcoef.append(np.nan)
           r_val.append(np.nan)
           stdr_val.append(np.nan)
           mean_REH.append(np.nan)
    np.save(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_coeffs.npy', coef, allow_pickle=False) 
    np.save(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_stdcoeffs.npy', stdcoef, allow_pickle=False)
    np.save(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_r_vals.npy', r_val, allow_pickle=False)
    np.save(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_stdr_vals.npy', stdr_val, allow_pickle=False)
    np.save(f'REHElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_REH_values.npy', mean_REH, allow_pickle=False) 
  days=list(range(1,366))
  dates=pd.read_csv('Dowy.csv').to_numpy()[:,1]
  startday=50
  startwinter=120
  endwinter=160
  endspring=200
  endday=250
  coef = coef*72
  stdcoef = stdcoef*72
  plt_num_txt = False
  
  
  plt.figure(1)
  plt.plot(days,np.asarray(coef),'g')
  plt.fill_between(days,np.add(coef,1.28*stdcoef),np.subtract(coef,1.28*stdcoef),alpha=0.2,facecolor='g')
  plt.fill_between(days,np.add(coef,2*stdcoef),np.subtract(coef,2*stdcoef),alpha=0.2,facecolor='g')
  plt.xticks(days[::30], dates[::30])
  plt.plot([endspring, endspring],[-29,29],'k--',alpha=0.5)
  plt.plot([startwinter, startwinter],[-29,29],'k--',alpha=0.5)
  plt.plot([endwinter, endwinter],[-29,29],'k--',alpha=0.5)
  plt.grid()
  plt.plot([startday,endday],[0,0],'k',alpha=0.5)
  plt.title('$\\Delta LW_{bc}$')
  if plt_num_txt == True:
    
    plt.text(85,-0.15,'$LW_{bc}$\n%.2f' % np.nanmean(mean_REH[startday-1:startwinter-1]),horizontalalignment='center')
    plt.text(85,-0.025,'$\\Delta LW_{bc}$\n%.4f' %  np.nanmean(coef[startday-1:startwinter-1]),horizontalalignment='center')
    
    plt.text(140,-0.15,'$LW_{bc}$\n%.2f' % np.nanmean(mean_REH[startwinter-1:endwinter-1]),horizontalalignment='center')
    plt.text(140,-0.025,'$\\Delta LW_{bc}$\n%.4f' %  np.nanmean(coef[startwinter-1:endwinter-1]),horizontalalignment='center')
    
    plt.text(180,-0.15,'$LW_{bc}$\n%.2f' % np.nanmean(mean_REH[endwinter-1:endspring-1]),horizontalalignment='center')
    plt.text(180,-0.025,'$\\Delta LW_{bc}$\n%.4f' %  np.nanmean(coef[endwinter-1:endspring-1]),horizontalalignment='center')
    
    plt.text(225,-0.15,'$LW_{bc}$\n%.2f' % np.nanmean(mean_REH[endspring-1:endday-1]),horizontalalignment='center')
    plt.text(225,-0.025,'$\\Delta LW_{bc}$\n%.4f' %  np.nanmean(coef[endspring-1:endday-1]),horizontalalignment='center')
    
  plt.text(85,-21.6,'Early\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(140,-21.6,'Core\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(180,-21.6,'Late\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(225,-21.6,'Melt\nOnset',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')

  plt.ylabel('Total change in $LW_{bc}$ (mm WE)')
  plt.xlabel('Date')
  plt.ylim([29,-29])
  plt.xlim([startday, endday])
  plt.savefig(os.path.join('Figures','REHElevationBinnedCoeffs', 'REH_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_water_year.png'),dpi=300)
 
  plt.figure(12)
  plt.plot(days,np.asarray(coef),'g')
  plt.fill_between(days,np.add(coef,1.28*stdcoef),np.subtract(coef,1.28*stdcoef),alpha=0.2,facecolor='g')
  plt.fill_between(days,np.add(coef,2*stdcoef),np.subtract(coef,2*stdcoef),alpha=0.2,facecolor='g')
  plt.xticks(days[::10], dates[::10])
  plt.grid()
  plt.plot([startday,endday],[0,0],'k',alpha=0.5)
  plt.title('$\\Delta LW_{bc}$')
  plt.ylabel('Coefficient of change in $LW_{bc}$ (mm WE)')
  plt.xlabel('Date')
  plt.ylim([0.4,-0.4])
  plt.xlim([startwinter, endwinter])
  plt.savefig(os.path.join('Figures','CoreWinter', 'melt_REH_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_water_year.png'),dpi=300)

  plt.figure(2)
  plt.plot(days,np.asarray(r_val),'b')
  plt.plot(days,np.add(r_val,2*stdr_val),'c')
  plt.plot(days,np.subtract(r_val,2*stdr_val),'c')
  plt.xticks(days[::30], dates[::30])
  plt.plot([startday, startday],[-0.4,0.4],'k:',alpha=0.5)
  plt.plot([startwinter, startwinter],[-0.4,0.4],'k--',alpha=0.5)
  plt.plot([endwinter, endwinter],[-0.4,0.4],'k--',alpha=0.5)
  plt.plot([endday, endday],[-0.4,0.4],'k:',alpha=0.5)
  plt.title(f'Elevation = {bottom_ele} to {top_ele} m \n Mean R value for fit = {np.nanmean(r_val)}')
  plt.ylabel('R value of fit')
  plt.xlabel('Date')
  plt.ylim([-1,1])
  plt.xlim([0,365])
  plt.xticks(days[::60], dates[::60])
  plt.grid()
  plt.savefig(os.path.join('Figures','REHRvalueElevationBinnedCoeffs', 'REH_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_rvalue.png'),dpi=300)

  calc_seasonal = True
  if calc_seasonal == True:  
    plt.figure(3)
    plt.title('$LW_{bc}$' % np.nanmean(mean_REH[startday:endday]))
    plt.ylabel('$LW_{bc}$ (mm WE)')
    plt.xlabel('Date')
    plt.ylim([0,-100])
    plt.xlim([0,300])
    plt.xticks(days[::60], dates[::60])
    spring_zeros = []
    winter_zeros = []
    fall_zeros = []
    spring_mean = []
    winter_mean = []
    fall_mean = []
    for yr in range(1951, 2023):
      plt.xlim([0,300])
      plt.grid()
      print('Loading year ', yr)
      print('\n')
      ds = xr.open_dataset('ModelNETcdfResults/water_year_' + str(yr) + '_Alpine3Doutput.nc')
      dowy = range(0,(len(ds.timestamp)))
      dowy = np.asarray(dowy)/8
      
      s_w = ds.liquidwatercontent.where(ds.swe>0.001)/(ds.swe.where(ds.swe >0.001)+ds.liquidwatercontent.where(ds.swe >0.001))
      s_w_min = s_w.where(ds.coldcontentsnow==0).where(ds.runoff>0).where(ds.swe>0).where(ds.liquidwatercontent>0).min(dim='timestamp')
      s_w_min = s_w_min.values[:,:,np.newaxis]
      ddds = ds.swe.values * s_w_min
      dds = ds.swe.values * (np.ones(np.shape(s_w_min))*0.07)
      dds[dds==0]=np.nan
      aa = (ds['coldcontentsnow']/0.334) - (ddds - ds['liquidwatercontent'])
      df = pd.read_csv('dem.csv',names=range(0,117))
      aa = np.asarray(aa)
      aa[aa>0] = 0
      df = np.asarray(df)
      aa[df<bottom_ele]= np.nan
      aa[df>=top_ele] = np.nan
      bb = np.nanmean(np.nanmean(aa,axis=0),axis=0)
      fall_zeros.append(np.count_nonzero(bb[400:959]==0))
      winter_zeros.append(np.count_nonzero(bb[960:1279]==0))
      spring_zeros.append(np.count_nonzero(bb[1280:1600]==0))
      plt.plot(dowy,bb,'g',alpha=0.2)
      fall_mean.append(np.nanmean(bb[400:952]))
      winter_mean.append(np.nanmean(bb[953:1280]))
      spring_mean.append(np.nanmean(bb[1281:1600]))
      
      
    plt.grid()  
    plt.text(85,-90,'Early\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(140,-90,'Core\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(180,-90,'Late\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(225,-90,'Melt\nOnset',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.plot([endspring, endspring],[-100,0],'k--',alpha=0.5)
    plt.plot([startwinter, startwinter],[-100,0],'k--',alpha=0.5)
    plt.plot([endwinter, endwinter],[-100,0],'k--',alpha=0.5)
    plt.plot([startday, startday],[-100,0],'k--',alpha=0.5)
    plt.plot([endday, endday],[-100,0],'k--',alpha=0.5)
    
    
    plt.plot(np.asarray(mean_REH), 'k', linewidth=2)
    plt.savefig(os.path.join('Figures','ElevationBinnedREH', 'REH_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    print('early years Early Accum. coefficient of variation = ',str(((np.std(fall_mean[0:36])/np.mean(fall_mean[0:35]))*100)))
    print('late years Early Accum. coefficient of variation = ',str(((np.std(fall_mean[36:73])/np.mean(fall_mean[36:73]))*100)))
    print('early years Core Accum. coefficient of variation = ',str(((np.std(winter_mean[0:36])/np.mean(winter_mean[0:35]))*100)))
    print('late years Core Accum. coefficient of variation = ',str(((np.std(winter_mean[36:73])/np.mean(winter_mean[36:73]))*100)))
    print('early years Late Accum. coefficient of variation = ',str(((np.std(spring_mean[0:36])/np.mean(spring_mean[0:35]))*100)))
    print('late years Late Accum. coefficient of variation = ',str(((np.std(spring_mean[36:73])/np.mean(spring_mean[36:73]))*100)))
    years=range(1951,2023,1)
    
    ### Plot Fall
    plt.figure(6)
    plt.bar(years,np.divide(fall_zeros,8), color='red')
    plt.title('Early Accumulation days at zero $LW_{bc}$\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Number of days at zero $LW_{bc}$')
    plt.ylim([0,50])
    plt.grid()
    early_fall = np.divide(fall_zeros[0:36],8)
    late_fall = np.divide(fall_zeros[36:73],8)
    print(fall_zeros)
    plt.text(1967,45,'Water year 1951 - 1986\n%.0f' %  sum(i >= 1 for i in early_fall),bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(2006,45,'Water year 1987 - 2022\n%.0f' %  sum(i >= 1 for i in late_fall),bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Fall_Days_at_zero_REH_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
    plt.figure(7)
    plt.bar(years,fall_mean, color='red')
    plt.title('Mean Early Accumulation $LW_{bc}$\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean $LW_{bc}$ (mm WE)')
    plt.ylim([0,-70])
    plt.grid()
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Fall_REH_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
    ### Plot Winter
    plt.figure(8)
    plt.bar(years,np.divide(winter_zeros,8), color='blue')
    plt.title('Core Accumulation days at zero $LW_{bc}$\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Number of days at zero $LW_{bc}$')
    plt.ylim([0,50])
    plt.grid()
    early_winter = np.divide(winter_zeros[0:36],8)
    late_winter = np.divide(winter_zeros[36:73],8)
    plt.text(1967,45,'Water year 1951 - 1986\n%.0f' %  sum(i >= 1 for i in early_winter),bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(2006,45,'Water year 1987 - 2022\n%.0f' %  sum(i >= 1 for i in late_winter),bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Winter_Days_at_zero_REH_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
    
    plt.figure(9)
    plt.bar(years,winter_mean, color='blue')
    plt.title('Mean Core Accumulation $LW_{bc}$\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean $LW_{bc}$ (mm WE)')
    plt.ylim([0,-70])
    plt.grid()
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Winter_REH_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
    ### Plot Spring
    plt.figure(10)
    plt.bar(years,np.divide(spring_zeros,8), color='green')
    plt.title('Late Accumulation days at zero $LW_{bc}$\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Number of days at zero $LW_{bc}$')
    plt.ylim([0,50])
    plt.grid()
    early_spring = np.divide(spring_zeros[0:36],8)
    late_spring = np.divide(spring_zeros[36:73],8)
    print(spring_zeros)
    print(bottom_ele, ' early years ', sum(i >= 1 for i in early_spring))
    print(bottom_ele, ' late years ', sum(i >= 1 for i in late_spring))
    plt.text(1967,45,'Water year 1951 - 1986\n%.0f' %  sum(i >= 1 for i in early_spring),bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(2006,45,'Water year 1987 - 2022\n%.0f' %  sum(i >= 1 for i in late_spring),bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Spring_Days_at_zero_REH_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
    
    plt.figure(11)
    plt.bar(years,spring_mean, color='green')
    plt.title('Mean Late Accumulation $LW_{bc}$\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean $LW_{bc}$ (mm WE)')
    plt.ylim([0,-70])
    plt.grid()
    
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Spring_REH_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
  
  
  

if __name__ == "__main__":
    bottom_ele = sys.argv[1]
    top_ele = sys.argv[2]
    water_year_coef_plot(int(top_ele),int(bottom_ele))
  
