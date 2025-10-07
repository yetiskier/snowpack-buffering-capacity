#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swe_elevationBinSWECalculate.py
Created Jul 2024

Adapted from elevationBinColdContentCalculate.py

calculates daily Liquid Water Buffering Capacity (LWbc) (originally thought of as Residual Energy Hurdle) trend over 72 year model period
filters out data from outside elevation range called
creates all associated plots

Example command line call:
python swe_elevationBinSWECalculate.py 900 2600

Note: relative folder locations and names are HARD CODED. If the paths are not correct or
do not exist, the files will not be written, the code will crash, or both.

@author: yeti
"""
import pandas as pd
import xarray as xr
import numpy as np
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib import cm
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
    fnam = 'swe_0'+ str(month) + '_0' + str(day) +'.npy'
  elif int(month) < 10 and int(day) >= 10:
    fnam = 'swe_0'+ str(month) + '_' + str(day) +'.npy'
  elif int(day) < 10 and int(month) >= 10 :
    fnam = 'swe_'+ str(month) + '_0' + str(day) +'.npy'
  else:
    fnam = 'swe_'+ str(month) + '_' + str(day) +'.npy'
  pathnam = os.path.join('SWE_NetCDF_ByDate', fnam)
  try:
    c = np.load(pathnam)
    print('Resizing Series')
    print('\n')
    
    swe_norm = c.reshape((196,117,71))
  except:
    print('File for date does not exist.')
    print('We will construct file.')
    print('This will take several minutes.')
    swe_norm = np.zeros((196,117,71))
    count = 0 
    for yr in range(1951, 2023):
      try:
        print('Loading year ', yr)
        print('\n')
        ds = xr.open_dataset('ModelNETcdfResults/water_year_' + str(yr) + '_Alpine3Doutput.nc')
        if int(month) <10 and int(day) < 10:
          date = '0' + str(month) + '-0' + str(day) + '-' + str(yr) + 'T12:00:00'
        elif int(month)<10 and int(day)>=10:
          date = '0' + str(month) + '-' + str(day) + '-' + str(yr) + 'T12:00:00'
        elif int(month)>=10 and int(day)<10:
          date = str(month) + '-0' + str(day) + '-' + str(yr-1) + 'T12:00:00'
        else:
          date = str(month) + '-' + str(day) + '-' + str(yr-1) + 'T12:00:00'
        print('Building Array')
        print('\n')
        b = ds['swe'].sel(timestamp= date)
        print('Calculating normalized cold content')
        
        c = b.to_dataframe().swe
        print('Resizing Series')
        print('\n')
        swe_norm[:,:,count] = c.values.reshape((196,117))
        print('Done Building Array')
        print('\n')
        count+=1
      except:
        count+=1
        pass
  
    np.save(pathnam,swe_norm.reshape(swe_norm.shape[0],-1), allow_pickle=False) 
  return swe_norm

def basinAverageSWETrend(month, day, top_ele, bottom_ele, NumIter=10000):
  """
  basinAverageSWETrend
  
  calculates elevation dependent trends in SWE using a modified bootstrap linear calculated NumIter times
  each iteration fits a random 80% of the data, for the 72 years examined, there are roughly 20 Billion 80/20 sets of data
  This means that every time this code is run, the exact solution will vary slightly.
  This is OK since we are calculating the PROBABLILTY that (1) the trend exists, (2) the direction of the trend, and (3) the magnitude of the trend.
  
  The time this function takes to run is dependent on NumIter, default of 10k fits generally takes 5-10 minutes if daily array files
  from make_A3D_series already exist
  
  This function creates two figures for each day of the water year:
  (1) Average SWE on that date for each year of model run with NumIter linear fits to the data
  (2) Histogram of the slopes (coefficients) with calculated gaussian overlain.
  
  Statistical values for coefficients are printed including mean, std, var, min, max, and intercept
  
  """
  
  swe_norm = make_A3D_series(month, day)
  swe_norm = swe_norm.reshape((196,117,71))
  df = pd.read_csv('dem.csv',names=range(0,117))
  intercepts = []
  mse = []
  r_vals = []
  coefs = []
  z = []
  
  entire_basin_trend = []
  year_trend = []
  for ii in range(0,71):
    aa = swe_norm[:,:,ii]
    aa[df<bottom_ele]= np.nan
    aa[df>=top_ele] = np.nan
    if np.nanmean(aa) >= 0:
      year_trend.append(ii+1951)
      entire_basin_trend.append(np.nanmean(aa))
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
        plt.plot(range(1951,2022),(range(1951,2022)*z[i][0]+z[i][1]),'c',alpha=0.1,label='Fit')
    else:
        plt.plot(range(1951,2022),(range(1951,2022)*z[i][0]+z[i][1]),'c',alpha=0.01)
  plt.plot(year_trend, entire_basin_trend,'r+',label='Mean SWE')
  plt.title(f'{mons[int(month)-1]} {day} SWE \n Mean SWE: %.4f (mm WE)' % np.nanmean(entire_basin_trend))
  plt.ylim([0, 1300])
  plt.xlabel('Date')
  plt.ylabel('SWE (mm WE)')
  plt.legend(loc='best') 
  plt.grid()
  
  plt.savefig(os.path.join('Figures','SWEFit',f'{bottom_ele}_{top_ele}', fname), dpi=300)    
  
  sigma = np.std(coefs,0)
  mu = np.mean(coefs,0)
  plt.figure(5)
  plt.clf()
  n, bins, patches = plt.hist(np.asarray(coefs),125, edgecolor='black', density=True)
  norm_y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
  plt.plot(bins,norm_y,'--')
  plt.grid()
  plt.xlim(-2,2)
  plt.title(f'Change on {day} {mons[int(month)-1]} SWE per year trend \n (linear coefficient)')
  plt.xlabel('mm WE/year')
  plt.ylabel('Probablilty density')
  
  plt.savefig(os.path.join('Figures','SWECoefficientHistograms',f'{bottom_ele}_{top_ele}', fname), dpi=300)
  
  print('Coefficients: ', np.mean(coefs,0))
  print('StDev of Coefficients: ', np.std(coefs,0))
  print('Varience of Coefficients: ', np.var(coefs,0))
  print('Max of Coefficients: ', np.max(coefs,0))
  print('Min of Coefficients: ', np.min(coefs,0))
  print('Intercepts:', np.mean(intercepts,0))
  print('StDev of intercepts:', np.std(intercepts,0))

        
  return [coefs, r_vals, swe_norm]

def water_year_coef_plot(top_ele=3000, bottom_ele=900):
  """
  water_year_coef_plot
  
  loads and plots statistical data created by basinAverageSWETrend
  
  
  if calc_seasonal = True, it also plots original SWE curves and seasonal means of data for each year
  
  """


  try: 
    coef = np.load(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_coeffs.npy')
    stdcoef = np.load(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_stdcoeffs.npy')
    r_val = np.load(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_r_vals.npy') 
    stdr_val = np.load(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_stdr_vals.npy')
    mean_swe = np.load(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_values.npy')
  except:
  
    months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    long_months = [10, 12, 1, 3, 5, 7, 8]
    coef = []
    stdcoef = []
    r_val = []
    stdr_val = []
    mean_swe = []
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
           [coefs, r_vals, swe_norm] = basinAverageSWETrend(ii, jj, top_ele, bottom_ele)
           mean_swe.append(np.nanmean(swe_norm))
           coef.append(float(np.mean(coefs,0)))
           stdcoef.append(float(np.std(coefs,0)))
           r_val.append(float(np.mean(r_vals,0)))
           stdr_val.append(float(np.std(r_vals,0)))
         except:
           coef.append(np.nan)
           stdcoef.append(np.nan)
           r_val.append(np.nan)
           stdr_val.append(np.nan)
           mean_swe.append(np.nan)
    np.save(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_coeffs.npy', coef, allow_pickle=False) 
    np.save(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_stdcoeffs.npy', stdcoef, allow_pickle=False)
    np.save(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_r_vals.npy', r_val, allow_pickle=False)
    np.save(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_stdr_vals.npy', stdr_val, allow_pickle=False)
    np.save(f'SWEElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_swe_values.npy', mean_swe, allow_pickle=False) 
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
  plt.plot(days,np.asarray(coef),'b')
  plt.fill_between(days,np.add(coef,1.28*stdcoef),np.subtract(coef,1.28*stdcoef),alpha=0.2,facecolor='b')
  plt.fill_between(days,np.add(coef,2*stdcoef),np.subtract(coef,2*stdcoef),alpha=0.2,facecolor='b')
  plt.xticks(days[::30], dates[::30])
  
  plt.plot([endspring, endspring],[432,-432],'k--',alpha=0.5)
  plt.plot([startwinter, startwinter],[432,-432],'k--',alpha=0.5)
  plt.plot([endwinter, endwinter],[432,-432],'k--',alpha=0.5)
  plt.grid()
  plt.plot([startday,endday],[0,0],'k',alpha=0.5)
  

  plt.title(f'$\\Delta$ SWE')
  if plt_num_txt == True:
    plt.text(180,0.5,'Mean Late Accumulation SWE\n%.2f (mm WE)' % np.nanmean(mean_swe[endwinter-1:endspring-1]),bbox=dict(facecolor='white', alpha=0.95, linewidth=1),horizontalalignment='center')
    plt.text(180,2.5,'Mean $\\Delta$ Late Accumulation SWE\n%.4f (mm WE / year)' %  np.nanmean(coef[endwinter-1:endspring-1]),bbox=dict(facecolor='white', alpha=0.95, linewidth=1),horizontalalignment='center')
  
  plt.text(85,288,'Early\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(140,288,'Core\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(180,288,'Late\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(225,288,'Melt\nOnset',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.ylabel('Total change in SWE (mm WE)')
  plt.xlabel('Date')
  plt.ylim([-432,432])
  plt.xlim([startday, endday])
  plt.savefig(os.path.join('Figures','SWEElevationBinnedCoeffs', 'swe_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_water_year.png'),dpi=300)
  
  plt.figure(4)
  plt.plot(days,np.asarray(coef),'b')
  plt.fill_between(days,np.add(coef,1.28*stdcoef),np.subtract(coef,1.28*stdcoef),alpha=0.2,facecolor='b')
  plt.fill_between(days,np.add(coef,2*stdcoef),np.subtract(coef,2*stdcoef),alpha=0.2,facecolor='b')
  plt.xticks(days[::10], dates[::10])
  
  plt.grid()
  plt.plot([startday,endday],[0,0],'k',alpha=0.5)
  

  plt.title('$\\Delta$ SWE')

  plt.ylabel('Total change in SWE (mm WE)')
  plt.xlabel('Date')
  plt.ylim([-432,432])
  plt.xlim([startwinter, endwinter])
  
  plt.savefig(os.path.join('Figures','CoreWinter', 'melt_swe_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_water_year.png'),dpi=300)
 
  plt.figure(2)
  plt.plot(days,np.asarray(r_val),'b')
  plt.plot(days,np.add(r_val,2*stdr_val),'c')
  plt.plot(days,np.subtract(r_val,2*stdr_val),'c')
  plt.plot([endspring, endspring],[-0.1,0.1],'k--',alpha=0.5)
  plt.plot([startwinter, startwinter],[-0.1,0.1],'k--',alpha=0.5)
  plt.plot([endwinter, endwinter],[-0.1,0.1],'k--',alpha=0.5)
  plt.plot([startday,endday],[0,0],'k',alpha=0.5)
  plt.title(f'Elevation = {bottom_ele} to {top_ele} m \n Mean R value for fit = {np.nanmean(r_val)}')
  plt.ylabel('R value of fit')
  plt.xlabel('Day of water year')
  plt.ylim([-1,1])
  plt.xlim([0,365])
  plt.xticks(days[::60], dates[::60])
  plt.grid()
  plt.savefig(os.path.join('Figures','SWERvalueElevationBinnedCoeffs', 'swe_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_rvalue.png'),dpi=300)
  
  
  calc_seasonal = True
  if calc_seasonal == True:
    plt.figure(3)
    plt.title('SWE')
    plt.ylabel('SWE (mm WE)')
    plt.xlabel('Date')
    plt.ylim([0,1500])
    plt.xticks(days[::60], dates[::60])
    spring_mean = []
    winter_mean = []
    fall_mean = []
    for yr in range(1951, 2023):
      print('Loading year ', yr)
      print('\n')
      ds = xr.open_dataset('ModelNETcdfResults/water_year_' + str(yr) + '_Alpine3Doutput.nc')
      dowy = range(0,(len(ds.timestamp)))
      dowy = np.asarray(dowy)/8
      aa = ds['swe']
      df = pd.read_csv('dem.csv',names=range(0,117))
      aa = np.asarray(aa)
      df = np.asarray(df)
      aa[df<bottom_ele]= np.nan
      aa[df>=top_ele] = np.nan
      bb = np.nanmean(np.nanmean(aa,axis=0),axis=0)
      plt.plot(dowy,bb,'b',alpha=0.2)
      fall_mean.append(np.nanmean(bb[400:952]))
      winter_mean.append(np.nanmean(bb[953:1280]))
      spring_mean.append(np.nanmean(bb[1281:1600]))
      
      
    
    plt.xlim([0,300])
    plt.grid() 
    plt.text(85,1350,'Early\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(140,1350,'Core\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(180,1350,'Late\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(225,1350,'Melt\nOnset',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.plot([endspring, endspring],[0,1500],'k--',alpha=0.5)
    plt.plot([startwinter, startwinter],[0,1500],'k--',alpha=0.5)
    plt.plot([endwinter, endwinter],[0,1500],'k--',alpha=0.5)
    plt.plot([startday, startday],[0,1500],'k--',alpha=0.5)
    plt.plot([endday, endday],[0,1500],'k--',alpha=0.5)

    plt.plot(np.asarray(mean_swe),'k', linewidth=2)
    plt.savefig(os.path.join('Figures','SWEElevationBinnedSWE', 'swe_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
  
  
    years=range(1951,2023,1)
  
    ### Plot Fall
    plt.figure(7)
    plt.bar(years,fall_mean, color='red')
    plt.title('Mean Early Accumulation SWE\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean SWE (mm WE)')
    plt.ylim([0,1500])
    plt.grid()
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Fall_SWE_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
  
  
    ### Plot Winter
    plt.figure(9)
    plt.bar(years,winter_mean, color='blue')
    plt.title('Mean Core Accumulation SWE\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean SWE (mm WE)')
    plt.ylim([0,1500])
    plt.grid()
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Winter_SWE_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
    ### Plot Spring
    plt.figure(11)
    plt.bar(years,spring_mean, color='green')
    plt.title('Mean Late Accumulation SWE\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean SWE (mm WE)')
    plt.ylim([0,1500])
    plt.grid()
    
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Spring_SWE_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)  
  


if __name__ == "__main__":
    bottom_ele = sys.argv[1]
    top_ele = sys.argv[2]
    water_year_coef_plot(int(top_ele),int(bottom_ele))
	
