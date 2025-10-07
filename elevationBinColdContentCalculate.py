#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
elevationBinColdContentCalculate.py
Created on Thu Jul  4 23:31:16 2024

calculates daily cold content trend over 72 year model period
filters out data from outside elevation range called
creates all associated plots

Example command line call:
python elevationBinColdContentCalculate.py 900 2600

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
    fnam = 'cc_0'+ str(month) + '_0' + str(day) +'.npy'
  elif int(month) < 10 and int(day) >= 10:
    fnam = 'cc_0'+ str(month) + '_' + str(day) +'.npy'
  elif int(day) < 10 and int(month) >= 10 :
    fnam = 'cc_'+ str(month) + '_0' + str(day) +'.npy'
  else:
    fnam = 'cc_'+ str(month) + '_' + str(day) +'.npy'
  pathnam = os.path.join('NormalizedColdContentByDate', fnam)
  try:
    c = np.load(pathnam)
    print('Resizing Series')
    print('\n')
    
    cc_norm = c.reshape((196,117,71))
  except:
    print('File for date does not exist.')
    print('We will construct file.')
    print('This will take several minutes.')
    cc_norm = np.zeros((196,117,71))
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
            
            a = ds['coldcontentsnow'].sel(timestamp = date) 
            b = ds['swe'].sel(timestamp= date)
            print('Calculating normalized cold content')
            
            c = (a.to_dataframe().coldcontentsnow.where(b.to_dataframe().swe>0.1))
            print('Resizing Series')
            print('\n')
            cc_norm[:,:,count] = c.values.reshape((196,117))
            print('Done Building Array')
            print('\n')
            count+=1
        except:
            count+=1
            pass
  
    np.save(pathnam,cc_norm.reshape(cc_norm.shape[0],-1), allow_pickle=False) 
  return cc_norm

def basinAverageColdContentTrend(month, day, top_ele, bottom_ele, NumIter=10000):
  """
  basinAverageColdContentTrend
  
  calculates elevation dependent trends in cold content using a modified bootstrap linear calculated NumIter times
  each iteration fits a random 80% of the data, for the 72 years examined, there are roughly 20 Billion 80/20 sets of data
  This means that every time this code is run, the exact solution will vary slightly.
  This is OK since we are calculating the PROBABLILTY that (1) the trend exists, (2) the direction of the trend, and (3) the magnitude of the trend.
  
  The time this function takes to run is dependent on NumIter, default of 10k fits generally takes 5-10 minutes if daily array files
  from make_A3D_series already exist
  
  This function creates two figures for each day of the water year:
  (1) Average CC on that date for each year of model run with NumIter linear fits to the data
  (2) Histogram of the slopes (coefficients) with calculated gaussian overlain.
  
  Statistical values for coefficients are printed including mean, std, var, min, max, and intercept
  
  """
  
  cc_norm = make_A3D_series(month, day)
  cc_norm = cc_norm.reshape((196,117,71))
  cc_norm[cc_norm > 0] = np.nan 
  df = pd.read_csv('dem.csv',names=range(0,117))
  intercepts = []
  mse = []
  r_vals = []
  coefs = []
  z = []
  
  entire_basin_trend = []
  year_trend = []
  for ii in range(0,71):
    aa = cc_norm[:,:,ii]
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
        plt.plot(range(1951,2022),(range(1951,2022)*z[i][0]+z[i][1]),'c',alpha=0.1,label='Fit')
    else:
        plt.plot(range(1951,2022),(range(1951,2022)*z[i][0]+z[i][1]),'c',alpha=0.01)
  plt.plot(year_trend, entire_basin_trend,'r+',label='Mean cold content')
  plt.title(f'{mons[int(month)-1]} {day} cold content \n Mean Cold Content: %.4f ($MJ/m^2$)' % np.nanmean(entire_basin_trend))
  plt.ylim([-15, 0])
  plt.xlabel('Date')
  plt.ylabel('Normalized Cold Content (mm WE)')
  plt.legend(loc='best') 
  plt.grid()
  
  plt.savefig(os.path.join('Figures','ColdContentFit',f'{bottom_ele}_to_{top_ele}', fname), dpi=300)    
  
  sigma = np.std(coefs,0)
  mu = np.mean(coefs,0)
  plt.figure(5)
  plt.clf()
  n, bins, patches = plt.hist(np.asarray(coefs),125, edgecolor='black', density=True)
  norm_y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
  plt.plot(bins,norm_y,'--')
  plt.grid()
  plt.xlim(-0.05,0.05)
  plt.title(f'Change on {day} {mons[int(month)-1]} cold content per year trend \n (linear coefficient)')
  plt.xlabel('mm WE/year')
  plt.ylabel('Probablilty density')
  
  plt.savefig(os.path.join('Figures','CoefficientHistograms',f'{bottom_ele}_to_{top_ele}', fname), dpi=300)
  print('Coefficients: ', np.mean(coefs,0))
  print('StDev of Coefficients: ', np.std(coefs,0))
  print('Varience of Coefficients: ', np.var(coefs,0))
  print('Max of Coefficients: ', np.max(coefs,0))
  print('Min of Coefficients: ', np.min(coefs,0))
  print('Intercepts:', np.mean(intercepts,0))
  print('StDev of intercepts:', np.std(intercepts,0))

        
  return [coefs, r_vals, cc_norm]

def water_year_coef_plot(top_ele=2600, bottom_ele=900):
  """
  water_year_coef_plot
  
  loads and plots statistical data created by basinAverageColdContentTrend
  
  
  if calc_seasonal = True, it also plots original CC curves and seasonal means of data for each year
  
  """


  try: 
    coef = np.load(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_coeffs.npy')
    stdcoef = np.load(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_stdcoeffs.npy')
    r_val = np.load(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_r_vals.npy') 
    stdr_val = np.load(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_stdr_vals.npy')
    mean_cc = np.load(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_values.npy')
  except:
  
    months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    long_months = [10, 12, 1, 3, 5, 7, 8]
    coef = []
    stdcoef = []
    r_val = []
    stdr_val = []
    mean_cc = []
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
           [coefs, r_vals, cc_norm] = basinAverageColdContentTrend(ii, jj, top_ele, bottom_ele)
           mean_cc.append(np.nanmean(cc_norm))
           coef.append(float(np.mean(coefs,0)))
           stdcoef.append(float(np.std(coefs,0)))
           r_val.append(float(np.mean(r_vals,0)))
           stdr_val.append(float(np.std(r_vals,0)))
         except:
           coef.append(np.nan)
           stdcoef.append(np.nan)
           r_val.append(np.nan)
           stdr_val.append(np.nan)
           mean_cc.append(np.nan)
    np.save(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_coeffs.npy', coef, allow_pickle=False) 
    np.save(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_stdcoeffs.npy', stdcoef, allow_pickle=False)
    np.save(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_r_vals.npy', r_val, allow_pickle=False)
    np.save(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_stdr_vals.npy', stdr_val, allow_pickle=False)
    np.save(f'ElevationAnalysisOutput/{bottom_ele}_to_{top_ele}_true_cc_values.npy', mean_cc, allow_pickle=False) 
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
  plt.plot(days,np.asarray(coef/0.334),'k')
  plt.fill_between(days,np.add(coef,1.28*stdcoef)/0.334,np.subtract(coef,1.28*stdcoef)/0.334,alpha=0.2,facecolor='k')
  plt.fill_between(days,np.add(coef,2*stdcoef)/0.334,np.subtract(coef,2*stdcoef)/0.334,alpha=0.2,facecolor='k')
  plt.xticks(days[::30], dates[::30])
  
  plt.plot([endspring, endspring],[-7.2,7.2],'k--',alpha=0.5)
  plt.plot([startwinter, startwinter],[-7.2,7.2],'k--',alpha=0.5)
  plt.plot([endwinter, endwinter],[-7.2,7.2],'k--',alpha=0.5)
  plt.grid()
  plt.plot([startday,endday],[0,0],'k',alpha=0.5)
  
  plt.title('$\\Delta$ CC')
  if plt_num_txt == True:
    plt.text(180,-0.025,'Mean Late Accumulation CC\n%.2f (mm WE)' % np.nanmean(mean_cc[endwinter-1:endspring-1]),bbox=dict(facecolor='white', alpha=0.95, linewidth=1),horizontalalignment='center')
    plt.text(180,-0.05,'Mean $\\Delta$ Late Accumulation CC\n%.4f (mm WE / year)' %  np.nanmean(coef[endwinter-1:endspring-1]),bbox=dict(facecolor='white', alpha=0.95, linewidth=1),horizontalalignment='center')
  
  plt.text(85,-5.256,'Early\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(140,-5.256,'Core\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(180,-5.256,'Late\nAccum.',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
  plt.text(225,-5.256,'Melt\nOnset',fontsize=15,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')

  plt.ylabel('Total change in CC (mm WE)')
  plt.xlabel('Date')
  plt.ylim([7.2,-7.2])
  plt.xlim([startday, endday])
  plt.savefig(os.path.join('Figures','ElevationBinnedCoeffs', 'cc_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_water_year.png'),dpi=300)
 
  plt.figure(4)
  plt.plot(days,np.asarray(coef/0.334),'k')
  plt.fill_between(days,np.add(coef,1.28*stdcoef)/0.334,np.subtract(coef,1.28*stdcoef)/0.334,alpha=0.2,facecolor='k')
  plt.fill_between(days,np.add(coef,2*stdcoef)/0.334,np.subtract(coef,2*stdcoef)/0.334,alpha=0.2,facecolor='k')
  plt.xticks(days[::10], dates[::10])
  plt.grid()
  plt.plot([startday,endday],[0,0],'k',alpha=0.5)
  plt.title(f'$\\Delta$ CC Core Accumulation\nElevation = {bottom_ele} to {top_ele} m')
  plt.ylabel('Coefficient of change in cold content (mm WE)')
  plt.xlabel('Date')
  plt.ylim([7.2,-7.2])
  plt.xlim([startwinter, endwinter])
  plt.savefig(os.path.join('Figures','CoreWinter', 'melt_cc_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_water_year.png'),dpi=300)
 
 
  plt.figure(2)
  plt.plot(days,np.asarray(r_val),'k')
  plt.plot(days,np.add(r_val,2*stdr_val),'c')
  plt.plot(days,np.subtract(r_val,2*stdr_val),'c')
  plt.plot([endspring, endspring],[-7.2,7.2],'k--',alpha=0.5)
  plt.plot([startwinter, startwinter],[-7.2,7.2],'k--',alpha=0.5)
  plt.plot([endwinter, endwinter],[-7.2,7.2],'k--',alpha=0.5)
  plt.grid()
  plt.plot([startday,endday],[0,0],'k',alpha=0.5)
  
  plt.title(f'Elevation = {bottom_ele} to {top_ele} m \n Mean R value for fit = {np.nanmean(r_val)}')
  plt.ylabel('R value of fit')
  plt.xlabel('Date')
  plt.ylim([-1,1])
  plt.xlim([0,365])
  plt.xticks(days[::60], dates[::60])
  plt.grid()
  plt.savefig(os.path.join('Figures','RvalueElevationBinnedCoeffs', 'cc_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '_rvalue.png'),dpi=300)
  
  
  calc_seasonal = True
  if calc_seasonal == True:
    plt.figure(3)
    plt.title('CC')
    plt.ylabel('CC (mm WE)')
    plt.xlabel('Date')
    plt.plot([startday, startday],[-21,0],'k:',alpha=0.5)
    plt.plot([endday, endday],[-21,0],'k:',alpha=0.5)
    plt.ylim([0, -21])
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
      aa = ds['coldcontentsnow']/0.334
      df = pd.read_csv('dem.csv',names=range(0,117))
      aa = np.asarray(aa)
      df = np.asarray(df)
      aa[df<bottom_ele]= np.nan
      aa[df>=top_ele] = np.nan
      bb = np.nanmean(np.nanmean(aa,axis=0),axis=0)
      if yr == 1951:
        plt.plot(dowy,bb,'gray',alpha=0.2,label='CC')
      else: 
        plt.plot(dowy,bb,'gray',alpha=0.2) 
      fall_mean.append(np.nanmean(bb[400:952]))
      winter_mean.append(np.nanmean(bb[953:1280]))
      spring_mean.append(np.nanmean(bb[1281:1600]))




    plt.text(85,-18,'Early\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(140,-18,'Core\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(180,-18,'Late\nAccum.',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.text(225,-18,'Melt\nOnset',fontsize=10,bbox=dict(facecolor='white', alpha=0.75, linewidth=0),horizontalalignment='center')
    plt.plot([endspring, endspring],[-21,0],'k--',alpha=0.5)
    plt.plot([startwinter, startwinter],[-21,0],'k--',alpha=0.5)
    plt.plot([endwinter, endwinter],[-21,0],'k--',alpha=0.5)
    plt.plot([startday, startday],[-21,0],'k--',alpha=0.5)
    plt.plot([endday, endday],[-21,0],'k--',alpha=0.5)  
    plt.grid()
    plt.xlim([0,300])
    
    plt.plot(np.asarray(mean_cc)/0.334,'k', linewidth=2,label='Mean CC')
    plt.savefig(os.path.join('Figures','ElevationBinnedColdContent', 'cc_0_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
  
  
  
    years=range(1951,2023,1)
  
    ### Plot Fall
    plt.figure(7)
    plt.bar(years,fall_mean, color='red')
    plt.title('Mean Early Accumulation CC\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean CC (mm WE)')
    plt.ylim([-17,0])
    plt.grid()
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Fall_CC_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
  
  
    ### Plot Winter
    plt.figure(9)
    plt.bar(years,winter_mean, color='blue')
    plt.title('Mean Core Accumulation CC\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean CC (mm WE)')
    plt.ylim([-17,0])
    plt.grid()
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Winter_CC_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)
    
    ### Plot Spring
    plt.figure(11)
    plt.bar(years,spring_mean, color='green')
    plt.title('Mean Late Accumulation CC\nElevation = '+str(bottom_ele)+' to '+str(top_ele)+' m')
    plt.xlabel('Year')
    plt.ylabel('Mean CC (mm WE)')
    plt.ylim([-17,0])
    plt.grid()
    
    
    plt.savefig(os.path.join('Figures','DaysAtZeroREH', 'Mean_Spring_CC_from_' + str(bottom_ele) + '_to_' + str(top_ele) + '.png'),dpi=300)   

if __name__ == "__main__":
    bottom_ele = sys.argv[1]
    top_ele = sys.argv[2]
    water_year_coef_plot(int(top_ele),int(bottom_ele))
