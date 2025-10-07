#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept. 3 2025

Compares trend and error for fit to Alpine3D model results using modified bootstrap approach used in this study with Theil-Sen estimator

@author: yeti
"""
import os
percent_diff=[]
diff=[]
bootstrap=[]
sens=[]
mos=['10', '11', '12', '01', '02', '03', '04', '05', '06', '07']
for i in mos:
    for j in range(1,32):
        try:
            if j <10:
                fnam='swe_' + i + '_0' + str(j) + '.npy'
            else:
                fnam='swe_' + i + '_' + str(j) + '.npy'
            pathnam = os.path.join('SWE_NetCDF_ByDate', fnam)

            print(i)
            print(j)
            import numpy as np
            import pandas as pd

            from sklearn import linear_model 
            from sklearn.model_selection import train_test_split 

            c = np.load(pathnam)
            print('Resizing Series')
            print('\n')

            cc_norm = c.reshape((196,117,71))
            df = pd.read_csv('dem.csv',names=range(0,117))
            entire_basin_trend = []
            year_trend = []
            bottom_ele=900
            top_ele=2600
            for ii in range(0,71):
              aa = cc_norm[:,:,ii]
              aa[df<bottom_ele]= np.nan
              aa[df>=top_ele] = np.nan
              if np.nanmean(aa) >=0:
                entire_basin_trend.append(np.nanmean(aa))
                year_trend.append(ii+1951)
              else:
                entire_basin_trend.append('nan')
                year_trend.append('nan')
            entire_basin_trend = [x for x in entire_basin_trend if str(x) != 'nan']
            year_trend = [x for x in year_trend if str(x) != 'nan']


            import matplotlib.pyplot as plt
            from scipy.stats.mstats import theilslopes


            # Calculate Sen's slope
            slope, intercept, lower_bound, upper_bound = theilslopes(np.divide(entire_basin_trend,1), year_trend)

            # Print the results
            print("Sen's slope:", slope)
            print("Lower bound of confidence interval:", lower_bound)
            print("Upper bound of confidence interval:", upper_bound)

            NumIter=10000
            intercepts = []
            mse = []
            r_vals = []
            coefs = []
            z = []
            for k in range(0,int(NumIter)):  
              X_train, X_test, y_train, y_test = train_test_split(year_trend, entire_basin_trend,test_size=0.2, random_state=None)
              regr = linear_model.LinearRegression(fit_intercept=True)  
              regr.fit(np.asarray(X_train).reshape(-1, 1),y_train)  
              y_pred = regr.predict(np.asarray(X_test).reshape(-1, 1))
              z.append(np.polyfit(X_test,y_pred,1))
              coefs.append(regr.coef_)
              r_vals.append(regr.score(np.asarray(year_trend).reshape(-1, 1), entire_basin_trend))
              intercepts.append(regr.intercept_)

            # Print the results
            print('')
            print('Coefficients: ', np.mean(coefs,0))
            print('Lower Bound: ', np.subtract(np.mean(coefs,0),np.std(coefs,0)*2))
            print('Upper Bound: ', np.add(np.mean(coefs,0),np.std(coefs,0)*2))

            print('')
            print('Difference Coefficients: ', np.subtract(np.mean(coefs,0),slope))
            print('Difference Lower Bound: ', np.subtract(np.subtract(np.mean(coefs,0),np.std(coefs,0)*2),lower_bound))
            print('Difference Upper Bound: ', np.subtract(np.add(np.mean(coefs,0),np.std(coefs,0)*2),upper_bound))

            print('')
            print('Percent difference slope: ', (np.subtract(np.mean(coefs,0),slope)/np.add(np.mean(coefs,0),slope)/2)*100)
            print('Percent difference lower bound: ', (np.subtract(np.subtract(np.mean(coefs,0),np.std(coefs,0)*2),lower_bound)/(np.add(np.subtract(np.mean(coefs,0),np.std(coefs,0)*2),lower_bound)/2))*100)
            print('Percent difference upper bound: ', (np.subtract(np.add(np.mean(coefs,0),np.std(coefs,0)*2),upper_bound)/(np.add(np.add(np.mean(coefs,0),np.std(coefs,0)*2),upper_bound)/2))*100)
            percent_diff.append( (np.subtract(np.mean(coefs,0),slope)/np.add(np.mean(coefs,0),slope)/2)*100)
            diff.append(np.subtract(np.mean(coefs,0),slope))
            bootstrap.append(np.mean(coefs,0))
            sens.append(slope)
        except:
            pass

plt.figure(1)
plt.plot(percent_diff)
plt.title('Percent difference bootstrap - TS')
plt.ylabel('Percent Difference')
plt.xlabel('Day of water year')
plt.grid()

plt.figure(2)
plt.plot(diff)
plt.title('Absolute difference bootstrap - TS')
plt.ylabel('mm W.E.')
plt.xlabel('Day of water year')
plt.grid()

plt.figure(3)
plt.plot(bootstrap,'b',label='Bootstrap')
plt.plot(sens,'k--',label='Sens')
plt.ylabel('mm W.E.')
plt.xlabel('Day of water year')
plt.grid()
plt.legend()

plt.show()


