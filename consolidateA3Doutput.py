#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consolidateA3Doutput.py
Created on Tue Jan 16 09:56:58 2024

Loads Alpine3D *.smet output files (text), 
consolidates pertinant information for entire grid to single NetCDF file
for each water year.

@author: yeti
"""
import sys

def consolidateA3Doutput(water_year):
    import numpy as np
    import xarray as xr
    # import rioxarray as rio
    import pandas as pd
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import mean_squared_error
    # from math import sqrt
    import os
    from datetime import datetime
    # loop through locations
    
    
    water_year= int(water_year)
    print(water_year)
    
    def read_text_file(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('latitude         ='):
                    latitude = line
                    latitude = latitude[18:].strip().split()
                    #print(latitude)
                elif line.startswith('longitude        ='):
                    longitude = line
                    #print(line)
                    longitude = longitude[18:].strip().split()
                elif line.startswith('altitude         ='):
                    altitude = line
                    #print(line)
                    altitude = altitude[18:].strip().split()
                elif line.startswith('slope_angle      ='):
                    slope_angle = line
                    #print(line)
                    slope_angle = slope_angle[18:].strip().split()
                elif line.startswith('slope_azi        ='):
                    slope_azi = line
                    #print(line)
                    slope_azi = slope_azi[18:].strip().split()
                elif line.startswith('station_id       ='):
                    station_id = line
                    station_id = station_id[18:].strip().split()
                    x = int(station_id[0].split('_')[0])
                    y = int(station_id[0].split('_')[1])
                elif line.startswith('fields           ='):
                    header = line
                    #print(line)
                    var_names = header[18:].strip().split()
                elif line.startswith('[DATA]'):
                    break

        #var_names = ['timestamp', 'ColdContentSnow', 'HS_mod','SWE']
        df = pd.read_csv(file_path, delim_whitespace=True, header=17, names=var_names)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        #df['latitude'] = latitude[0]
        #df['longitude'] = longitude[0]
        #df['elevation'] = altitude[0]
        #df['slope_angle'] = slope_angle[0]
        #df['slope_azi'] = slope_azi[0]
        #df['station_id'] = station_id[0]
        #df.set_index(['latitude', 'longitude'])
        #['longitude'] + ['altitude'] + ['slope_angle'] + ['slope_azi'] + ['station_id']
        return df, x, y
    
    # iterate over files in dir
    count = 0
    alpine3Ddir = './meteo/WY'+ str(water_year)+ '/'
    
    for filename in os.listdir(alpine3Ddir):
      if filename.endswith('A3D.smet'):
        file_path = os.path.join(alpine3Ddir, filename)
        df, x, y = read_text_file(file_path)
        if count == 0:
            ccs = np.empty((196,117,len(df['ColdContentSnow'])))
            hs = np.empty((196,117,len(df['HS_mod'])))
            swe = np.empty((196,117,len(df['SWE'])))
            lwc = np.empty((196,117,len(df['MS_Water'])))
            evap = np.empty((196,117,len(df['MS_Evap'])))
            sub = np.empty((196,117,len(df['MS_Sublimation'])))
            runoff = np.empty((196,117,len(df['MS_SN_Runoff'])))
        
        ccs[196-y,x,:] = df['ColdContentSnow'] 
        hs[196-y,x,:] = df['HS_mod']
        swe[196-y,x,:] = df['SWE']
        lwc[196-y,x,:] = df['MS_Water']
        evap[196-y,x,:] = df['MS_Evap']
        sub[196-y,x,:] = df['MS_Sublimation']
        runoff[196-y,x,:] = df['MS_SN_Runoff']
        print(count)
        
        count+=1
    
    output_file = 'water_year_' + str(water_year) + '_Alpine3Doutput.nc'
    drunoff = xr.DataArray(runoff, dims=('lon', 'lat','timestamp'), coords=(np.arange(48.5, 47.2, ((47.2 - 48.5)/196)).tolist(), np.arange(-114.1 , -112.9, ((114.1 - 112.9)/117)).tolist(), df.timestamp))
    print(drunoff.max())
    dsub = xr.DataArray(sub, dims=('lon', 'lat','timestamp'), coords=(np.arange(48.5, 47.2, ((47.2 - 48.5)/196)).tolist(), np.arange(-114.1 , -112.9, ((114.1 - 112.9)/117)).tolist(), df.timestamp))
    print(dsub.max())
    devap = xr.DataArray(evap, dims=('lon', 'lat','timestamp'), coords=(np.arange(48.5, 47.2, ((47.2 - 48.5)/196)).tolist(), np.arange(-114.1 , -112.9, ((114.1 - 112.9)/117)).tolist(), df.timestamp))
    print(devap.max())
    dlwc = xr.DataArray(lwc, dims=('lon', 'lat','timestamp'), coords=(np.arange(48.5, 47.2, ((47.2 - 48.5)/196)).tolist(), np.arange(-114.1 , -112.9, ((114.1 - 112.9)/117)).tolist(), df.timestamp))
    print(dlwc.max())
    dswe = xr.DataArray(swe, dims=('lon', 'lat','timestamp'), coords=(np.arange(48.5, 47.2, ((47.2 - 48.5)/196)).tolist(), np.arange(-114.1 , -112.9, ((114.1 - 112.9)/117)).tolist(), df.timestamp))
    print(dswe.max())
    dhs = xr.DataArray(hs, dims=('lon', 'lat','timestamp'), coords=(np.arange(48.5, 47.2, ((47.2 - 48.5)/196)).tolist(), np.arange(-114.1 , -112.9, ((114.1 - 112.9)/117)).tolist(), df.timestamp))
    print(dhs.max())
    dccs = xr.DataArray(ccs, dims=('lon', 'lat','timestamp'), coords=(np.arange(48.5, 47.2, ((47.2 - 48.5)/196)).tolist(), np.arange(-114.1 , -112.9, ((114.1 - 112.9)/117)).tolist(), df.timestamp))
    print(dccs.min())
    ds = xr.Dataset({'swe': dswe, 'heightsnow': dhs, 'coldcontentsnow': dccs, 'liquidwatercontent': dlwc, 'evaporation': devap, 'sublimation': dsub, 'runoff': drunoff})
    ds.to_netcdf(output_file)
    print(ds)
    print(f"File saved as NetCDF: {output_file}")
  
if __name__ == "__main__":
  water_year = sys.argv[1]
  
  consolidateA3Doutput(water_year)


