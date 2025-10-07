#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:56:32 2024

Concatenates full water year of ERA5-Land data into single file.

@author: yeti
"""
import os
import xarray as xr

for water_year in range(1951,2024):
    input_dir = '../Download/ERA5_data/'
    water_year -= 1
    files=[]
    months = ['10', '11', '12', '01', '02', '03', '04', '05', '06', '07', '08', '09']
    for month in months:
        if month == '01':
            water_year += 1
        files.append("../Download/ERA5_data/ERA5_" + str(water_year) + "_" + month + ".nc")
    
    datasets = [xr.open_dataset(file) for file in files]
    combined_dataset = xr.concat(datasets, dim='time')
    output_dir = '../Download/ERA5-Land_annual_water_year/'
    output_filename = 'combined_ERA5-Land_'+ str(water_year) + '.nc'
    output_path = os.path.join(output_dir, output_filename)
    combined_dataset.to_netcdf(output_path)
    for dataset in datasets:
        dataset.close()
    print("Combined NetCDF file saved successfully:", output_path)    
