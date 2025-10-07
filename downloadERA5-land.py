#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
downloadERA5-land.py
Created on Nov. 2024

Downloads ERA5-Land data with pertinent info for running Alpine3D opens zip file, gives unique name to each data set,

Script atapted from API request code @ https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download#area

@author: yeti
"""

import cdsapi
import os
import zipfile

c = cdsapi.Client()

year = ['1950', '1951', '1952', '1953', '1954','1955', '1956', '1957', '1958', '1959','1960', '1961', '1962', '1963', '1964','1965', '1966', '1967', '1968', '1969','1970', '1971', '1972', '1973', '1974','1975', '1976', '1977', '1978', '1979','1980', '1981', '1982', '1983', '1984','1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994','1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008','2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022','2023']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

for yr in year:
    for mo in month:
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                    '2m_temperature', 'skin_temperature', 'soil_temperature_level_1',
                    'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards', 'total_precipitation', 'snowfall',
                ],
                'year': yr,
                'month': mo,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    48.5, -114.1, 47.2,
                    -112.9,
                ],
                'format': 'netcdf.zip',
            },
            'download.netcdf.zip')
        with zipfile.ZipFile('download.netcdf.zip', 'r') as zip_ref:
            zip_ref.extractall('../Download/ERA5_data/')
        newname =  "ERA5_" + yr + "_" + mo + ".nc"
        os.rename("../Download/ERA5_data/data.nc", "../Download/ERA5_data/" + newname)
        os.remove('download.netcdf.zip')
