#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:34:43 2023

This is a series of functions that create *.smet files for input to SNOWPACK or Alpine3D
    NOTE THAT PATHS ARE HARD CODED!!!! 

 

@author: Dr. Joel Brown

"""
def createMetio(fnam, site, fields='', start_date='2006-09-06', periods=8760,
                freq='H', diurnal=True, seasonal=True, proxy_temp_depth=0.8,
                max_temp=-4, units_offset='', formats='', t1_index=1151,
                t1_period=8760, precip='random', lower_boundary_temp=-14.16,
                temp_factor=1, *args, **kwargs):
    """
    createMetio
    
    Creates simple synthetic data
    

    Parameters
    ----------
    fnam : TYPE
        DESCRIPTION.
    site : TYPE
        DESCRIPTION.
    fields : TYPE, optional
        DESCRIPTION. The default is ''.
    start_date : TYPE, optional
        DESCRIPTION. The default is '2007-05-01'.
    periods : TYPE, optional
        DESCRIPTION. The default is 8760.
    freq : TYPE, optional
        DESCRIPTION. The default is 'H'.
    diurnal : TYPE, optional
        DESCRIPTION. The default is True.
    seasonal : TYPE, optional
        DESCRIPTION. The default is True.
    proxy_temp_depth : TYPE, optional
        DESCRIPTION. The default is 0.8.
    max_temp : TYPE, optional
        DESCRIPTION. The default is -4.
    units_offset : TYPE, optional
        DESCRIPTION. The default is ''.
    formats : TYPE, optional
        DESCRIPTION. The default is ''.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    import pandas as pd
    import numpy as np

    
    # Initialize dataframe
    df =  pd.DataFrame()
    if precip == 'random':
        # Make psudo random precip data
        precip = np.random.rand(periods)
    else:
        precip = precip
    
    # Make timestamp series data
    df.insert(0, 'timestamp', pd.date_range(start_date, periods=periods, freq=freq))
    
    
    if diurnal:
        # Load temperature data from firn temp profile
        tdf=pd.read_csv('2019-2020-T3-Temp-32m.csv', header=5)        
        # Resample temp data to match hourly met out
        tdf = tdf.iloc[::2, :].reset_index(drop=True)
        # Take single year of data from firn temp profile
        tdf = tdf.iloc[t1_index:t1_index+t1_period].reset_index(drop=True)
        # Repeat data to length of met data
        atemps = np.tile(tdf[str(proxy_temp_depth)], periods)[:periods]
        atemps = ((atemps - lower_boundary_temp)*temp_factor) + lower_boundary_temp + 1
        # Add temp data to dataframe
        df.insert(1, 'airtemp', atemps)
    else:
        # Add temp data to dataframe
        df.insert(1, 'airtemp', -11)
    # Add constant values for other fields in met data
    df.insert(2, 'relative_humidity', 0.61)
    df.insert(3, 'lower_boundary', lower_boundary_temp)
    df.insert(4, 'wind_velocity', 3)
    df.insert(5, 'wind_direction', 20)
    df.insert(6, 'incoming_short_wave_rad', 2)
    df.insert(7, 'incoming_long_wavee_rad', 40)
    df.insert(8, 'precip', (precip-0.9)*10)
    df.insert(0, 'time', '')
    # Set boundaries on temp and precip data
    df.airtemp = df.airtemp.mask(df.airtemp.gt(max_temp),max_temp)
    df.precip = df.precip.mask(df.precip.lt(0),0)
    # Reformat timestamp to MetIO standard format
    for ii in range(0,df.shape[0]):
        ts = str(df.timestamp[ii]).split()[0] + 'T' + str(df.timestamp[ii]).split()[1]
        df.loc[ii,'time'] = ts[0:19]
    
    # set header fields
    units_offset = "0 273.15 0 273.15 0 0 0 0 0" + units_offset
    fields = "timestamp TA RH TSG VW DW ISWR ILWR PSUM" + fields    
    head = ["SMET 1.1 ASCII",
    		"[HEADER]",
    		"station_id       = " + site,
    		"station_name     = " + site,
    		"latitude         = " + '69.7836',
    		"longitude        = " + '-47.670183',
    		"altitude         = " + '1819',
    		"nodata           = -999",
    		"tz               = 8",
    		"fields           = " + fields,
            "units_offset     = " + units_offset,
    		"[DATA]"]
    # Delete unformatted timestamp data
    del df['timestamp']
    # Define output format for each column
    formats = '%19s %8.2f %10.3f %8.2f %6.1f %5.0f %6.0f %6.0f %10.3f' + formats
    
    # Output *.smet file to correct folder
    np.savetxt('../examples/input/'+ site + '.smet', df, fmt=formats, header='\n'.join(head), comments='')
    
def merra2metio(fnam, site, start_date='19800101', end_date='20230801', firstday=0,
                site_lat = 69.7817, site_lon=-47.6684, site_ele=1779, 
                fields='', lower_boundary_temp=-14.16, *args, **kwargs):
    """
    Download MERRA-2 reanalysis data and output metIO file for use in snowpack model.
    NOTE THAT PATHS ARE HARD CODED!!!! 
    	
    Parameters
    ----------
    fnam : string
        Full name of *.nc file to download from GES DISC
    site : string
    Name of site - this is used as the output name for the *.smet file
    fields : string
        Non-required fields to be used in the snowpack model run
        required fields include:
        timestamp TA RH TSG VW DW ISWR ILWR PSUM
        optional fields include:
        OSWR OLWR TS1 TS2 TS3 TS4 TS5 TS6 TS7
        
    'usr_cor', 'ulr', 't_i_1', 't_i_2', 't_i_3', 't_i_4','t_i_5', 't_i_6', 't_i_7'
    
    %6.0f %6.0f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f
    
    0,0,273.15,273.15,273.15,273.15,273.15,273.15,273.15
    
    Returns
    -------
    None.
    
    """
    import netCDF4 as nc4
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import requests
    import time
    
    pointdf = pd.read_csv(fnam)	
    	
    start = datetime(int(start_date[0:4]),int(start_date[4:6]),int(start_date[6:8]), 0, 0)        
    end = datetime(int(end_date[0:4]),int(end_date[4:6]),int(end_date[6:8]), 0, 0)
    	
    def makedaterange(start, end):
        dt = timedelta(days=1)
        while start < end:
            yield start
            start += dt
            
    datetimelist=[]
    year = []
    month = []
    h=firstday
    
    
    for day in makedaterange(start, end):
        datetimelist.append(day.strftime("%Y%m%d"))
        year.append(day.strftime("%Y"))
        month.append(day.strftime("%m"))
    
    metio_timestamp = []
    starttime = datetime(int(start_date[0:4]),int(start_date[4:6]),int(start_date[6:8]), 0, 0)        
    endtime = datetime(int(end_date[0:4]),int(end_date[4:6]),int(end_date[6:8])+1, 0, 0)
    
    def maketimerange(starttime, endtime):
        dt =  timedelta(hours=1)
        while starttime < endtime:
            yield starttime
            starttime += dt
    
    for hour in maketimerange(starttime, endtime):
        metio_timestamp.append(hour.strftime("%Y-%m-%dT%H:%M:%S"))
    
    for ii in range(firstday, len(datetimelist)):
        print("Dayof run = " + str(h))
        h += 1
        if int(year[ii]) < 1992:
            svv = '100'
        elif int(year[ii]) < 2001:
            svv = '200'
        elif int(year[ii]) < 2011:
            svv = '300'
        elif int(year[ii]) == 2020 and int(month[ii]) == 9:
            svv ='401'
        elif int(year[ii]) == 2021 and int(month[ii]) >= 6 and int(month[ii]) < 10:
            svv = '401'
        elif int(year[ii]) >= 2011:
            svv = '400'
        try:
            print("Importing flux data from " + datetimelist[ii])
            url = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_flx_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=PRECTOTCORR%2CQSH%2CSPEED%2CTLML%2CVLML%2CULML&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
            # url = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4')
            datflx = requests.get(url).content
            data = nc4.Dataset('flx_data', memory=datflx)
            print("Importing radiation data from " + datetimelist[ii])
            url1 = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_rad_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=LWGNT%2CSWGNT&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
            # url1 = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4#mode=bytes')
            datrad = requests.get(url1).content
            raddata = nc4.Dataset('rad_data', memory=datrad)
            print('Done loading data for ' + datetimelist[ii])
        except OSError:
            print("OSError unknown file format... \n Usually means bad download... trying again.")
            connected = False
            count=0
            while not connected and count < 100:
                time.sleep(0.5)
                try:
                    print("Importing flux data from " + datetimelist[ii])
                    url = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_flx_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=PRECTOTCORR%2CQSH%2CSPEED%2CTLML%2CVLML%2CULML&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                    # url = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4')
                    datflx = requests.get(url).content
                    data = nc4.Dataset('flx_data', memory=datflx)
                    print("Importing radiation data from " + datetimelist[ii])
                    url1 = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_rad_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=LWGNT%2CSWGNT&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                    # url1 = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4#mode=bytes')
                    datrad = requests.get(url1).content
                    raddata = nc4.Dataset('rad_data', memory=datrad)
                    print('Success! Done loading data for ' + datetimelist[ii])
                    connected = True
                except OSError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
                except ConnectionError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
                except RuntimeError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
                    
        except ConnectionError:
            print("ConnectionError.. trying again.")
            connected = False
            count=0
            while not connected and count < 100:
                time.sleep(0.5)
                try:
                    print("Importing flux data from " + datetimelist[ii])
                    url = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_flx_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=PRECTOTCORR%2CQSH%2CSPEED%2CTLML%2CVLML%2CULML&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                    # url = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4')
                    datflx = requests.get(url).content
                    data = nc4.Dataset('flx_data', memory=datflx)
                    print("Importing radiation data from " + datetimelist[ii])
                    url1 = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_rad_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=LWGNT%2CSWGNT&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                    # url1 = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4#mode=bytes')
                    datrad = requests.get(url1).content
                    raddata = nc4.Dataset('rad_data', memory=datrad)
                    print('Success! Done loading data for ' + datetimelist[ii])
                    connected = True
                except OSError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
                except ConnectionError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
                except RuntimeError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
                    
        except RuntimeError:
            print("RuntimeError... trying again.")
            connected = False
            count=0
            while not connected and count < 100:
                time.sleep(0.5)
                try:
                    print("Importing flux data from " + datetimelist[ii])
                    url = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_flx_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=PRECTOTCORR%2CQSH%2CSPEED%2CTLML%2CVLML%2CULML&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                    # url = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4')
                    datflx = requests.get(url).content
                    data = nc4.Dataset('flx_data', memory=datflx)
                    print("Importing radiation data from " + datetimelist[ii])
                    url1 = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_rad_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=LWGNT%2CSWGNT&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                    # url1 = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4#mode=bytes')
                    datrad = requests.get(url1).content
                    raddata = nc4.Dataset('rad_data', memory=datrad)
                    print('Success! Done loading data for ' + datetimelist[ii])
                    connected = True
                except OSError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
                except ConnectionError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
                except RuntimeError:
                    count+=1
                    print("Trying again, attempt number: " + str(count))
        for jj in range(0, len(pointdf)):
            site_lat = pointdf.Lat[jj]
            site_lon = pointdf.Lon[jj]
            precip = data.variables["PRECTOTCORR"][:]
            wind_speed = data.variables["SPEED"][:]
            north_wind_component = data.variables["VLML"][:]
            east_wind_component = data.variables["ULML"][:]
            specific_humidity = data.variables["QSH"][:]
            in_short_rad = raddata.variables["SWGNT"][:]
            in_long_rad = raddata.variables["LWGNT"][:]
            air_temp = data.variables["TLML"][:]
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            ix = round((site_lon - lon[0])/float(data.LongitudeResolution))
            iy = round((site_lat - lat[0])/float(data.LatitudeResolution))
              
            if ii == 0:
                units_offset = "0 273.15 0 273.15 0 0 0 0 0"
                fields = "timestamp TA RH TSG VW DW ISWR ILWR PSUM" + fields
                wind_direction = np.degree(np.arctan2(north_wind_component,east_wind_component)-(np.pi/2))
                # Define header for *.smet file
                head = ["SMET 1.1 ASCII",
    		        "[HEADER]",
    		        "station_id       = " + site,
    		        "station_name     = " + site,
    		        "latitude         = " + str(site_lat),
    		        "longitude        = " + str(site_lon),
    		        "altitude         = " + str(site_ele),
    		        "nodata           = -999",
    		        "tz               = 8",
    		        "fields           = " + fields,
                    "units_offset     = " + units_offset,
    		        "[DATA]"]
                
                # Initialize dataframe
                df =  pd.DataFrame()
                saturation_point=6.11*(10**(7.5*(air_temp-273.15)/air_temp))
                rel_humidity=specific_humidity/saturation_point*1000
                print(np.max(rel_humidity))
                # Create Dataframe for output
                df.insert(0, 'timestamp', metio_timestamp[0:24])
                df.insert(1, 'airtemp', air_temp[0:24,iy,ix]-273.15)
                df.insert(2, 'relative_humidity', rel_humidity[0:24,iy,ix])
                df.insert(3, 'lower_boundary', lower_boundary_temp)
                df.insert(4, 'wind_velocity', wind_speed[0:24,iy,ix])
                df.insert(5, 'wind_direction', wind_direction[0:24,iy,ix])
                df.insert(6, 'incoming_short_wave_rad', in_short_rad[0:24,iy,ix])
                df.insert(7, 'incoming_long_wavee_rad', in_long_rad[0:24,iy,ix])
                df.insert(8, 'precip', precip[0:24,iy,ix]*60*60)
                formats = '%19s %8.2f %10.3f %8.2f %6.1f %5.0f %6.0f %6.0f %10.3f'
                
                # Output *.smet file to correct folder
                np.savetxt('../examples/input/smet_files/MERRA-2_data/'+ site + '.smet', df, fmt=formats, header='\n'.join(head), comments='')
                data.close()
                
            else:
                wind_direction = np.degree(np.arctan2(north_wind_component,east_wind_component)-(np.pi/2))
                # Initialize dataframe
                df =  pd.DataFrame()
                saturation_point=6.11*(10**(7.5*(air_temp-273.15)/air_temp))
                rel_humidity=specific_humidity/saturation_point*1000
                print(np.max(rel_humidity))
                hds = 24*(h-1)
                hde = (24*h)
                # Create Dataframe for output
                df.insert(0, 'timestamp', metio_timestamp[hds:hde])
                df.insert(1, 'airtemp', air_temp[0:24,iy,ix]-273.15)
                df.insert(2, 'relative_humidity', rel_humidity[0:24,iy,ix])
                df.insert(3, 'lower_boundary', lower_boundary_temp)
                df.insert(4, 'wind_velocity', wind_speed[0:24,iy,ix]) 
                df.insert(5, 'wind_direction', wind_direction[0:24,iy,ix])
                df.insert(6, 'incoming_short_wave_rad', in_short_rad[0:24,iy,ix])
                df.insert(7, 'incoming_long_wavee_rad', in_long_rad[0:24,iy,ix])
                df.insert(8, 'precip', precip[0:24,iy,ix]*60*60)
                
                formats = '%19s %8.2f %10.3f %8.2f %6.1f %5.0f %6.0f %6.0f %10.3f'
                with open('../examples/input/smet_files/MERRA-2_data/'+ site + '.smet','ab') as f:
                  # Output *.smet file to correct folder
                  f.write(b"")
                  np.savetxt(f, df, fmt=formats)
                  data.close()

def merra2csv(fnam, site, start_date='19800101', end_date='20230801', firstday=0,
                fields='', *args, **kwargs):
    """
    Download MERRA-2 reanalysis data and output *.csv file for multiple locations.
    Written for Clement to compare Merra2 to Snodas
    NOTE THAT PATHS ARE HARD CODED!!!! 
    
    
    Example usage:
        
        import grisMet2snowpackMetIO as gr2sno
        gr2sno.merra2csv('BasinPoints.csv','SForkFlathead',start_date='20040101',end_date='20230901')

    Parameters
    ----------
    fnam : STRING
        DESCRIPTION:
            Name of *.csv file with Lat and Lon locations. Format for file is
            three columns, comma delimited, column names should be "Site", "Lat",
            and "Lon"
    site : STRING
        DESCRIPTION:
            Site name. This is used in naming output file.
    start_date : STRING, optional
        DESCRIPTION. The default is '19800101'.
    end_date : STRING, optional
        DESCRIPTION. The default is '20230801'.
    firstday : INT, optional
        DESCRIPTION:
            Value used for restarting download and appending file with data.
            If firstday = 0, a new file will be written, if greater than 0, 
            file will be appended. The number corresponds to the number of days
            after the start_date. The default is 0.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Yields
    ------
    TYPE *.csv
        DESCRIPTION:
            Output file is comma delimited. Naming convention is <site>_lat_lon.csv
            where lat and lon are the closest MERRA-2 grid point to the input locations.
            Data will include:
                Timestamp, 
                Airtemp (K), 
                Relative Humidity (%), 
                Wind Speed (m/s), 
                Wind Speed - north component (m/s), 
                Wind Speed - east component (m/s), 
                Incoming Shortwave Radiation (W/m^2), 
                Incoming Longwave Radiation (W/m^2), 
                Precipitation (kg/m^2), 
                Snowfall (kg/m^2), 
                Cloud Cover (%)
                
            Timestamp is in UTC


    """

    import netCDF4 as nc4
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import requests
    import time
    
    pointdf = pd.read_csv(fnam)
    	
    start = datetime(int(start_date[0:4]),int(start_date[4:6]),int(start_date[6:8]), 0, 0)        
    end = datetime(int(end_date[0:4]),int(end_date[4:6]),int(end_date[6:8]), 0, 0)
    	
    def makedaterange(start, end):
        dt = timedelta(days=1)
        while start < end:
            yield start
            start += dt
            
    datetimelist=[]
    year = []
    month = []
    h=firstday
    
    
    for day in makedaterange(start, end):
        datetimelist.append(day.strftime("%Y%m%d"))
        year.append(day.strftime("%Y"))
        month.append(day.strftime("%m"))
    
    metio_timestamp = []
    starttime = datetime(int(start_date[0:4]),int(start_date[4:6]),int(start_date[6:8]), 0, 0)        
    endtime = datetime(int(end_date[0:4]),int(end_date[4:6]),int(end_date[6:8])+1, 0, 0)
    
    def maketimerange(starttime, endtime):
        dt =  timedelta(hours=1)
        while starttime < endtime:
            yield starttime
            starttime += dt
    
    for hour in maketimerange(starttime, endtime):
        metio_timestamp.append(hour.strftime("%Y-%m-%dT%H:%M:%S"))
    
    for ii in range(firstday, len(datetimelist)):
        print("Dayof run = " + str(h))
        h += 1
        if int(year[ii]) < 1992:
            svv = '100'
        elif int(year[ii]) < 2001:
            svv = '200'
        elif int(year[ii]) < 2011:
            svv = '300'
        elif int(year[ii]) == 2020 and int(month[ii]) == 9:
            svv ='401'
        elif int(year[ii]) == 2021 and int(month[ii]) >= 6 and int(month[ii]) < 10:
            svv = '401'
        elif int(year[ii]) >= 2011:
            svv = '400'
        try:
            print("Importing flux data from " + datetimelist[ii])
            url = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_flx_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=PRECTOTCORR%2CSPEED%2CTLML%2CPRECSNO%2CVLML%2CULML&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
            # url = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4')
            datflx = requests.get(url).content
            data = nc4.Dataset('flx_data', memory=datflx)
            print("Importing radiation data from " + datetimelist[ii])
            url1 = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_rad_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=LWGNT%2CSWGNT%2CCLDTOT&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
            # url1 = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4#mode=bytes')
            datrad = requests.get(url1).content
            raddata = nc4.Dataset('rad_data', memory=datrad)
            print('Done loading data for ' + datetimelist[ii])
        except OSError:
            print("OSError unknown file format... \n Usually means bad download... trying again.")
            connected = False
            count=0
            while not connected and count < 100:
                
               time.sleep(0.5)
               try:
                   print("Importing flux data from " + datetimelist[ii])
                   url = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_flx_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=PRECTOTCORR%2CSPEED%2CTLML%2CPRECSNO%2CVLML%2CULML&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                   # url = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4')
                   datflx = requests.get(url).content
                   data = nc4.Dataset('flx_data', memory=datflx)
                   print("Importing radiation data from " + datetimelist[ii])
                   url1 = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_rad_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=LWGNT%2CSWGNT%2CCLDTOT&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                   # url1 = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4#mode=bytes')
                   datrad = requests.get(url1).content
                   raddata = nc4.Dataset('rad_data', memory=datrad)
                   print('Success! Done loading data for ' + datetimelist[ii])
                   connected = True
               except OSError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
               except ConnectionError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
               except RuntimeError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
        except ConnectionError:
            print("ConnectionError.. trying again.")
            connected = False
            count=0
            while not connected and count < 100:
                
               time.sleep(0.5)
               try:
                   print("Importing flux data from " + datetimelist[ii])
                   url = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_flx_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=PRECTOTCORR%2CSPEED%2CTLML%2CPRECSNO%2CVLML%2CULML&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                   # url = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4')
                   datflx = requests.get(url).content
                   data = nc4.Dataset('flx_data', memory=datflx)
                   print("Importing radiation data from " + datetimelist[ii])
                   url1 = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_rad_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=LWGNT%2CSWGNT%2CCLDTOT&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                   # url1 = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4#mode=bytes')
                   datrad = requests.get(url1).content
                   raddata = nc4.Dataset('rad_data', memory=datrad)
                   print('Success! Done loading data for ' + datetimelist[ii])
                   connected = True
               except OSError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
               except ConnectionError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
               except RuntimeError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
        except RuntimeError:
            print("RuntimeError... trying again.")
            connected = False
            count=0
            while not connected and count < 100:
                
               time.sleep(0.5)
               try:
                   print("Importing flux data from " + datetimelist[ii])
                   url = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXFLX.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_flx_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=PRECTOTCORR%2CSPEED%2CTLML%2CPRECSNO%2CVLML%2CULML&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                   # url = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_flx_Nx.' + datetimelist[ii] + '.nc4')
                   datflx = requests.get(url).content
                   data = nc4.Dataset('flx_data', memory=datflx)
                   print("Importing radiation data from " + datetimelist[ii])
                   url1 = ('https://goldsmr4.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?FILENAME=%2Fdata%2FMERRA2%2FM2T1NXRAD.5.12.4%2F'+ year[ii] + '%2F' + month[ii] + '%2FMERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4&LABEL=MERRA2_' + svv + '.tavg1_2d_rad_Nx.'+ datetimelist[ii] +'.SUB.nc&VARIABLES=LWGNT%2CSWGNT%2CCLDTOT&BBOX=-90%2C-180%2C90%2C180&VERSION=1.02&SERVICE=L34RS_MERRA2&DATASET_VERSION=5.12.4&FORMAT=bmM0Lw&SHORTNAME=M2T1NXFLX')
                   # url1 = ('https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2T1NXRAD.5.12.4/'+ year[ii] + '/' + month[ii] + '/MERRA2_' + svv + '.tavg1_2d_rad_Nx.' + datetimelist[ii] + '.nc4#mode=bytes')
                   datrad = requests.get(url1).content
                   raddata = nc4.Dataset('rad_data', memory=datrad)
                   print('Success! Done loading data for ' + datetimelist[ii])
                   connected = True
               except OSError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
               except ConnectionError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
               except RuntimeError:
                   count+=1
                   print("Trying again, attempt number:" + str(count))
                   
        for jj in range(0, len(pointdf)):
            site_lat = pointdf.Lat[jj]
            site_lon = pointdf.Lon[jj]
            precip = data.variables["PRECTOTCORR"][:]
            wind_speed = data.variables["SPEED"][:]
            in_short_rad = raddata.variables["SWGNT"][:]
            in_long_rad = raddata.variables["LWGNT"][:]
            cloud_cover = raddata.variables["CLDTOT"][:]
            air_temp = data.variables["TLML"][:]
            snowfall = data.variables["PRECSNO"][:]
            north_wind_component = data.variables["VLML"][:]
            east_wind_component = data.variables["ULML"][:]
            lat = data.variables['lat'][:]
            lon = data.variables['lon'][:]
            ix = round((site_lon - lon[0])/float(data.LongitudeResolution))
            iy = round((site_lat - lat[0])/float(data.LatitudeResolution))
            
            if ii == 0:

                
                # Define header for *.smet file
                head = ["Timestamp, Airtemp (K), Wind Speed (m/s), Wind Speed - north component (m/s), Wind Speed - east component (m/s), Incoming Shortwave Radiation (W/m^2), Incoming Longwave Radiation (W/m^2), Precipitation (kg/m^2), Snowfall (kg/m^2), Cloud Cover (%)"]
                
                # Initialize dataframe
                df =  pd.DataFrame()
                
                # Create Dataframe for output
                df.insert(0, 'timestamp', metio_timestamp[0:24])
                df.insert(1, 'airtemp', air_temp[0:24,iy,ix]-273.15)
                df.insert(2, 'wind_velocity', wind_speed[0:24,iy,ix])
                df.insert(3, 'north_wind_component', north_wind_component[0:24,iy,ix])
                df.insert(4, 'east_wind_component', east_wind_component[0:24,iy,ix])
                df.insert(5, 'incoming_short_wave_rad', in_short_rad[0:24,iy,ix])
                df.insert(6, 'incoming_long_wavee_rad', in_long_rad[0:24,iy,ix])
                df.insert(7, 'precip', precip[0:24,iy,ix]*60*60)
                df.insert(8, 'snowfall', snowfall[0:24,iy,ix]*60*60)
                df.insert(9, 'cloud_cover', cloud_cover[0:24,iy,ix])
                
                formats = '%s'
                
                
                
                # Output *.smet file to correct folder
                np.savetxt(site + '_' + str(data.variables['lat'][iy]).replace('.','_') + '_' + str(data.variables['lon'][ix]).replace('.','_') + '.txt', df, fmt=formats, delimiter=',', header='\n'.join(head), comments='')
                
                
            else:
                
                df =  pd.DataFrame()
                hds = 24*(h-1)
                
                hde = (24*h)
                # Create Dataframe for output
                df.insert(0, 'timestamp', metio_timestamp[hds:hde])
                df.insert(1, 'airtemp', air_temp[0:24,iy,ix]-273.15)
                df.insert(2, 'wind_velocity', wind_speed[0:24,iy,ix])
                df.insert(3, 'north_wind_component', north_wind_component[0:24,iy,ix])
                df.insert(4, 'east_wind_component', east_wind_component[0:24,iy,ix])
                df.insert(5, 'incoming_short_wave_rad', in_short_rad[0:24,iy,ix])
                df.insert(6, 'incoming_long_wavee_rad', in_long_rad[0:24,iy,ix])
                df.insert(7, 'precip', precip[0:24,iy,ix]*60*60)
                df.insert(8, 'snowfall', snowfall[0:24,iy,ix]*60*60)
                df.insert(9, 'cloud_cover', cloud_cover[0:24,iy,ix])
                
                formats = '%s'
                with open(site + '_' + str(data.variables['lat'][iy]).replace('.','_') + '_' + str(data.variables['lon'][ix]).replace('.','_') + '.txt','ab') as f:
                  # Output *.smet file to correct folder
                  f.write(b"")
                  np.savetxt(f, df, delimiter=',', fmt=formats)
                  
            print(str(pointdf.Site[jj]) + ' saved')
        data.close()

def era2metio(water_year, precip_factor=1, air_temp_offset=0):
    """
    Read ERA5-Land data (pre-downloaded with downloadERA5-land.py) and output .smet file to be loaded into SNOWPACK or Alpine3D.
    
    
    NOTE THAT PATHS ARE HARD CODED!!!! 
    
    Example usage:
        
        import grisMet2snowpackMetIO as gr2sno
        gr2sno.era2metio(water_year=1964, precip_factor=1, air_temp_offset=0)

    Parameters
    ----------
    water_year : INT
        DESCRIPTION:
            Water year to process
    precip_factor : INT, optional
        DESCRIPTION:
            Multiplier to be used with ERA5_land precip value
        The default is '1'.
    air_temp_offset : INT, optional
        DESCRIPTION: 
          Additive to be used with ERA5-land temp values
        The default is '0'.
    

    Yields
    ------
    *.smet file with correct formatting for input into SNOWPACK or Alpine3D


    """
    import numpy as np
    import xarray as xr
    import rioxarray as rio
    import pandas as pd
    water_year -= 1
    # Relative Humidity constants from Alduchov and Eskridge 1996
    alpha1 = 17.625
    beta1 = 243.04
    met_locations = pd.read_csv('BasinMetLocations.csv',delimiter=",", header=None)
    month = ['09', '10', '11', '12', '01', '02', '03', '04', '05', '06', '07', '08', '09']
    first_month = True
    for month in month:
        if month == '01':
            water_year += 1
        print("Month = " + month)
        dw = xr.open_dataset("../Download/ERA5_data/ERA5_" + str(water_year) + "_" + month + ".nc")
        dgp = xr.open_dataset("../Download/Geopotential/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc")
        sds = rio.open_rasterio("SFF_slope.tif")
        ads = rio.open_rasterio("SFF_aspect.tif")
        dds = rio.open_rasterio("SFF_DEM.tif")
        for loc in range(0,len(met_locations)):
            if first_month:
                print(met_locations[0][loc])
                print(met_locations[1][loc])
                #print(dds.sel(x=met_locations[1][loc], y=met_locations[0][loc], method="nearest").values)
                units_offset = "0 273.15 0 273.15 0 0 0 0 0"
                units_multiplier = "1 1 1 1 1 1 1 1 1"
                fields = "timestamp TA RH TSG VW DW ISWR ILWR PSUM"
               
                # Define header for *.smet file - ERA5-Land elevation is from geopotential grid point divided by gravitational acceleration constant
                head = ["SMET 1.1 ASCII",
                        "[HEADER]",
                        "station_id       = " + 'SSF_metio_' + str(met_locations[0][loc]).replace('.','_') + '_' + str(met_locations[1][loc]).replace('.','_'),
                        "station_name     = " + 'SSF_metio_' + str(met_locations[0][loc]).replace('.','_') + '_' + str(met_locations[1][loc]).replace('.','_'),
                        "latitude         = " + str(met_locations[0][loc]),
                        "longitude        = " + str(met_locations[1][loc]),
                        "altitude         = " + str(dgp.z.data[0,np.where(dgp.z.latitude==float(met_locations[0][loc]))[0][0],np.where(dgp.z.longitude==float(360+met_locations[1][loc]))[0][0]]/9.80665),
                        "nodata           = -999",
                        "tz               = -7",
                        "fields           = " + fields,
                        "units_offset     = " + units_offset,
                        "units_multiplier = " + units_multiplier,
                        "[DATA]"]
               
                try:
                    timeseries = dw.time.dt.strftime("%Y-%m-%dT%H:%M:%S")
                except AttributeError:
                    timeseries = dw.valid_time.dt.strftime("%Y-%m-%dT%H:%M:%S")
                air_temp = dw.t2m.where((dw.t2m.latitude==float(met_locations[0][loc])) & (dw.t2m.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.t2m.where((dw.t2m.latitude==float(met_locations[0][loc])) & (dw.t2m.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                dew_temp = dw.d2m.where((dw.d2m.latitude==float(met_locations[0][loc])) & (dw.d2m.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.d2m.where((dw.d2m.latitude==float(met_locations[0][loc])) & (dw.d2m.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                east_wind = dw.u10.where((dw.u10.latitude==float(met_locations[0][loc])) & (dw.u10.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.u10.where((dw.u10.latitude==float(met_locations[0][loc])) & (dw.u10.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                north_wind = dw.v10.where((dw.v10.latitude==float(met_locations[0][loc])) & (dw.v10.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.v10.where((dw.v10.latitude==float(met_locations[0][loc])) & (dw.v10.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                incoming_short_wave_rad = dw.ssrd.where((dw.ssrd.latitude==float(met_locations[0][loc])) & (dw.ssrd.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.ssrd.where((dw.ssrd.latitude==float(met_locations[0][loc])) & (dw.ssrd.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                incoming_long_wave_rad = dw.strd.where((dw.strd.latitude==float(met_locations[0][loc])) & (dw.strd.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.strd.where((dw.strd.latitude==float(met_locations[0][loc])) & (dw.strd.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                precip = dw.tp.where((dw.tp.latitude==float(met_locations[0][loc])) & (dw.tp.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.tp.where((dw.tp.latitude==float(met_locations[0][loc])) & (dw.tp.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                try:
                    lower_boundary = dw.stl1.where((dw.stl1.latitude==float(met_locations[0][loc])) & (dw.stl1.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.stl1.where((dw.stl1.latitude==float(met_locations[0][loc])) & (dw.stl1.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                except AttributeError:
                    dw1 = xr.open_dataset("../Download/ERA5_data/ERA5_" + str(water_year) + "_" + month + "_stl1.nc")
                    lower_boundary = dw1.stl1.where((dw1.stl1.latitude==float(met_locations[0][loc])) & (dw1.stl1.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw1.stl1.where((dw1.stl1.latitude==float(met_locations[0][loc])) & (dw1.stl1.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                wind_direction = np.degrees(np.arctan2(north_wind,east_wind)-(np.pi/2))
                wind_direction[wind_direction < 0] = wind_direction[wind_direction < 0]+360
               
                precip_start_of_day = precip
                precip = np.diff(precip)
                precip = np.insert(precip, 0, precip_start_of_day[0], axis=0)
                precip = precip * 1000 * precip_factor
                precip[precip<0] = precip_start_of_day[precip<0] * 1000
                precip[precip<0] = 0
               
                incoming_short_wave_rad_start_of_day = incoming_short_wave_rad
                #incoming_short_wave_rad = np.diff(incoming_short_wave_rad)
                #incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 0, incoming_short_wave_rad_start_of_day[0], axis=0)
                #incoming_short_wave_rad[incoming_short_wave_rad<0] = incoming_short_wave_rad_start_of_day[incoming_short_wave_rad<0]                
               
                incoming_long_wave_rad_start_of_day = incoming_long_wave_rad
                #incoming_long_wave_rad = np.diff(incoming_long_wave_rad)
                #incoming_long_wave_rad = np.insert(incoming_long_wave_rad, 0, incoming_long_wave_rad_start_of_day[0], axis=0)
                #incoming_long_wave_rad[incoming_long_wave_rad<0] = incoming_long_wave_rad_start_of_day[incoming_long_wave_rad<0]
               
                df = pd.DataFrame()
                df.insert(0, 'timestamp', timeseries)
                df.insert(1, 'airtemp', air_temp - 273.15 + air_temp_offset)
                df.insert(2, 'relative_humidity', ((np.exp((alpha1 * (dew_temp - 273.15))/(beta1 + (dew_temp- 273.15))))/(np.exp((alpha1 * (air_temp - 273.15))/(beta1 + (air_temp - 273.15))))))
                df.insert(3, 'lower_boundary', (lower_boundary - 273.15)*0)
                df.insert(4, 'wind_velocity', np.sqrt((north_wind ** 2)+(east_wind ** 2)))
                df.insert(5, 'wind_direction', wind_direction)
                #df.insert(6, 'incoming_short_wave_rad', incoming_short_wave_rad/3600/24)
                #df.insert(7, 'incoming_long_wave_rad', incoming_long_wave_rad/3600/24)
                df.insert(6, 'incoming_short_wave_rad', incoming_short_wave_rad/3600)
                df.insert(7, 'incoming_long_wave_rad', incoming_long_wave_rad/3600)
                df.insert(8, 'precip', precip)
               
                formats = '%19s %8.2f %10.3f %8.2f %6.1f %5.0f %6.0f %6.0f %10.6f'
               
                np.savetxt('../examples/input/smet_files/ERA5_data/' + 'SSF_metio_' + str(met_locations[0][loc]).replace('.','_') + '_' + str(met_locations[1][loc]).replace('.','_') + '.smet',  df, fmt=formats, header='\n'.join(head), comments='')
               
                fields_sno = "timestamp Layer_Thick  T  Vol_Frac_I  Vol_Frac_W  Vol_Frac_V  Vol_Frac_S Rho_S Conduc_S HeatCapac_S  rg  rb  dd  sp  mk mass_hoar ne CDot metamo"
                units_offset_sno = "0 0 273.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
                # Create Header for *.sno file
                head_sno = ["SMET 1.1 ASCII",
                    "[HEADER]",
                    "station_id       = " + 'SSF_metio_' + str(met_locations[0][loc]).replace('.','_') + '_' + str(met_locations[1][loc]).replace('.','_'),
                    "station_name     = " + 'SSF_metio_' + str(met_locations[0][loc]).replace('.','_') + '_' + str(met_locations[1][loc]).replace('.','_'),
                    "latitude         = " + str(met_locations[0][loc]),
                    "longitude        = " + str(met_locations[1][loc]),
                    "altitude         = " + str(dgp.z.data[0,np.where(dgp.z.latitude==float(met_locations[0][loc]))[0][0],np.where(dgp.z.longitude==float(360+met_locations[1][loc]))[0][0]]/9.80665),
                    "nodata           = -999",
                    "tz               = -7",
                    "source           = University of Montana; JBrown, 2023",
                    "ProfileDate      = 2023-01-01T00:00",
                    "HS_Last          = 0.0000",
                    "SlopeAngle       = 0",
                    "SlopeAzi         = 0",
                    "nSoilLayerData   = 6",
                    "nSnowLayerData   = 0",
                    "SoilAlbedo       = 0.09",
                    "BareSoil_z0      = 0.20",
                    "CanopyHeight     = 0.00",
                    "CanopyLeafAreaIndex = 0.000000",
                    "CanopyDirectThroughfall = 1.00",
                    "WindScalingFactor = 1.00",
                    "ErosionLevel     = 0",
                    "TimeCountDeltaHS = 0.",
                    "fields           = " + fields_sno,
                    "units_offset     = " + units_offset_sno,
                    "[DATA]",
                    str(water_year) + "-09-30T00:00:00 2.000 273.515 0.000 0.005 0.000 0.905 2600.0 2.6 1100.0 10000 0.00 0.0 0.0 0 0.0 1 0 0",
                    str(water_year) + "-09-30T00:00:00 1.000 273.515 0.000 0.010 0.000 0.990 2600.0 2.6 1100.0 10000 0.00 0.0 0.0 0 0.0 1 0 0",
                    str(water_year) + "-09-30T00:00:00 0.500 273.515 0.000 0.030 0.000 0.970 2600.0 2.6 1100.0 10000 0.00 0.0 0.0 0 0.0 1 0 0",
                    str(water_year) + "-09-30T00:00:00 0.500 273.515 0.000 0.050 0.000 0.950 2600.0 2.6 1100.0 10000 0.00 0.0 0.0 0 0.0 1 0 0",
                    str(water_year) + "-09-30T00:00:00 0.050 273.515 0.000 0.050 0.000 0.950 2600.0 2.6 1100.0 10000 0.00 0.0 0.0 0 0.0 1 0 0",
                    str(water_year) + "-09-30T00:00:00 0.050 273.515 0.000 0.050 0.000 0.950 2600.0 2.6 1100.0 10000 0.00 0.0 0.0 0 0.0 1 0 0"
                    ]
               
                df=pd.DataFrame()
                formats_sno = '%s'
                np.savetxt('../examples/input/sno_files/ERA5_data/' + 'SSF_metio_' + str(met_locations[0][loc]).replace('.','_') + '_' + str(met_locations[1][loc]).replace('.','_') + '.sno', df, fmt=formats_sno, header='\n'.join(head_sno), comments='')
               
            else:
                print(met_locations[0][loc])
                print(met_locations[1][loc])
               
               
                try:
                    timeseries = dw.time.dt.strftime("%Y-%m-%dT%H:%M:%S")
                except AttributeError:
                    timeseries = dw.valid_time.dt.strftime("%Y-%m-%dT%H:%M:%S")
                air_temp = dw.t2m.where((dw.t2m.latitude==float(met_locations[0][loc])) & (dw.t2m.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.t2m.where((dw.t2m.latitude==float(met_locations[0][loc]) )& (dw.t2m.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                dew_temp = dw.d2m.where((dw.d2m.latitude==float(met_locations[0][loc])) & (dw.d2m.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.d2m.where((dw.d2m.latitude==float(met_locations[0][loc])) & (dw.d2m.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                east_wind = dw.u10.where((dw.u10.latitude==float(met_locations[0][loc])) & (dw.u10.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.u10.where((dw.u10.latitude==float(met_locations[0][loc])) & (dw.u10.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                north_wind = dw.v10.where((dw.v10.latitude==float(met_locations[0][loc])) & (dw.v10.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.v10.where((dw.v10.latitude==float(met_locations[0][loc])) & (dw.v10.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                incoming_short_wave_rad = dw.ssrd.where((dw.ssrd.latitude==float(met_locations[0][loc])) & (dw.ssrd.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.ssrd.where((dw.ssrd.latitude==float(met_locations[0][loc])) & (dw.ssrd.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                incoming_long_wave_rad = dw.strd.where((dw.strd.latitude==float(met_locations[0][loc])) & (dw.strd.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.strd.where((dw.strd.latitude==float(met_locations[0][loc])) & (dw.strd.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                precip = dw.tp.where((dw.tp.latitude==float(met_locations[0][loc])) & (dw.tp.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.tp.where((dw.tp.latitude==float(met_locations[0][loc])) & (dw.tp.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                try:
                    lower_boundary = dw.stl1.where((dw.stl1.latitude==float(met_locations[0][loc])) & (dw.stl1.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw.stl1.where((dw.stl1.latitude==float(met_locations[0][loc])) & (dw.stl1.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                except AttributeError:
                    dw1 = xr.open_dataset("../Download/ERA5_data/ERA5_" + str(water_year) + "_" + month + "_stl1.nc")
                    lower_boundary = dw1.stl1.where((dw1.stl1.latitude==float(met_locations[0][loc])) & (dw1.stl1.longitude == float(met_locations[1][loc]))).stack(stacked=[...]).values[~np.isnan(dw1.stl1.where((dw1.stl1.latitude==float(met_locations[0][loc])) & (dw1.stl1.longitude == float(met_locations[1][loc])))).stack(stacked=[...])]
                wind_direction = np.degrees(np.arctan2(north_wind,east_wind)-(np.pi/2))
                wind_direction[wind_direction < 0] = wind_direction[wind_direction < 0]+360
               
                precip_start_of_day = precip
                precip = np.diff(precip)
                precip = np.insert(precip, 0, precip_start_of_day[1], axis=0)
                precip = precip * 1000 * precip_factor
                precip[precip<0] = precip_start_of_day[precip<0] * 1000
                precip[precip<0] = 0
               
                incoming_short_wave_rad_start_of_day = incoming_short_wave_rad
                # These next lines fill in missing ISWR ERA5-Land values
                if water_year == 1956 and met_locations[0][loc] == 47.4 and  met_locations[1][loc] == -113.3 and month == '10':
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 704, incoming_short_wave_rad_start_of_day[704], axis=0)
                if water_year == 1963 and met_locations[0][loc] == 47.4 and  met_locations[1][loc] == -113.0 and month == '10':
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 704, incoming_short_wave_rad_start_of_day[704], axis=0)
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 704, incoming_short_wave_rad_start_of_day[704], axis=0)
                if water_year == 1967 and met_locations[0][loc] == 47.7 and  met_locations[1][loc] == -113.1 and month == '01':
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 704, incoming_short_wave_rad_start_of_day[704], axis=0)
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 704, incoming_short_wave_rad_start_of_day[704], axis=0)
                if water_year == 1985 and met_locations[0][loc] == 48.1 and  met_locations[1][loc] == -113.8 and month == '10':
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 583, incoming_short_wave_rad_start_of_day[583], axis=0)
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 583, incoming_short_wave_rad_start_of_day[583], axis=0)
                if water_year == 1995 and met_locations[0][loc] == 47.4 and  met_locations[1][loc] == -113.3 and month == '01':
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 610, incoming_short_wave_rad_start_of_day[610], axis=0)
                if water_year == 2003 and met_locations[0][loc] == 48.3 and  met_locations[1][loc] == -113.8 and month == '01':
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 704, incoming_short_wave_rad_start_of_day[704], axis=0)
                  incoming_short_wave_rad = np.insert(incoming_short_wave_rad, 704, incoming_short_wave_rad_start_of_day[704], axis=0)
                #incoming_short_wave_rad[incoming_short_wave_rad<0] = incoming_short_wave_rad_start_of_day[incoming_short_wave_rad<0]                
               
                incoming_long_wave_rad_start_of_day = incoming_long_wave_rad
                #incoming_long_wave_rad = np.diff(incoming_long_wave_rad)
                #incoming_long_wave_rad = np.insert(incoming_long_wave_rad, 0, incoming_long_wave_rad_start_of_day[1], axis=0)
                #incoming_long_wave_rad[incoming_long_wave_rad<0] = incoming_long_wave_rad_start_of_day[incoming_long_wave_rad<0]
               
                df = pd.DataFrame()
                df.insert(0, 'timestamp', timeseries)
                df.insert(1, 'airtemp', air_temp - 273.15 + air_temp_offset)
                df.insert(2, 'relative_humidity', ((np.exp((alpha1 * (dew_temp - 273.15))/(beta1 + (dew_temp- 273.15))))/(np.exp((alpha1 * (air_temp - 273.15))/(beta1 + (air_temp - 273.15))))))
                df.insert(3, 'lower_boundary', (lower_boundary - 273.15)*0)
                df.insert(4, 'wind_velocity', np.sqrt((north_wind ** 2)+(east_wind ** 2)))
                df.insert(5, 'wind_direction', wind_direction)
                #df.insert(6, 'incoming_short_wave_rad', incoming_short_wave_rad/3600/24)
                #df.insert(7, 'incoming_long_wave_rad', incoming_long_wave_rad/3600/24)
                df.insert(6, 'incoming_short_wave_rad', incoming_short_wave_rad/3600)
                df.insert(7, 'incoming_long_wave_rad', incoming_long_wave_rad/3600)
                df.insert(8, 'precip', precip)
               
                formats = '%19s %8.2f %10.3f %8.2f %6.1f %5.0f %6.0f %6.0f %10.6f'
               
                with open('../input/smet_files/ERA5_data/' + 'SSF_metio_' + str(met_locations[0][loc]).replace('.','_') + '_' + str(met_locations[1][loc]).replace('.','_') + '.smet','ab') as f:
                  # Output *.smet file to correct folder
                  f.write(b"")
                  np.savetxt(f, df, fmt=formats)

        first_month = False          
        
    
def promice2metio(fnam, site, fields='', *args, **kwargs):
    """
    Read *.csv automatic weather station data file and output metIO file for 
    use in snowpack model.
    NOTE THAT PATHS ARE HARD CODED!!!! 

    Parameters
    ----------
    fnam : string
        Full name of *.csv file downloaded from promice AWS network
    site : string
        Name of site - this is used as the output name for the *.smet file
    fields : string
        Non-required fields to be used in the snowpack model run
        required fields include:
        timestamp TA RH TSG VW DW ISWR ILWR PSUM
        optional fields include:
        OSWR OLWR TS1 TS2 TS3 TS4 TS5 TS6 TS7

    'usr_cor', 'ulr', 't_i_1', 't_i_2', 't_i_3', 't_i_4','t_i_5', 't_i_6', 't_i_7'
    
    %6.0f %6.0f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f
    
    0,0,273.15,273.15,273.15,273.15,273.15,273.15,273.15

    Returns
    -------
    None.

    """
    
    import pandas as pd
    import numpy as np

    # Load pypromice v1.2.1 data file
    df = pd.read_csv(fnam)
    
    units_offset = "0 273.15 0 273.15 0 0 0 0 0"
    
    fields = "timestamp TA RH TSG VW DW ISWR ILWR PSUM" + fields

    # Reformat timestamp to MetIO standard format
    for ii in range(0,df.shape[0]):
        ts = df.time[ii].split()[0] + 'T' + df.time[ii].split()[1]
        df.at[ii,'time'] = ts

    # Define header for *.smet file
    head = ["SMET 1.1 ASCII",
		"[HEADER]",
		"station_id       = " + site,
		"station_name     = " + site,
		"latitude         = " + str(df['gps_lat'][0]),
		"longitude        = " + str(df['gps_lon'][0]),
		"altitude         = " + str(df.gps_alt[0]),
		"nodata           = -999",
		"tz               = 8",
		"fields           = " + fields,
        "units_offset     = " + units_offset,
		"[DATA]"]

    # Create Dataframe for output
    dat = df[['time', 't_l', 'rh_l_cor', 't_lower_bc', 'wspd_l', 'wdir_l', 'dsr_cor',
		   'dlr', 'precip_l_cor']]

    # Convert temps from deg C to deg K
    dat = dat.add(['', 3,0,0,0,0,0,0,0], axis='columns')

    # Convert precip from accumulated to hourly
    dat.precip_l_cor = dat.precip_l_cor.mask(df.precip_l_cor.lt(0),0)
    dat['precip_l_cor'] = dat['precip_l_cor'].diff()
    dat.t_lower_bc = -14.158
    dat.precip_l_cor = dat.precip_l_cor.interpolate(method='linear', axis=0).bfill()
    dat.t_l = df.t_l.interpolate(method='linear', axis=0)
    dat.rh_l_cor = df.rh_l_cor.interpolate(method='linear', axis=0)
    dat.wspd_l = df.wspd_l.interpolate(method='linear', axis=0)
    dat.wdir_l = df.wdir_l.interpolate(method='linear', axis=0)
    dat.dsr_cor = df.dsr_cor.interpolate(method='linear', axis=0)
    dat.dlr = df.dlr.interpolate(method='linear', axis=0)

    # Convert rel humidity to decimal percent
    dat['rh_l_cor'] = dat['rh_l_cor']/100
    
    
    
    
    
    dat.rh_l_cor = dat.rh_l_cor.mask(dat.rh_l_cor.lt(0.3),0.3)
    #dat.t_l = dat.t_l.mask(dat.t_l.eq(-999),-14)

    #dat = dat.fillna(-999)
    
    # Set limits on precip, wind speed, air temperature, and relative humidity
    dat.precip_l_cor = dat.precip_l_cor.mask(dat.precip_l_cor.lt(0),0)
    dat.precip_l_cor = dat.precip_l_cor.mask(dat.precip_l_cor.gt(100),0)
    dat.dlr = dat.dlr.mask(dat.dlr.gt(180),180) 
    dat.dsr_cor = dat.dsr_cor.mask(dat.dsr_cor.gt(80),80)
    dat.wspd_l = dat.wspd_l.mask(dat.wspd_l.lt(0),0)

    # Define output format for each column
    formats = '%19s %8.2f %10.3f %8.2f %6.1f %5.0f %6.0f %6.0f %10.3f'

    # Output *.smet file to correct folder
    np.savetxt('../examples/input/'+ site + '.smet', dat, fmt=formats, header='\n'.join(head), comments='')
    
def core2ice(fnam, tfnam, site, tempcol=100, *args, **kwargs):
    """
    Read *.csv file defining depth, temperature, and density from core data
    and output *.sno file for snowpack model
    NOTE THAT PATHS ARE HARD CODED!!!! 
    
    -this can be used for modeling firn when pre-existing core data is available for the model site.
    
    formatted to work with data from Harper group at the University of Montana

    Parameters
    ----------
    fnam : string
        file name for core density measurements.
    tfnam : string
        file name for temperature measurements at core depths
    site: string
        site name
    tempcol : INT
        number of layers
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(fnam)
    tdf = pd.read_csv(tfnam)
    
    # Define needed fields for header
    fields = "timestamp Layer_Thick  T  Vol_Frac_I  Vol_Frac_W  Vol_Frac_V  Vol_Frac_S Rho_S Conduc_S HeatCapac_S  rg  rb  dd  sp  mk mass_hoar ne CDot metamo"
    units_offset = "0 0 273.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    
    # Create Header for *.sno file
    head = ["SMET 1.1 ASCII",
		"[HEADER]",
		"station_id       = " + site,
		"station_name     = " + site,
		"latitude         = " + '69.872469',
		"longitude        = " + '-47.036429',
		"altitude         = " + '1948.5',
		"nodata           = -999",
		"tz               = 8",
        "source           = University of Montana; JBrown, 2023",
        "ProfileDate      = 2023-01-01T00:00", 
        "HS_Last          = 10.139567",
        "SlopeAngle       = 0.00",
        "SlopeAzi         = 0.00",
        "nSoilLayerData   = 0",
        "nSnowLayerData   = " + str(df.shape[0]),
        "SoilAlbedo       = 0.30",
        "BareSoil_z0      = 0.020",
        "CanopyHeight     = 0.00",
        "CanopyLeafAreaIndex = 0.000000",
        "CanopyDirectThroughfall = 1.00",
        "WindScalingFactor = 1.00",
        "ErosionLevel     = 181",
        "TimeCountDeltaHS = 0.",
		"fields           = " + fields,
        "units_offset     = " + units_offset,
		"[DATA]"
        ]

    ts=[]
    for ii in range(0,df.shape[0]):
        ts.append(str(2006 - ii) + "-01-01T00:00") 
    

    # Add timestamp to Dataframe
    df.insert(0, "timestamp", ts, True)
    
    # Add layer thickness to dataframe
    df.insert(1,"layer_thickness", (df.To-df.From)/100, True)
    
    # Create temperature column in Dataframe
    df.insert(2,'temp', np.nan)    
    
    # Assign temperatures to depth based on firn temp profiles
    for ii in tdf.Depth:
        idd = df[df.To.gt(ii*100)].index[0]
        idt = tdf[tdf.Depth == ii].index[0]
        idtx = tdf.columns[tempcol]
        df.loc[idd,'temp']=tdf.iloc[idt][idtx]
        
    # Interpolate temp data and extrapolate (linear) to ends of core
    df.temp = df.temp.interpolate(limit_direction='both')
    
    # Clean up Dataframe
    df = df.drop(['From', 'To', 'Mass'], axis=1)
    
    
    # Ice volume %
    df.insert(3, "Vol_Frac_I", df.Density/1000, True)
    # Water Volume %
    df.insert(4, "Vol_Frac_W", 0, True)
    # Voids volume %
    df.insert(5, "Vol_Frac_V", 1-df.Vol_Frac_I, True)
    # Soil volume percentage = 0
    df.insert(6, "Vol_Frac_S", 0, True)
    # Soil
    df.insert(8, 'conductivity', 0)
    # Soil
    df.insert(9, 'heat_capacity',0)
    # Grain Radius in mm
    df.insert(10, 'rg', 0.7)
    # Bond radius in mm
    df.insert(11, 'rb', 0.25)
    # Dendricity    0=old 1=new
    df.insert(12, 'dd', 0)
    # Sphericity    0=faceted; 1=rounded 
    # df.insert(13, 'sp', (df.Density/1000)-0.2)
    df.insert(13, 'sp', 1)

    # Microstructure marker 1 = faceted, 2 = rounded, 3 = Surface Hoar, 7 = glacier ice, 8 = ice layer, can be multiple numbers ie 28
    df.insert(14, 'mk', 2)
    # Mass of SURFACE hoar
    df.insert(15, 'mass_hoar', 0)
    # Number of elements (????)
    df.insert(16, 'ne', 1)
    # Stress change rate **initialize with 0**
    df.insert(17, 'CDot', 0)
    # NOT USED CURRENTLY
    df.insert(18,'metamo',0)
    
    # Limit volume fraction of ice to 100%
    df.Vol_Frac_I = df.Vol_Frac_I.mask(df.Vol_Frac_I.gt(1),1)
    
    # Limit volume fraction of void space to 0%
    df.Vol_Frac_V = df.Vol_Frac_V.mask(df.Vol_Frac_V.lt(0),0)
    
    # Flip dataframe for SNOPACK *.sno file output
    df = df.reindex(index=df.index[::-1])
    
    # Define output format for each column
    formats = '%19s %8.5f %9.4f %8.5f %8.5f %8.5f %8.5f %6.1f %6.3f %6.3f %6.2f %6.2f %6.2f %6.2f %6.0f %6.2f %6.0f %6.0f %6.0f'
    
    # Output *.smet file to correct folder
    np.savetxt('../examples/input/' + site + '.sno', df, fmt=formats, header='\n'.join(head), comments='')
        
def syntheticice(fnam, tfnam, site, tempcol=100,dt=0.2):
    """
    This is the same as above EXCEPT all of the data is fake. 
    NOTE THAT PATHS ARE HARD CODED!!!! 
    
    - can be used for initial conditions for firn modeling with SNOWPACK where data is non-existent or synthetic situation is preferable

    Parameters
    ----------
    fnam : string
        file name for core density measurements.
    tfnam : string
        file name for temperature measurements at core depths
    site: string
        site name
    tempcol : INT
        number of layers
    dt : FLOAT
        depth inrement
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    import firndens
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    rho_h, h, h_550 = firndens.herron_langway()
    dz_dt, time, depth, dep, dens = firndens.li_zwally(max_time=47, dt=dt)
    
    dat=[dz_dt, time[0:len(dz_dt)], depth, dep, dens]
    
    df = pd.DataFrame(data=np.transpose(dat) ,columns=['dz_dt', 'time', 'depth', 'dep', 'dens'])
    
    # df = pd.read_csv(fnam)
    tdf = pd.read_csv(tfnam)

    # Define needed fields for header
    fields = "timestamp Layer_Thick  T  Vol_Frac_I  Vol_Frac_W Vol_Frac_V  Vol_Frac_S Rho_S Conduc_S HeatCapac_S  rg  rb  dd  sp  mk mass_hoar ne CDot metamo"
    units_offset = "0 0 273.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

    # Create Header for *.sno file
    head = ["SMET 1.1 ASCII",
        "[HEADER]",
        "station_id       = " + site,
        "station_name     = " + site,
        "latitude         = " + '69.872469',
        "longitude        = " + '-47.036429',
        "altitude         = " + '1948.5',
        "nodata           = -999",
        "tz               = 8",
        "source           = University of Montana; JBrown, 2023",
        "ProfileDate      = 2023-01-01T00:00",
        "HS_Last          = 10.139567",
        "SlopeAngle       = 0.00",
        "SlopeAzi         = 0.00",
        "nSoilLayerData   = 0",
        "nSnowLayerData   = " + str(df.shape[0]),
        "SoilAlbedo       = 0.30",
        "BareSoil_z0      = 0.020",
        "CanopyHeight     = 0.00",
        "CanopyLeafAreaIndex = 0.000000",
        "CanopyDirectThroughfall = 1.00",
        "WindScalingFactor = 1.00",
        "ErosionLevel     = 181",
        "TimeCountDeltaHS = 0.",
        "fields           = " + fields,
        "units_offset     = " + units_offset,
        "[DATA]"
        ]
    year_end = 1979
    year_start = year_end-int(np.ceil(np.max(df.time)))
    start = datetime(year_start,1,1, 0, 0)        
    end = datetime(year_end,1,1, 0, 0)
    	
    def makedaterange(start, end):
        dtime = timedelta(days=365*dt)
        while start <= end:
            yield end
            end -= dtime
            
    metio_timestamp = []
    
    def maketimerange(starttime, endtime):
        dt =  timedelta(hours=1)
        while starttime < endtime:
            yield starttime
            starttime += dt
    
    for hour in makedaterange(start, end):
        metio_timestamp.append(hour.strftime("%Y-%m-%dT%H:%M"))
        
    # ts=[]
    # for ii in range(0,df.shape[0]):
    #     ts.append(str(1979 - ii) + "-01-01T00:00")


    # Add timestamp to Dataframe
    df.insert(0, "timestamp", metio_timestamp[0:len(df.depth)])

    # Add layer thickness to dataframe
    df.insert(1,"layer_thickness", (df.depth), True)

    # Create temperature column in Dataframe
    df.insert(2,'temp', np.nan)

    # Assign temperatures to depth based on firn temp profiles
    for ii in tdf.Depth:
        idd = df[df.dep.gt(ii)].index[0]
        idt = tdf[tdf.Depth == ii].index[0]
        # idt = depth[depth == ii].index[0]
        idtx = tdf.columns[tempcol]
        df.loc[idd,'temp']=tdf.iloc[idt][idtx]

    # Interpolate temp data and extrapolate (linear) to ends of core
    df.temp = df.temp.interpolate(limit_direction='both')


    # Ice volume %
    # Assign density data based on H&L model and interpolate and extrapolate to end of core
    df.insert(3, "Vol_Frac_I", np.nan)
    for ii in tdf.Depth:
        idd = df[df.dep.gt(ii)].index[0]
        idt = tdf[tdf.Depth == ii].index[0]
        df.loc[idd,"Vol_Frac_I"] = int(round(np.asarray(rho_h)[np.asarray(np.where(np.asarray(h) > tdf.Depth[idt]))[0][0]]))/1000
    df.Vol_Frac_I = df.Vol_Frac_I.interpolate(limit_direction='both')

    # Clean up Dataframe
    df = df.drop(['dz_dt', 'time', 'depth', 'dep', 'dens'], axis=1)

    # Water Volume %
    df.insert(4, "Vol_Frac_W", 0, True)
    # Voids volume %
    df.insert(5, "Vol_Frac_V", 1-df.Vol_Frac_I, True)
    # Soil volume percentage = 0
    df.insert(6, "Vol_Frac_S", 0, True)
    # Soil density
    df.insert(7, "Rho_S", 0, True)
    # Soil
    df.insert(8, 'conductivity', 0)
    # Soil
    df.insert(9, 'heat_capacity',0)
    # Grain Radius in mm
    df.insert(10, 'rg', 0.76)
    # Bond radius in mm
    df.insert(11, 'rb', 0.68)
    # Dendricity    0=old 1=new (for snow)
    df.insert(12, 'dd', 0)
    # Sphericity    0=faceted; 1=rounded
    # df.insert(13, 'sp', (df.Density/1000)-0.2)
    df.insert(13, 'sp', 0.5)

    # Microstructure marker 1 = faceted, 2 = rounded, 3 = Surface Hoar, 7 = glacier ice, 8 = ice layer, can be multiple numbers ie 28
    df.insert(14, 'mk', 1)
    # Mass of SURFACE hoar
    df.insert(15, 'mass_hoar', 0)
    # Number of elements (????)
    df.insert(16, 'ne', 1)
    # Stress change rate **initialize with 0**
    df.insert(17, 'CDot', 0)
    # NOT USED CURRENTLY
    df.insert(18,'metamo',0)

    # Limit volume fraction of ice to 100%
    df.Vol_Frac_I = df.Vol_Frac_I.mask(df.Vol_Frac_I.gt(1),1)

    # Limit volume fraction of void space to 0%
    df.Vol_Frac_V = df.Vol_Frac_V.mask(df.Vol_Frac_V.lt(0),0)

    # Flip dataframe for SNOPACK *.sno file output
    df = df.reindex(index=df.index[::-1])

    # Define output format for each column
    formats = '%19s %8.6f %9.4f %8.5f %8.5f %8.5f %8.5f %9.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f'

    # Output *.smet file to correct folder
    np.savetxt('../examples/input/' + site + '.sno', df, fmt=formats, header='\n'.join(head), comments='')
    
    print("Finished writing " + site + '.sno')

