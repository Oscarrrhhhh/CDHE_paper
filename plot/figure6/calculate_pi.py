from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd

P0 = 101325  # Pa
L = -0.0065  # K/m
T0 = 288.15  # K
g = 9.80665  # m/s²
M = 0.0289644  #kg/mol
R = 8.314    #J/(mol·K)

infil=Dataset("/home/zhanghan/wrfinput_d01")
terrainheight=np.array(infil.variables['HGT'][:])
pa = P0 * (1 - (L * terrainheight) / T0) ** (g * M / (R * L)) / 1000.0
#repeat pa to 16515
pa = np.repeat(pa, 540, axis=0)


#calculate net radiation
sh_fil=xr.open_dataset("AHFX.nc")
time_strings = sh_fil["AHFX"].time.dt.strftime('%Y-%m-%d %H:%M:%S').values
converted_time = pd.to_datetime(time_strings)
sh=sh_fil["AHFX"][:,0,:,:]


lh_fil=xr.open_dataset("ALFX.nc")
time_strings = lh_fil["ALFX"].time.dt.strftime('%Y-%m-%d %H:%M:%S').values
converted_time = pd.to_datetime(time_strings)
lh=lh_fil["ALFX"][:,0,:,:]

t2m_fil=xr.open_dataset("AT2M.nc")
time_strings = t2m_fil["AT2M"].time.dt.strftime('%Y-%m-%d %H:%M:%S').values
converted_time = pd.to_datetime(time_strings)
t2m=t2m_fil["AT2M"][:,0,:,:]

pa=pa+t2m-t2m

Ta=t2m-273.15 #C
lamda=Ta*(-0.002361)+2.501 #MJ/kg
gamma=pa/lamda*0.0016286
es=np.exp(Ta/(Ta+237.3)*17.27)*0.618
delta=es*4098/(Ta+237.3)**2 #%des/dTa in kPa/oC
A=sh+lh#:*surface available energy, equals net radiation minus ground heat flux (
A=A*0.0864
A=np.array(A)/np.array(lamda)
a=1.26
PET=delta/(delta+gamma)*A*a
pet=PET
pet.to_netcdf("pet_MPI_CWRF.nc")
def calculate_daily_anomalies_std(data):
    # Calculate the daily climatology
    climatology = data.groupby("time.dayofyear").mean("time")
    # Calculate anomalies by subtracting the climatology from the original data
    anomalies = data.groupby("time.dayofyear") - climatology
    # Calculate the standard deviation of the anomalies then devide anomalies by the standard deviation
    std = anomalies.groupby("time.dayofyear").std("time")
    anomalies = anomalies.groupby("time.dayofyear") / std
    # Return the anomalies
    return anomalies

net_rad_anomalies = calculate_daily_anomalies_std(sh+lh)
sh_anomalies = calculate_daily_anomalies_std(sh)
lh_anomalies = calculate_daily_anomalies_std(lh)
pet_anomalies = calculate_daily_anomalies_std(pet)
t2m_anomalies = calculate_daily_anomalies_std(t2m)
print(t2m_anomalies)



term1=sh   #W/m2=J/s/m2
term1.to_netcdf("term1.nc")
term2=sh+lh-np.array(lamda)*np.array(pet)
term2.to_netcdf("term2.nc")
t2m_anomalies.to_netcdf("t2m_anomalies.nc")
term1_anomaly=calculate_daily_anomalies_std(term1)
term2_anomaly=calculate_daily_anomalies_std(term2)
pi=(term1_anomaly-term2_anomaly)*np.array(t2m_anomalies)
#write pi to nc
pi.to_netcdf("pi_MPI_CWRF.nc")
#close all files

