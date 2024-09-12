from netCDF4 import Dataset
import numpy as np
import os

infil=Dataset("ALL_CHW_CHD_baseall_allSPEISPI.nc")
print(infil.variables.keys())

model_names = ['ERA5','CESM','MPI','CESM_CWRF','MPI_CWRF']
scernario_names = ['hist','ssp585','ssp245']
var_names = ['chd','spi','sti','spei','chp']

hist_std = {}
hist_mean = {}
normalied = {}
for scernario in scernario_names:
    for var in var_names:
        for model in model_names:
            if model == 'ERA5' and scernario != 'hist':
                continue
            varname = model + '_' + scernario + '_' + var
            var_mat = np.array(infil.variables[varname][:])
            if scernario == 'hist':
                var_std = np.std(var_mat, axis=0)
                var_mean = np.mean(var_mat, axis=0)
                var_normalized = (var_mat - var_mean) / var_std
                hist_std[varname] = var_std
                hist_mean[varname] = var_mean
            else:
                var_normalized = (var_mat - hist_mean[model + '_hist_' + var]) / hist_std[model + '_hist_' + var]
            normalied[varname] = var_normalized

for var in var_names:
    varname = 'CESM_LWRF_hist_' + var
    var_mat = np.array(infil.variables[varname][:])
    var_std = np.std(var_mat, axis=0)
    var_mean = np.mean(var_mat, axis=0)
    var_normalized = (var_mat - var_mean) / var_std
    normalied[varname] = var_normalized


os.system("rm -f ALL_CHW_CHD_basehist_normalized_SPI_SPEI.nc")
outfil = Dataset("ALL_CHW_CHD_basehist_normalized_SPI_SPEI.nc", "w")
outfil.createDimension("time", 540)
outfil.createDimension("lat", 171)
outfil.createDimension("lon", 231)
for varname in normalied.keys():
    outfil.createVariable(varname, np.float32, ("time", "lat", "lon"))
    outfil.variables[varname][:] = normalied[varname]
outfil.close()

