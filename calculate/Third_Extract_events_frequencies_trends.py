from netCDF4 import Dataset
import numpy as np
import xarray as xr
import os
from scipy.stats import linregress

global_thresh=1

def trendtrend(matrix):
    trend=np.zeros((matrix.shape[1],matrix.shape[2]))
    intercept=np.zeros((matrix.shape[1],matrix.shape[2]))
    trend_d_mean=np.zeros((matrix.shape[1],matrix.shape[2]))
    pvalue=np.zeros((matrix.shape[1],matrix.shape[2]))
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[2]):
            if np.isnan(matrix[:,i,j]).all():
                trend[i,j]=np.nan
                pvalue[i,j]=np.nan
            else:
                data = matrix[:,i,j]
                #index and value of non-nan data
                x = np.arange(len(data))
                x = x[~np.isnan(data)]
                y = data[~np.isnan(data)]
                trend[i,j],intercept[i,j],_,pvalue[i,j],_=linregress(x,y)
                trend_d_mean[i,j]=trend[i,j]/np.nanmean(matrix[:,i,j])
    return trend,intercept,pvalue


def maskout(data,mask):
    #data is 3d, mask is 2d
    mask_useid=np.where(mask>0,1,np.nan)
    mask_useid=np.expand_dims(mask_useid,axis=0)
    mask_useid=np.repeat(mask_useid,data.shape[0],axis=0)
    data_maskout=data*mask_useid
    data_maskout[:,:,0:70]=np.nan
    return data_maskout

def mask2dout(data,mask):
    #data is 2d, mask is 2d
    mask_useid=np.where(mask>0,1,np.nan)
    data_maskout=data*mask_useid
    data_maskout[:,0:70]=np.nan
    return data_maskout

def cut_edge(data):
    #cut the lateral 7 grids
    #if the data is 3d, then cut the lateral 7 grids
    if len(data.shape)==3:
        data=data[:,7:-7,7:-7]
    #if the data is 2d, then cut the lateral 7 grids
    if len(data.shape)==2:
        data=data[7:-7,7:-7]
    return data



def get_yearly_frequency(matrix,histmatrix,threshold):
    # histmatrix_threshold=np.nanpercentile(histmatrix,threshold*100,axis=0)
    matrix_thresholded=np.where(matrix>1,1,0)
    matrix_reshape=np.reshape(matrix_thresholded,(45,12,matrix.shape[1],matrix.shape[2]))
    matrix_yearly=np.nansum(matrix_reshape,axis=1)/12
    return matrix_yearly


def get_yearly_intensity(matrix,histmatrix,threshold):
    # histmatrix_threshold=np.nanpercentile(histmatrix,threshold*100,axis=0)
    matrix_thresholded=np.where(matrix>1,matrix,np.nan)
    matrix_reshape=np.reshape(matrix_thresholded,(45,12,matrix.shape[1],matrix.shape[2]))
    matrix_yearly=np.nanmean(matrix_reshape,axis=1)
    return matrix_yearly





CNregionfil=Dataset("./newmask2.nc")
CNmaskraw=CNregionfil.variables["reg_mask"][:]
CNmaskraw=np.where(CNmaskraw==7,-1,CNmaskraw)
CNmaskraw=np.where(CNmaskraw==8,-1,CNmaskraw)
CNmaskraw=np.where(CNmaskraw==9,-1,CNmaskraw)
CNmask=cut_edge(CNmaskraw)


infil=Dataset("ALL_CHW_CHD_basehist_normalized_SPI_SPEI.nc")
os.system("rm -f checkcheck.nc")
outfil=Dataset("checkcheck.nc","w")
outfil.createDimension("time", 45)
outfil.createDimension("fiveyear", 3)
outfil.createDimension("lat", 171-14)
outfil.createDimension("lon", 231-14)
print(infil.variables.keys())

varnameuse = ['chd_chd','spi_dry','sti_hot','spei_dry','chdspi']
for vid,varname in enumerate(['chd','spi','sti','spei','chp']):
    for scernario in ['hist','ssp585','ssp245']:
        for model in ['ERA5','CESM','CESM_CWRF','MPI','MPI_CWRF']:
            if model == 'ERA5' and scernario != 'hist':
                continue
            var = model + '_' + scernario + '_' + varname
            varnameout = model + '_' + scernario + '_' + varnameuse[vid]
            print(var)
            matrix = np.array(infil.variables[var][:])
            matrix = cut_edge(matrix)
            matrix = maskout(matrix, CNmask)
            histmatrix = np.array(infil.variables[model + '_hist_' + varname][:])
            histmatrix = cut_edge(histmatrix)
            histmatrix = maskout(histmatrix, CNmask)
            matrix_yearly = get_yearly_frequency(matrix,histmatrix, global_thresh)
            outfil.createVariable(varnameout+"_freq", np.float32, ("time", "lat", "lon"))
            outfil.variables[varnameout+"_freq"][:] = matrix_yearly
            trend, intercept, pvalue = trendtrend(matrix_yearly)
            outfil.createVariable(varnameout+"_freq_trend", np.float32, ("lat", "lon"))
            outfil.createVariable(varnameout+"_freq_intercept", np.float32, ("lat", "lon"))
            outfil.createVariable(varnameout+"_freq_pvalue", np.float32, ("lat", "lon"))
            outfil.variables[varnameout+"_freq_trend"][:] = trend*10.0
            outfil.variables[varnameout+"_freq_intercept"][:] = intercept
            outfil.variables[varnameout+"_freq_pvalue"][:] = pvalue

            matrix_yearly = get_yearly_intensity(matrix,histmatrix, global_thresh)
            outfil.createVariable(varnameout+"_inte", np.float32, ("time", "lat", "lon"))
            outfil.variables[varnameout+"_inte"][:] = matrix_yearly
            trend, intercept, pvalue = trendtrend(matrix_yearly)
            outfil.createVariable(varnameout+"_inte_trend", np.float32, ("lat", "lon"))
            outfil.createVariable(varnameout+"_inte_intercept", np.float32, ("lat", "lon"))
            outfil.createVariable(varnameout+"_inte_pvalue", np.float32, ("lat", "lon"))
            outfil.variables[varnameout+"_inte_trend"][:] = trend*10.0
            outfil.variables[varnameout+"_inte_intercept"][:] = intercept
            outfil.variables[varnameout+"_inte_pvalue"][:] = pvalue

    var = "CESM_LWRF_hist_" + varname
    varnameout = "CESM_LWRF_hist_" + varnameuse[vid]
    matrix = np.array(infil.variables[var][:])
    matrix = cut_edge(matrix)
    matrix = maskout(matrix, CNmask)
    histmatrix = np.array(infil.variables["CESM_LWRF_hist_" + varname][:])
    histmatrix = cut_edge(histmatrix)
    histmatrix = maskout(histmatrix, CNmask)
    matrix_yearly = get_yearly_frequency(matrix,histmatrix, global_thresh)
    outfil.createVariable(varnameout+"_freq", np.float32, ("time", "lat", "lon"))
    outfil.variables[varnameout+"_freq"][:] = matrix_yearly
    trend, intercept, pvalue = trendtrend(matrix_yearly)
    outfil.createVariable(varnameout+"_freq_trend", np.float32, ("lat", "lon"))
    outfil.createVariable(varnameout+"_freq_intercept", np.float32, ("lat", "lon"))
    outfil.createVariable(varnameout+"_freq_pvalue", np.float32, ("lat", "lon"))
    outfil.variables[varnameout+"_freq_trend"][:] = trend*10.0
    outfil.variables[varnameout+"_freq_intercept"][:] = intercept
    outfil.variables[varnameout+"_freq_pvalue"][:] = pvalue
    matrix_yearly = get_yearly_intensity(matrix,histmatrix, global_thresh)
    outfil.createVariable(varnameout+"_inte", np.float32, ("time", "lat", "lon"))
    outfil.variables[varnameout+"_inte"][:] = matrix_yearly
    trend, intercept, pvalue = trendtrend(matrix_yearly)
    outfil.createVariable(varnameout+"_inte_trend", np.float32, ("lat", "lon"))
    outfil.createVariable(varnameout+"_inte_intercept", np.float32, ("lat", "lon"))
    outfil.createVariable(varnameout+"_inte_pvalue", np.float32, ("lat", "lon"))
    outfil.variables[varnameout+"_inte_trend"][:] = trend*10.0
    outfil.variables[varnameout+"_inte_intercept"][:] = intercept
    outfil.variables[varnameout+"_inte_pvalue"][:] = pvalue

outfil.close()
