from multiprocessing import Pool
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

#Key import: index calculation function
from index_lib import ref_hist_obs_drought_index_SPISPEI



ERA5_AT2M_hist = np.array(Dataset('../fusion/Fused_OBS_AT2M_PRAVG.nc').variables['AT2M']).reshape(540,171,231)
ERA5_PRAVG_hist = np.array(Dataset('../fusion/Fused_OBS_AT2M_PRAVG.nc').variables['PRAVG']).reshape(540,171,231)
ERA5_AT2M_hist=np.where(ERA5_AT2M_hist>400,np.nan,ERA5_AT2M_hist)
ERA5_AT2M_hist=np.where(ERA5_AT2M_hist<200,np.nan,ERA5_AT2M_hist)
ERA5_PRAVG_hist=np.where(ERA5_PRAVG_hist<0,np.nan,ERA5_PRAVG_hist)
ERA5_PRAVG_hist=np.where(ERA5_PRAVG_hist>10000,np.nan,ERA5_PRAVG_hist)
ERA5_PET_hist = np.array(Dataset('OBS/ERA5_PET.nc').variables['t2m'])[0:540,:,:]
ERA5_PRPET_hist = ERA5_PRAVG_hist-ERA5_PET_hist

CESM_AT2M_hist = np.array(Dataset('/home/zhanghan/qTERA07/climate_chd/supdata/hist/tas_CESM_RCMgrid_HIST_sel.nc').variables['tas'])[0:540,:,:]
CESM_PRAVG_hist = np.array(Dataset('/home/zhanghan/qTERA07/climate_chd/supdata/hist/pr_CESM_RCMgrid_HIST_sel.nc').variables['pr'])[0:540,:,:]*86400
CESM_PET_hist = np.array(Dataset('/home/zhanghan/qTERA07/climate_chd/supdata/hist/pet_CESM_HIST.nc').variables['tas'])[0:540,:,:]
CESM_PRPET_hist = CESM_PRAVG_hist-CESM_PET_hist

CESM_AT2M_ssp585 = np.array(Dataset('CMIP/CESM2_s585_use.nc4').variables['tas_CESM2_s585_regrid_b'])[0:540,:,:]
CESM_PRAVG_ssp585 = np.array(Dataset('CMIP/CESM2_s585_use.nc4').variables['pr_CESM2_s585_regrid_b'])[0:540,:,:]*86400
CESM_PET_ssp585 = np.array(Dataset('CMIP/pet_CESM2_SSP585.nc').variables['tas'])[0:540,:,:]
CESM_PRPET_ssp585 = CESM_PRAVG_ssp585-CESM_PET_ssp585

CESM_AT2M_ssp245 = np.array(Dataset('CMIP/tas_CESM_RCMgrid_SSP245.nc').variables['tas'])[0:540,:,:]
CESM_PRAVG_ssp245 = np.array(Dataset('CMIP/pr_CESM_RCMgrid_SSP245.nc').variables['pr'])[0:540,:,:]*86400
CESM_PET_ssp245 = np.array(Dataset('CMIP/pet_CESM_SSP245.nc').variables['tas'])[0:540,:,:]
CESM_PRPET_ssp245 = CESM_PRAVG_ssp245-CESM_PET_ssp245

MPI_AT2M_hist = np.array(Dataset('CMIP/MPI_hist_use.nc4').variables['tas_MPI_hist_regrid_b'])[0:540,:,:]
MPI_PRAVG_hist = np.array(Dataset('CMIP/MPI_hist_use.nc4').variables['pr_MPI_hist_regrid_b'])[0:540,:,:]*86400
MPI_PET_hist = np.array(Dataset('CMIP/MPI_PET.nc').variables['tas'])[0:540,:,:]
MPI_PRPET_hist = MPI_PRAVG_hist-MPI_PET_hist

MPI_AT2M_ssp585 = np.array(Dataset('CMIP/MPI_s585_use.nc4').variables['tas_MPI_s585_regrid_b'])[0:540,:,:]
MPI_PRAVG_ssp585 = np.array(Dataset('CMIP/MPI_s585_use.nc4').variables['pr_MPI_s585_regrid_b'])[0:540,:,:]*86400
MPI_PET_ssp585 = np.array(Dataset('CMIP/pet_MPI_SSP585.nc').variables['tas'])[0:540,:,:]
MPI_PRPET_ssp585 = MPI_PRAVG_ssp585-MPI_PET_ssp585

MPI_AT2M_ssp245 = np.array(Dataset('CMIP/tas_MPI_RCMgrid_SSP245.nc').variables['tas'])[0:540,:,:]
MPI_PRAVG_ssp245 = np.array(Dataset('CMIP/pr_MPI_RCMgrid_SSP245.nc').variables['pr'])[0:540,:,:]*86400
MPI_PET_ssp245 = np.array(Dataset('CMIP/pet_MPI_SSP245.nc').variables['tas'])[0:540,:,:]
MPI_PRPET_ssp245 = MPI_PRAVG_ssp245-MPI_PET_ssp245

CESM_CWRF_AT2M_hist = np.array(Dataset('CWRF/CESM_CWRF_hist_AT2M.nc').variables['AT2M'])[:,0,:,:]
CESM_CWRF_PRAVG_hist = np.array(Dataset('CWRF/CESM_CWRF_hist_PRAVG.nc').variables['PRAVG'])[:,0,:,:]
CESM_CWRF_PET_hist = np.array(Dataset('CWRF/CESM_CWRF_pet_monmean.nc').variables['AT2M'])[:,:,:]
CESM_CWRF_PRPET_hist = CESM_CWRF_PRAVG_hist-CESM_CWRF_PET_hist

CESM_CWRF_AT2M_ssp585 = np.array(Dataset('CWRF/CESM_CWRF_ssp585_AT2M.nc').variables['AT2M'])[:,0,:,:]
CESM_CWRF_PRAVG_ssp585 = np.array(Dataset('CWRF/CESM_CWRF_ssp585_PRAVG.nc').variables['PRAVG'])[:,0,:,:]
CESM_CWRF_PET_ssp585 = np.array(Dataset('CWRF/SSP585_CESM_CWRF_pet_monmean.nc').variables['AT2M'])[:,:,:]
CESM_CWRF_PRPET_ssp585 = CESM_CWRF_PRAVG_ssp585-CESM_CWRF_PET_ssp585

CESM_CWRF_AT2M_ssp245 = np.array(Dataset('CWRF/CESM_CWRF_ssp245_AT2M.nc').variables['AT2M'])[:,0,:,:]
CESM_CWRF_PRAVG_ssp245 = np.array(Dataset('CWRF/CESM_CWRF_ssp245_PRAVG.nc').variables['PRAVG'])[:,0,:,:]
CESM_CWRF_PET_ssp245 = np.array(Dataset('CWRF/SSP245_CESM_CWRF_pet_monmean.nc').variables['__xarray_dataarray_variable__'])[:,:,:]
CESM_CWRF_PRPET_ssp245 = CESM_CWRF_PRAVG_ssp245-CESM_CWRF_PET_ssp245

MPI_CWRF_AT2M_hist = np.array(Dataset('CWRF/MPI_CWRF_hist_AT2M.nc').variables['AT2M'])[:,:,:]
MPI_CWRF_PRAVG_hist = np.array(Dataset('CWRF/MPI_CWRF_hist_PRAVG.nc').variables['PRAVG'])[:,:,:]*86400
MPI_CWRF_PET_hist = np.array(Dataset('CWRF/MPI_CWRF_pet_monmean.nc').variables['AT2M'])[:,:,:]
MPI_CWRF_PRAVG_hist = np.where(np.abs(MPI_CWRF_PRAVG_hist)>10000,np.nan,MPI_CWRF_PRAVG_hist)
MPI_CWRF_PET_hist = np.where(np.abs(MPI_CWRF_PET_hist)>10000,np.nan,MPI_CWRF_PET_hist)
MPI_CWRF_PRPET_hist = MPI_CWRF_PRAVG_hist-MPI_CWRF_PET_hist

MPI_CWRF_AT2M_ssp585 = np.array(Dataset('CWRF/MPI_CWRF_ssp585_AT2M.nc').variables['AT2M'])[:,:,:]
MPI_CWRF_PRAVG_ssp585 = np.array(Dataset('CWRF/MPI_CWRF_ssp585_PRAVG.nc').variables['PRAVG'])[:,:,:]*86400
MPI_CWRF_PET_ssp585 = np.array(Dataset('CWRF/SSP585_MPI_CWRF_pet_monmean.nc').variables['AT2M'])[:,:,:]
MPI_CWRF_PRAVG_ssp585 = np.where(np.abs(MPI_CWRF_PRAVG_ssp585)>10000,np.nan,MPI_CWRF_PRAVG_ssp585)
MPI_CWRF_PET_ssp585 = np.where(np.abs(MPI_CWRF_PET_ssp585)>10000,np.nan,MPI_CWRF_PET_ssp585)
MPI_CWRF_PRPET_ssp585 = MPI_CWRF_PRAVG_ssp585-MPI_CWRF_PET_ssp585

MPI_CWRF_AT2M_ssp245 = np.array(Dataset('CWRF/MPI_CWRF_ssp245_AT2M.nc').variables['AT2M'])[:,0,:,:]
MPI_CWRF_PRAVG_ssp245 = np.array(Dataset('CWRF/MPI_CWRF_ssp245_PRAVG.nc').variables['PRAVG'])[:,0,:,:]
MPI_CWRF_PET_ssp245 = np.array(Dataset('CWRF/SSP245_MPI_CWRF_pet_monmean.nc').variables['AT2M'])[:,:,:]
MPI_CWRF_PRPET_ssp245 = MPI_CWRF_PRAVG_ssp245-MPI_CWRF_PET_ssp245

CESM_LWRF_AT2M_hist = np.array(Dataset('CWRF/CESM_LWRF_hist_AT2M.nc').variables['__xarray_dataarray_variable__'])[:,:,:]
CESM_LWRF_PRAVG_hist = np.array(Dataset('CWRF/CESM_LWRF_hist_PRAVG.nc').variables['__xarray_dataarray_variable__'])[:,:,:]
CESM_LWRF_PET_hist = np.array(Dataset('CWRF/CESM_LWRF_hist_pet_monmean.nc').variables['__xarray_dataarray_variable__'])[:,:,:]
CESM_LWRF_PRPET_hist = CESM_LWRF_PRAVG_hist-CESM_LWRF_PET_hist

#This is to avoid the negative value in the PR-PET which will cause failure in the calculation of SPEI

ERA5_PRPET_hist = ERA5_PRPET_hist + 11

CESM_PRPET_hist = CESM_PRPET_hist + 11
CESM_PRPET_ssp585 = CESM_PRPET_ssp585 + 11
CESM_PRPET_ssp245 = CESM_PRPET_ssp245 + 11

MPI_PRPET_hist = MPI_PRPET_hist + 11
MPI_PRPET_ssp585 = MPI_PRPET_ssp585 + 11
MPI_PRPET_ssp245 = MPI_PRPET_ssp245 + 11

CESM_CWRF_PRPET_hist = CESM_CWRF_PRPET_hist + 11
CESM_CWRF_PRPET_ssp585 = CESM_CWRF_PRPET_ssp585 + 11
CESM_CWRF_PRPET_ssp245 = CESM_CWRF_PRPET_ssp245 + 11

MPI_CWRF_PRPET_hist = MPI_CWRF_PRPET_hist + 11
MPI_CWRF_PRPET_ssp585 = MPI_CWRF_PRPET_ssp585 + 11
MPI_CWRF_PRPET_ssp245 = MPI_CWRF_PRPET_ssp245 + 11

CESM_LWRF_PRPET_hist = CESM_LWRF_PRPET_hist + 11


#Also add value to PRAVG to align with PR-PET As 

ERA5_PRAVG_hist = ERA5_PRAVG_hist + 11

CESM_PRAVG_hist = CESM_PRAVG_hist + 11
CESM_PRAVG_ssp585 = CESM_PRAVG_ssp585 + 11
CESM_PRAVG_ssp245 = CESM_PRAVG_ssp245 + 11

MPI_PRAVG_hist = MPI_PRAVG_hist + 11
MPI_PRAVG_ssp585 = MPI_PRAVG_ssp585 + 11
MPI_PRAVG_ssp245 = MPI_PRAVG_ssp245 + 11

CESM_CWRF_PRAVG_hist = CESM_CWRF_PRAVG_hist + 11
CESM_CWRF_PRAVG_ssp585 = CESM_CWRF_PRAVG_ssp585 + 11
CESM_CWRF_PRAVG_ssp245 = CESM_CWRF_PRAVG_ssp245 + 11

MPI_CWRF_PRAVG_hist = MPI_CWRF_PRAVG_hist + 11
MPI_CWRF_PRAVG_ssp585 = MPI_CWRF_PRAVG_ssp585 + 11
MPI_CWRF_PRAVG_ssp245 = MPI_CWRF_PRAVG_ssp245 + 11

CESM_LWRF_PRAVG_hist = CESM_LWRF_PRAVG_hist + 11



def process_pixel(args):
    #calculate one pixel job
    ilat, ilon = args
    return (ilat, ilon,
            ref_hist_obs_drought_index_SPISPEI(ERA5_PRAVG_hist[:,ilat,ilon],ERA5_PRPET_hist[:,ilat,ilon],ERA5_AT2M_hist[:,ilat,ilon],ERA5_PRAVG_hist[:,ilat,ilon],ERA5_PRPET_hist[:,ilat,ilon],ERA5_AT2M_hist[:,ilat,ilon],ERA5_PRAVG_hist[:,ilat,ilon],ERA5_PRPET_hist[:,ilat,ilon],ERA5_AT2M_hist[:,ilat,ilon]),
            
            ref_hist_obs_drought_index_SPISPEI(CESM_PRAVG_hist[:,ilat,ilon],CESM_PRPET_hist[:,ilat,ilon],CESM_AT2M_hist[:,ilat,ilon],CESM_PRAVG_hist[:,ilat,ilon],CESM_PRPET_hist[:,ilat,ilon],CESM_AT2M_hist[:,ilat,ilon],CESM_PRAVG_ssp585[:,ilat,ilon],CESM_PRPET_ssp585[:,ilat,ilon],CESM_AT2M_ssp585[:,ilat,ilon]),
            ref_hist_obs_drought_index_SPISPEI(CESM_PRAVG_hist[:,ilat,ilon],CESM_PRPET_hist[:,ilat,ilon],CESM_AT2M_hist[:,ilat,ilon],CESM_PRAVG_hist[:,ilat,ilon],CESM_PRPET_hist[:,ilat,ilon],CESM_AT2M_hist[:,ilat,ilon],CESM_PRAVG_ssp245[:,ilat,ilon],CESM_PRPET_ssp245[:,ilat,ilon],CESM_AT2M_ssp245[:,ilat,ilon]),
            
            ref_hist_obs_drought_index_SPISPEI(MPI_PRAVG_hist[:,ilat,ilon],MPI_PRPET_hist[:,ilat,ilon],MPI_AT2M_hist[:,ilat,ilon],MPI_PRAVG_hist[:,ilat,ilon],MPI_PRPET_hist[:,ilat,ilon],MPI_AT2M_hist[:,ilat,ilon],MPI_PRAVG_ssp585[:,ilat,ilon],MPI_PRPET_ssp585[:,ilat,ilon],MPI_AT2M_ssp585[:,ilat,ilon]),
            ref_hist_obs_drought_index_SPISPEI(MPI_PRAVG_hist[:,ilat,ilon],MPI_PRPET_hist[:,ilat,ilon],MPI_AT2M_hist[:,ilat,ilon],MPI_PRAVG_hist[:,ilat,ilon],MPI_PRPET_hist[:,ilat,ilon],MPI_AT2M_hist[:,ilat,ilon],MPI_PRAVG_ssp245[:,ilat,ilon],MPI_PRPET_ssp245[:,ilat,ilon],MPI_AT2M_ssp245[:,ilat,ilon]),
            
            ref_hist_obs_drought_index_SPISPEI(CESM_CWRF_PRAVG_hist[:,ilat,ilon],CESM_CWRF_PRPET_hist[:,ilat,ilon],CESM_CWRF_AT2M_hist[:,ilat,ilon],CESM_CWRF_PRAVG_hist[:,ilat,ilon],CESM_CWRF_PRPET_hist[:,ilat,ilon],CESM_CWRF_AT2M_hist[:,ilat,ilon],CESM_CWRF_PRAVG_ssp585[:,ilat,ilon],CESM_CWRF_PRPET_ssp585[:,ilat,ilon],CESM_CWRF_AT2M_ssp585[:,ilat,ilon]),
            ref_hist_obs_drought_index_SPISPEI(CESM_CWRF_PRAVG_hist[:,ilat,ilon],CESM_CWRF_PRPET_hist[:,ilat,ilon],CESM_CWRF_AT2M_hist[:,ilat,ilon],CESM_CWRF_PRAVG_hist[:,ilat,ilon],CESM_CWRF_PRPET_hist[:,ilat,ilon],CESM_CWRF_AT2M_hist[:,ilat,ilon],CESM_CWRF_PRAVG_ssp245[:,ilat,ilon],CESM_CWRF_PRPET_ssp245[:,ilat,ilon],CESM_CWRF_AT2M_ssp245[:,ilat,ilon]),
            
            ref_hist_obs_drought_index_SPISPEI(MPI_CWRF_PRAVG_hist[:,ilat,ilon],MPI_CWRF_PRPET_hist[:,ilat,ilon],MPI_CWRF_AT2M_hist[:,ilat,ilon],MPI_CWRF_PRAVG_hist[:,ilat,ilon],MPI_CWRF_PRPET_hist[:,ilat,ilon],MPI_CWRF_AT2M_hist[:,ilat,ilon],MPI_CWRF_PRAVG_ssp585[:,ilat,ilon],MPI_CWRF_PRPET_ssp585[:,ilat,ilon],MPI_CWRF_AT2M_ssp585[:,ilat,ilon]),
            ref_hist_obs_drought_index_SPISPEI(MPI_CWRF_PRAVG_hist[:,ilat,ilon],MPI_CWRF_PRPET_hist[:,ilat,ilon],MPI_CWRF_AT2M_hist[:,ilat,ilon],MPI_CWRF_PRAVG_hist[:,ilat,ilon],MPI_CWRF_PRPET_hist[:,ilat,ilon],MPI_CWRF_AT2M_hist[:,ilat,ilon],MPI_CWRF_PRAVG_ssp245[:,ilat,ilon],MPI_CWRF_PRPET_ssp245[:,ilat,ilon],MPI_CWRF_AT2M_ssp245[:,ilat,ilon]),
            
            ref_hist_obs_drought_index_SPISPEI(CESM_LWRF_PRAVG_hist[:,ilat,ilon],CESM_LWRF_PRPET_hist[:,ilat,ilon],CESM_LWRF_AT2M_hist[:,ilat,ilon],CESM_LWRF_PRAVG_hist[:,ilat,ilon],CESM_LWRF_PRPET_hist[:,ilat,ilon],CESM_LWRF_AT2M_hist[:,ilat,ilon],CESM_LWRF_PRAVG_hist[:,ilat,ilon],CESM_LWRF_PRPET_hist[:,ilat,ilon],CESM_LWRF_AT2M_hist[:,ilat,ilon]),)
            


lat_lon_pairs = [(ilat, ilon) for ilat in range(171) for ilon in range(231)]
#mask out lat_lon_pairs where ERA5 is nan
lat_lon_pairs = [(ilat, ilon) for ilat, ilon in lat_lon_pairs if not np.isnan(ERA5_AT2M_hist[0, ilat, ilon])]

#create a pool of workers
pool = Pool(processes=78)  
results = list(tqdm(pool.imap(process_pixel, lat_lon_pairs), total=len(lat_lon_pairs)))

# create empty arrays to store the results
ERA5_hist_chd = np.ndarray((540,171,231))
ERA5_hist_chp = np.ndarray((540,171,231))
ERA5_hist_spi = np.ndarray((540,171,231))
ERA5_hist_spei = np.ndarray((540,171,231))
ERA5_hist_sti = np.ndarray((540,171,231))

ERA5_ssp585_chd = np.ndarray((540,171,231))
ERA5_ssp585_chp = np.ndarray((540,171,231))
ERA5_ssp585_spi = np.ndarray((540,171,231))
ERA5_ssp585_spei = np.ndarray((540,171,231))
ERA5_ssp585_sti = np.ndarray((540,171,231))
#==============hist===========================
CESM_hist_chd = np.ndarray((540,171,231))
CESM_hist_chp = np.ndarray((540,171,231))
CESM_hist_spi = np.ndarray((540,171,231))
CESM_hist_spei = np.ndarray((540,171,231))
CESM_hist_sti = np.ndarray((540,171,231))
MPI_hist_chd = np.ndarray((540,171,231))
MPI_hist_chp = np.ndarray((540,171,231))
MPI_hist_spi = np.ndarray((540,171,231))
MPI_hist_spei = np.ndarray((540,171,231))
MPI_hist_sti = np.ndarray((540,171,231))
CESM_CWRF_hist_chd = np.ndarray((540,171,231))
CESM_CWRF_hist_chp = np.ndarray((540,171,231))
CESM_CWRF_hist_spi = np.ndarray((540,171,231))
CESM_CWRF_hist_spei = np.ndarray((540,171,231))
CESM_CWRF_hist_sti = np.ndarray((540,171,231))
MPI_CWRF_hist_chd = np.ndarray((540,171,231))
MPI_CWRF_hist_chp = np.ndarray((540,171,231))
MPI_CWRF_hist_spi = np.ndarray((540,171,231))
MPI_CWRF_hist_spei = np.ndarray((540,171,231))
MPI_CWRF_hist_sti = np.ndarray((540,171,231))
#=========================================


#==============SSP585===========================
CESM_ssp585_chd = np.ndarray((540,171,231))
CESM_ssp585_chp = np.ndarray((540,171,231))
CESM_ssp585_spi = np.ndarray((540,171,231))
CESM_ssp585_spei = np.ndarray((540,171,231))
CESM_ssp585_sti = np.ndarray((540,171,231))
MPI_ssp585_chd = np.ndarray((540,171,231))
MPI_ssp585_chp = np.ndarray((540,171,231))
MPI_ssp585_spi = np.ndarray((540,171,231))
MPI_ssp585_spei = np.ndarray((540,171,231))
MPI_ssp585_sti = np.ndarray((540,171,231))
CESM_CWRF_ssp585_chd = np.ndarray((540,171,231))
CESM_CWRF_ssp585_chp = np.ndarray((540,171,231))
CESM_CWRF_ssp585_spi = np.ndarray((540,171,231))
CESM_CWRF_ssp585_spei = np.ndarray((540,171,231))
CESM_CWRF_ssp585_sti = np.ndarray((540,171,231))
MPI_CWRF_ssp585_chd = np.ndarray((540,171,231))
MPI_CWRF_ssp585_chp = np.ndarray((540,171,231))
MPI_CWRF_ssp585_spi = np.ndarray((540,171,231))
MPI_CWRF_ssp585_spei = np.ndarray((540,171,231))
MPI_CWRF_ssp585_sti = np.ndarray((540,171,231))
#=========================================
#==============SSP245===========================
CESM_ssp245_chd = np.ndarray((540,171,231))
CESM_ssp245_chp = np.ndarray((540,171,231))
CESM_ssp245_spi = np.ndarray((540,171,231))
CESM_ssp245_spei = np.ndarray((540,171,231))
CESM_ssp245_sti = np.ndarray((540,171,231))
MPI_ssp245_chd = np.ndarray((540,171,231))
MPI_ssp245_chp = np.ndarray((540,171,231))
MPI_ssp245_spi = np.ndarray((540,171,231))
MPI_ssp245_spei = np.ndarray((540,171,231))
MPI_ssp245_sti = np.ndarray((540,171,231))
CESM_CWRF_ssp245_chd = np.ndarray((540,171,231))
CESM_CWRF_ssp245_chp = np.ndarray((540,171,231))
CESM_CWRF_ssp245_spi = np.ndarray((540,171,231))
CESM_CWRF_ssp245_spei = np.ndarray((540,171,231))
CESM_CWRF_ssp245_sti = np.ndarray((540,171,231))
MPI_CWRF_ssp245_chd = np.ndarray((540,171,231))
MPI_CWRF_ssp245_chp = np.ndarray((540,171,231))
MPI_CWRF_ssp245_spi = np.ndarray((540,171,231))
MPI_CWRF_ssp245_spei = np.ndarray((540,171,231))
MPI_CWRF_ssp245_sti = np.ndarray((540,171,231))
#=========================================

CESM_LWRF_hist_chd = np.ndarray((540,171,231))
CESM_LWRF_hist_chp = np.ndarray((540,171,231))
CESM_LWRF_hist_spei = np.ndarray((540,171,231))
CESM_LWRF_hist_spi = np.ndarray((540,171,231))
CESM_LWRF_hist_sti = np.ndarray((540,171,231))

for res in results:
    #fill in the results
    ilat, ilon, ERA5_res, CESM_res, CESM_res_ssp245, MPI_res, MPI_res_ssp245, CESM_CWRF_res, CESM_CWRF_res_ssp245,MPI_CWRF_res, MPI_CWRF_res_ssp245,CESM_LWRF_res = res
    ERA5_ssp585_chd[:,ilat,ilon], ERA5_ssp585_chp[:,ilat,ilon], ERA5_ssp585_spei[:,ilat,ilon], ERA5_ssp585_spi[:,ilat,ilon], ERA5_ssp585_sti[:,ilat,ilon],    ERA5_hist_chd[:,ilat,ilon], ERA5_hist_chp[:,ilat,ilon], ERA5_hist_spei[:,ilat,ilon], ERA5_hist_spi[:,ilat,ilon], ERA5_hist_sti[:,ilat,ilon],= ERA5_res

    CESM_ssp585_chd[:,ilat,ilon], CESM_ssp585_chp[:,ilat,ilon], CESM_ssp585_spei[:,ilat,ilon], CESM_ssp585_spi[:,ilat,ilon], CESM_ssp585_sti[:,ilat,ilon],    CESM_hist_chd[:,ilat,ilon], CESM_hist_chp[:,ilat,ilon], CESM_hist_spei[:,ilat,ilon], CESM_hist_spi[:,ilat,ilon], CESM_hist_sti[:,ilat,ilon],= CESM_res
    CESM_ssp245_chd[:,ilat,ilon], CESM_ssp245_chp[:,ilat,ilon], CESM_ssp245_spei[:,ilat,ilon], CESM_ssp245_spi[:,ilat,ilon], CESM_ssp245_sti[:,ilat,ilon],    _,_,_,_,_= CESM_res_ssp245

    MPI_ssp585_chd[:,ilat,ilon], MPI_ssp585_chp[:,ilat,ilon], MPI_ssp585_spei[:,ilat,ilon], MPI_ssp585_spi[:,ilat,ilon], MPI_ssp585_sti[:,ilat,ilon],    MPI_hist_chd[:,ilat,ilon], MPI_hist_chp[:,ilat,ilon], MPI_hist_spei[:,ilat,ilon], MPI_hist_spi[:,ilat,ilon], MPI_hist_sti[:,ilat,ilon],= MPI_res
    MPI_ssp245_chd[:,ilat,ilon], MPI_ssp245_chp[:,ilat,ilon], MPI_ssp245_spei[:,ilat,ilon], MPI_ssp245_spi[:,ilat,ilon], MPI_ssp245_sti[:,ilat,ilon],    _,_,_,_,_= MPI_res_ssp245

    CESM_CWRF_ssp585_chd[:,ilat,ilon], CESM_CWRF_ssp585_chp[:,ilat,ilon], CESM_CWRF_ssp585_spei[:,ilat,ilon], CESM_CWRF_ssp585_spi[:,ilat,ilon], CESM_CWRF_ssp585_sti[:,ilat,ilon],    CESM_CWRF_hist_chd[:,ilat,ilon], CESM_CWRF_hist_chp[:,ilat,ilon], CESM_CWRF_hist_spei[:,ilat,ilon], CESM_CWRF_hist_spi[:,ilat,ilon], CESM_CWRF_hist_sti[:,ilat,ilon],= CESM_CWRF_res
    CESM_CWRF_ssp245_chd[:,ilat,ilon], CESM_CWRF_ssp245_chp[:,ilat,ilon], CESM_CWRF_ssp245_spei[:,ilat,ilon], CESM_CWRF_ssp245_spi[:,ilat,ilon], CESM_CWRF_ssp245_sti[:,ilat,ilon],    _,_,_,_,_= CESM_CWRF_res_ssp245

    MPI_CWRF_ssp585_chd[:,ilat,ilon], MPI_CWRF_ssp585_chp[:,ilat,ilon], MPI_CWRF_ssp585_spei[:,ilat,ilon], MPI_CWRF_ssp585_spi[:,ilat,ilon], MPI_CWRF_ssp585_sti[:,ilat,ilon],    MPI_CWRF_hist_chd[:,ilat,ilon], MPI_CWRF_hist_chp[:,ilat,ilon], MPI_CWRF_hist_spei[:,ilat,ilon], MPI_CWRF_hist_spi[:,ilat,ilon], MPI_CWRF_hist_sti[:,ilat,ilon],= MPI_CWRF_res
    MPI_CWRF_ssp245_chd[:,ilat,ilon], MPI_CWRF_ssp245_chp[:,ilat,ilon], MPI_CWRF_ssp245_spei[:,ilat,ilon], MPI_CWRF_ssp245_spi[:,ilat,ilon], MPI_CWRF_ssp245_sti[:,ilat,ilon],    _,_,_,_,_= MPI_CWRF_res_ssp245

    _,_,_,_,_,CESM_LWRF_hist_chd[:,ilat,ilon], CESM_LWRF_hist_chp[:,ilat,ilon], CESM_LWRF_hist_spei[:,ilat,ilon], CESM_LWRF_hist_spi[:,ilat,ilon], CESM_LWRF_hist_sti[:,ilat,ilon],= CESM_LWRF_res

# Save to netCDF file
outfilename='ALL_CHW_CHD_baseall_allSPEISPI.nc'
os.system('rm -f ' + outfilename)
outfile = Dataset(outfilename, 'w')
outfile.createDimension('lon', 231)
outfile.createDimension('lat', 171)
outfile.createDimension('time', 540)

for modelname in ['ERA5_hist','ERA5_ssp585',
                'CESM_hist','CESM_ssp585','CESM_ssp245',
                'MPI_hist','MPI_ssp585','MPI_ssp245',
                'CESM_CWRF_hist','CESM_CWRF_ssp585','CESM_CWRF_ssp245',
                'MPI_CWRF_hist','MPI_CWRF_ssp585','MPI_CWRF_ssp245',
                'CESM_LWRF_hist']:
    outfile.createVariable(modelname+'_chd', 'f4', ('time','lat','lon'))
    outfile.createVariable(modelname+'_chp', 'f4', ('time','lat','lon'))
    outfile.createVariable(modelname+'_spei', 'f4', ('time','lat','lon'))
    outfile.createVariable(modelname+'_spi', 'f4', ('time','lat','lon'))
    outfile.createVariable(modelname+'_sti', 'f4', ('time','lat','lon'))
    outfile.variables[modelname+'_chd'][:] = eval(modelname+'_chd')
    outfile.variables[modelname+'_chp'][:] = eval(modelname+'_chp')
    outfile.variables[modelname+'_spei'][:] = eval(modelname+'_spei')
    outfile.variables[modelname+'_spi'][:] = eval(modelname+'_spi')
    outfile.variables[modelname+'_sti'][:] = eval(modelname+'_sti')

outfile.close()

