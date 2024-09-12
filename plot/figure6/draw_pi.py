from netCDF4 import Dataset
import numpy as np
from multiprocessing import Process, Manager
import os
from scipy import stats
import cmaps
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cnmaps import draw_maps,get_adm_maps
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
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

def calculate_rmse_pcc(OBS,MOD):
    #OBS and MOD are 2d
    OBS=OBS.flatten()
    MOD=MOD.flatten()
    OBS=OBS[~np.isnan(MOD)]
    MOD=MOD[~np.isnan(MOD)]
    MOD=MOD[~np.isnan(OBS)]
    OBS=OBS[~np.isnan(OBS)]
    rmse=np.sqrt(np.nanmean((OBS-MOD)**2))
    pcc=np.corrcoef(OBS,MOD)[0,1]
    return pcc,rmse


def mapdraw(lon,lat,matrix,seas,model,cm,diff_levs,CNmaskraw,pcc,rmse,figl):
    figname="coupling_"+seas+"_"+model
    fig=plt.figure(figsize=(5,4))
    cwrf_cnproj = ccrs.LambertConformal(central_longitude=110.0, central_latitude=35.17781, false_easting=0.0, false_northing=0.0,  standard_parallels = (30, 60), globe=None, cutoff=-30)
    ax = plt.subplot(1,1,1, projection=cwrf_cnproj)
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.3, zorder=99)
    ax.add_feature(cfeat.OCEAN, facecolor="white", zorder=10)
    draw_maps(get_adm_maps(level='国'),color='grey',zorder=99,linewidth=0.8)
    draw_maps(get_adm_maps(level='省'),color='grey',zorder=99,linewidth=0.4)
    ax.set_extent([101,126,18,54])
    # cn=ax.pcolormesh(lon,lat,matrix,cmap=cm,vmin=diff_levs[0],vmax=diff_levs[-1],transform = ccrs.PlateCarree())
    cn=ax.contourf(lon,lat,matrix,levels=diff_levs,cmap=cm,transform = ccrs.PlateCarree(),extend='both')
    CNmaskraw_local=cut_edge(CNmaskraw)
    CNmaskraw_local=np.where(CNmaskraw_local==-1,-1,np.nan)
    ax.contourf(lon,lat,CNmaskraw_local,levels=[-1,0],colors='grey',transform = ccrs.PlateCarree(),zorder=100)
    ax.text(0.02,0.98,figl+") "+model,transform=ax.transAxes,fontsize=10,ha='left',va='top',fontweight="bold")
    
    plt.savefig('./pngs/'+figname+'.png',dpi=300,bbox_inches='tight')
    os.system("convert -trim ./pngs/"+figname+'.png ./pngs/'+figname+'.png')
        
    if model=="ERA5Land" or model=="ERA5Land_trend":
        fig=plt.figure(figsize=(20,1))
        cbar_ax = fig.add_axes([0.1, 0.07, 0.85, 0.2])
        cbar=plt.colorbar(cn,cax=cbar_ax,label="",orientation='horizontal',shrink=0.02,extend='both')
        if model=="ERA5Land":
            cbar.set_ticks(diff_levs[::1])
        else:
            cbar.set_ticks(diff_levs[::2])
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("Land-Air Coupling Strength Index",fontsize=18,fontweight="bold")
        plt.savefig("./pngs/"+figname+'_diffcolorbar_'+'.png',bbox_inches='tight')




def barplot_PCC_RMSE(OBS,CESM,MPI,CWRF_CESM,CWRF_MPI,varname):
    if len(OBS.shape)==3:
        OBS_clim=np.nanmean(OBS,axis=0)
        CESM_clim=np.nanmean(CESM,axis=0)
        MPI_clim=np.nanmean(MPI,axis=0)
        CWRF_CESM_clim=np.nanmean(CWRF_CESM,axis=0)
        CWRF_MPI_clim=np.nanmean(CWRF_MPI,axis=0)
    else:
        OBS_clim=OBS
        CESM_clim=CESM
        MPI_clim=MPI
        CWRF_CESM_clim=CWRF_CESM
        CWRF_MPI_clim=CWRF_MPI
    CESM_PCC,CESM_RMSE=calculate_rmse_pcc(OBS_clim,CESM_clim)
    MPI_PCC,MPI_RMSE=calculate_rmse_pcc(OBS_clim,MPI_clim)
    CWRF_CESM_PCC,CWRF_CESM_RMSE=calculate_rmse_pcc(OBS_clim,CWRF_CESM_clim)
    CWRF_MPI_PCC,CWRF_MPI_RMSE=calculate_rmse_pcc(OBS_clim,CWRF_MPI_clim)
    print(CESM_PCC,MPI_PCC,CWRF_CESM_PCC,CWRF_MPI_PCC)
    categories = ['CESM','CESM_CWRF','MPI','MPI_CWRF']
    values1 = [CESM_PCC,CWRF_CESM_PCC,MPI_PCC,CWRF_MPI_PCC]
    values2 = [CESM_RMSE,CWRF_CESM_RMSE,MPI_RMSE,CWRF_MPI_RMSE]


    fig=plt.figure(figsize=(10.5,5))
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1,ax2 = fig.subplots(1, 2, sharex=True)
    ax1.bar(categories, values1, color=['pink','red','lightblue','blue'],width=0.4)
    if varname=="ANN_trend":
        ax1.set_ylim((-0.2,0.2))
    else:
        ax1.set_ylim((0.,1))
    ax1.set_xlim((-0.5,3.5))
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_ylabel('PCC',fontsize=18,fontweight="bold")

    #annotate the text and value vertically on bar
    for a,b in zip(categories, values1):
        if b<0:
            dir="top"
        else:
            dir="bottom"
        ax1.text(a, b, '%.2f' % b, ha='center', va= dir,fontsize=16,fontweight="bold")
        ax1.text(a,0.025, a, ha='center',color='white', va= 'bottom',fontsize=12,fontweight="bold",rotation=90)

    ax1.text(0.12,1.1,"b) Bar charts",transform=ax1.transAxes,fontsize=15,ha='right',va='top',fontweight="bold")

    ax2.bar(categories, values2, color=['pink','red','lightblue','blue'],width=0.4)
    if varname=="ANN_trend":
        ax2.set_ylim((0,0.05))
    else:
        ax2.set_ylim((0.,0.25))
    ax2.set_xlim((-0.5,3.5))
    ax2.set_ylabel('RMSE',fontsize=18,fontweight="bold")
    # ax2.set_xticklabels(['CESM',  'CWRF_CESM','MPI', 'CWRF_MPI'],fontsize=15,fontweight="bold")
    ax2.set_xticklabels(['',  '','', ''],fontsize=15,fontweight="bold")
    ax2.tick_params(axis='y', labelsize=16)
    for a,b in zip(categories, values2):
        if b<0:
            dir="top"
        else:
            dir="bottom"
        ax2.text(a, b, '%.2f' % b, ha='center', va= dir,fontsize=16,fontweight="bold")
        ax2.text(a,0.003, a, ha='center',color='white', va= 'bottom',fontsize=12,fontweight="bold",rotation=90)
    
    plt.tight_layout()
    plt.savefig("./pngs/"+varname+"_rmsebar.png",bbox_inches='tight')


CNregionfil=Dataset("./newmask2.nc")
CNmaskraw=CNregionfil.variables["reg_mask"][:]
CNmaskraw=np.where(CNmaskraw==7,-1,CNmaskraw)
CNmaskraw=np.where(CNmaskraw==8,-1,CNmaskraw)
CNmaskraw=np.where(CNmaskraw==9,-1,CNmaskraw)
CNmask=cut_edge(CNmaskraw)

CNregionfil=Dataset("./CN_Subregion_new.nc")
lat2d=np.array(CNregionfil.variables["latitude"][:])
lon2d=np.array(CNregionfil.variables["longitude"][:])
lat2d=cut_edge(lat2d)
lon2d=cut_edge(lon2d)


modelnames=['ERA5Land','CESM','MPI','CWRF_CESM','CWRF_MPI']
modeldisplaynames=['ERA5Land','CESM','MPI','CESM_CWRF','MPI_CWRF']

ERA5Landraw=xr.open_dataset("ERA5Land_pi_yearmean.nc")['Rnet']
CESMraw=xr.open_dataset("CESM2_pi_yearmean.nc")['__xarray_dataarray_variable__']
MPIraw=xr.open_dataset("MPI_pi_yearmean.nc")['__xarray_dataarray_variable__']
CWRF_CESMraw=xr.open_dataset("CWRF_CESM_pi_yearmean.nc")['__xarray_dataarray_variable__']
CWRF_MPIraw=xr.open_dataset("CWRF_MPI_pi_yearmean.nc")['__xarray_dataarray_variable__']



ERA5Land_3d=maskout(cut_edge(ERA5Landraw),CNmask)
CESM_3d=maskout(cut_edge(CESMraw),CNmask)
MPI_3d=maskout(cut_edge(MPIraw),CNmask)
CWRF_CESM_3d=maskout(cut_edge(CWRF_CESMraw),CNmask)
CWRF_MPI_3d=maskout(cut_edge(CWRF_MPIraw),CNmask)


ERA5Land=np.nanmean(ERA5Land_3d,axis=0)
CESM=np.nanmean(CESM_3d,axis=0)
MPI=np.nanmean(MPI_3d,axis=0)
CWRF_CESM=np.nanmean(CWRF_CESM_3d,axis=0)
CWRF_MPI=np.nanmean(CWRF_MPI_3d,axis=0)

E5MASK_pos=np.where(ERA5Land>0.1,1,np.nan)
E5MASK_neg=np.where(ERA5Land<-0.1,1,np.nan)

print(ERA5Land.shape)
print(CESM.shape)
print(MPI.shape)
print(CWRF_CESM.shape)
print(CWRF_MPI.shape)

barplot_PCC_RMSE(ERA5Land,CESM,MPI,CWRF_CESM,CWRF_MPI,"ANN")



