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
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import ks_2samp
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import random
import time
from tqdm import tqdm
import seaborn as sns
p_cri=0.01

globalthresh_lower = 0.02
globalthresh_upper = 0.04

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
    return rmse,pcc

def cleanaligan(OBS,CESM,MPI,CESM_CWRF,MPI_CWRF):
    OBS=np.where(OBS==0,np.nan,OBS)
    CESM=CESM[~np.isnan(OBS)]
    MPI=MPI[~np.isnan(OBS)]
    CESM_CWRF=CESM_CWRF[~np.isnan(OBS)]
    MPI_CWRF=MPI_CWRF[~np.isnan(OBS)]
    OBS=OBS[~np.isnan(OBS)]
    return OBS,CESM,MPI,CESM_CWRF,MPI_CWRF


def calculate_TS(OBS,MODEL,threshold):
    hits=np.nansum(np.where((OBS>threshold)&(MODEL>threshold),1,0))
    misses=np.nansum(np.where((OBS>threshold)&(MODEL<=threshold),1,0))
    falseA=np.nansum(np.where((OBS<=threshold)&(MODEL>threshold),1,0))
    correctN=np.nansum(np.where((OBS<=threshold)&(MODEL<=threshold),1,0))
    hits_random=(hits+misses)*(hits+falseA)/(hits+misses+falseA+correctN)
    TS=(hits)/(hits+misses+falseA)
    return TS


def calculate_TS_pv(OBS,MODEL,threshold):
    hits=np.nansum(np.where((OBS<threshold)&(MODEL<threshold),1,0))
    misses=np.nansum(np.where((OBS<threshold)&(MODEL>=threshold),1,0))
    falseA=np.nansum(np.where((OBS>=threshold)&(MODEL<threshold),1,0))
    correctN=np.nansum(np.where((OBS>=threshold)&(MODEL>=threshold),1,0))
    hits_random=(hits+misses)*(hits+falseA)/(hits+misses+falseA+correctN)
    TS=(hits)/(hits+misses+falseA)
    return TS

def TS_matrix(OBS,CESM,MPI,CESM_CWRF,MPI_CWRF):
    OBS,CESM,MPI,CESM_CWRF,MPI_CWRF=cleanaligan(OBS,CESM,MPI,CESM_CWRF,MPI_CWRF)
    matrix=np.ndarray([5,4])
    for tid,thresh in enumerate([0,globalthresh_lower,globalthresh_upper,0.2]):
        matrix[3-tid,0]=calculate_TS(OBS,CESM,thresh)
        matrix[3-tid,1]=calculate_TS(OBS,CESM_CWRF,thresh)
        matrix[3-tid,2]=calculate_TS(OBS,MPI,thresh)
        matrix[3-tid,3]=calculate_TS(OBS,MPI_CWRF,thresh)
    return matrix


def draw_mosaic_matrix_once(figname,largematrix):
    fig=plt.figure(figsize=(12,3))    
    names=["STI-Hot","SPEI-Dry","CDHE"]
    anos=["a)","b)","c)"]
    largematrix=largematrix[[1,0,2],:,:]
    for vid in range(3):
        ax = plt.subplot(1,3,vid+1)
        matrix=largematrix[vid,[1,2,4],:]
        print(matrix)

        levels=np.linspace(0,1,21)
        cn=sns.heatmap(matrix,annot=True,fmt='.2f',cbar=False,ax=ax,vmin=0,vmax=1,cmap=cmaps.MPL_YlOrRd,linewidths=0.5,linecolor='grey',annot_kws={"size": 14,"weight":"bold","color": "black"})
        for i in range(3):
            for j in range(4):
                if np.isnan(matrix[i,j]):
                    ax.text(j+0.5,i+0.5,"-",ha='center',va='center',fontsize=12)
                else:
                    if j in [1,3]:
                        diff=np.round(matrix[i,j],2)-np.round(matrix[i,j-1],2)
                        if diff>0:
                            color='lime'
                        if diff<0:
                            color='red'
                        if diff==0:
                            color='dimgray'
                        tet="("+str(np.round(diff,2))+")"
                        ax.text(j+0.5,i+0.72,tet,ha='center',va='center',fontweight='bold',fontsize=12,color=color)

        plt.xticks([0.5,1.5,2.5,3.5],["CESM","CESM_CWRF","MPI","MPI_CWRF"],fontsize=8,fontweight="bold")
        plt.yticks([0.5,1.5,2.5],["≥"+str(globalthresh_upper)+"/dec.","≥"+str(globalthresh_lower)+"/dec.","Sig"],fontsize=8,fontweight="bold")
        name=names[vid]
        ano=anos[vid]
        ax.set_title("TS: "+name,fontsize=12,fontweight="bold")
        ax.plot([1,1],[0,5],color='black',linewidth=1.5)
        ax.plot([2,2],[0,5],color='black',linewidth=2.5)
        ax.plot([3,3],[0,5],color='black',linewidth=1.5)
        ax.text(-0.05,1.03,ano,transform=ax.transAxes,fontsize=12,fontweight="bold")
    #add common colorbar
    plt.savefig('./pngs/'+figname+'MATALL.png',dpi=300,bbox_inches='tight')
    os.system("convert -trim ./pngs/"+figname+"MATALL.png ./pngs/"+figname+"MATALL.png")
    fig=plt.figure(figsize=(20,1))
    cbar_ax = fig.add_axes([0.1, 0.07, 0.85, 0.2])
    cbar=plt.colorbar(cn.collections[0],cax=cbar_ax,label="",orientation='horizontal',shrink=0.02,extend='both')
    cbar.ax.tick_params(labelsize=16)
    # set label
    cbar.set_label("Threat Score",fontsize=18,fontweight="bold")
    plt.savefig("./pngs/MATbar.png",bbox_inches='tight')
    time.sleep(3)
    os.system("convert -resize 2857x ./pngs/MATbar.png ./pngs/MATbar.png")
    os.system("convert -append ./pngs/"+figname+"MATALL.png ./pngs/MATbar.png "+figname+"MATALL.png")



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


model=['ERA5','CESM_CWRF','MPI_CWRF','CESM','MPI']
varia=['spei_dry','sti_hot','chd_chd']

trend_dict={}
infil=Dataset("checkcheck.nc","r")
for variab in varia:
    for mod in model:
        trend_key=mod+"_hist_"+variab+"_freq_trend"
        pvalu_key=mod+"_hist_"+variab+"_freq_pvalue"
        trend_dict[trend_key]=np.array(infil[trend_key][:])
        trend_dict[pvalu_key]=np.array(infil[pvalu_key][:])
        trend_key=mod+"_hist_"+variab+"_freq_trend"
        pvalu_key=mod+"_hist_"+variab+"_freq_pvalue"
        trend_dict[trend_key]=np.array(infil[trend_key][:])
        trend_dict[pvalu_key]=np.array(infil[pvalu_key][:])

freq_large_mat=np.ndarray([3,5,4])
inte_large_mat=np.ndarray([3,5,4])

seasname=['DJF','MAM','JJA','SON']
figletters=['e','f','g','h','i']
figletters_future=['xx','e','g','d','f','xxx']
for vid,variab in enumerate(varia):
    OBS=trend_dict["ERA5_hist_"+variab+"_freq_trend"]
    CESM=trend_dict["CESM_hist_"+variab+"_freq_trend"]
    MPI=trend_dict["MPI_hist_"+variab+"_freq_trend"]
    CESM_CWRF=trend_dict["CESM_CWRF_hist_"+variab+"_freq_trend"]
    MPI_CWRF=trend_dict["MPI_CWRF_hist_"+variab+"_freq_trend"]
    ts_mat=TS_matrix(OBS,CESM,MPI,CESM_CWRF,MPI_CWRF)
    ts_mat[4,0]=calculate_TS_pv(trend_dict["ERA5_hist_"+variab+"_freq_pvalue"],trend_dict["CESM_hist_"+variab+"_freq_pvalue"],p_cri)
    ts_mat[4,1]=calculate_TS_pv(trend_dict["ERA5_hist_"+variab+"_freq_pvalue"],trend_dict["CESM_CWRF_hist_"+variab+"_freq_pvalue"],p_cri)
    ts_mat[4,2]=calculate_TS_pv(trend_dict["ERA5_hist_"+variab+"_freq_pvalue"],trend_dict["MPI_hist_"+variab+"_freq_pvalue"],p_cri)
    ts_mat[4,3]=calculate_TS_pv(trend_dict["ERA5_hist_"+variab+"_freq_pvalue"],trend_dict["MPI_CWRF_hist_"+variab+"_freq_pvalue"],p_cri)
    freq_large_mat[vid,:,:]=ts_mat



draw_mosaic_matrix_once("freq",freq_large_mat)
