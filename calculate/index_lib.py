import numpy as np
from scipy.stats import genextreme as gev
from scipy.stats import gamma, lognorm, norm, gennorm, expon, pearson3
from scipy import stats
import pyvinecopulib as pv


def select_distribution(c):
    distribution = norm, lognorm, pearson3, gamma, gev, gennorm
    for fenbu in distribution:
        t = fenbu.fit(c)
        b1 = stats.kstest(c, fenbu.cdf, args=(t), alternative='two-sided')
        if b1[1] > 0.05:
            return fenbu
    return "404"


def standard_index(yearly_input):
    #input: yearly_input 1D numpy array of multiple_year_monthly_records
    #output: standardized index, bestfit distribution for this month
    shape_of_input = int(yearly_input.shape[0])
    output = np.ndarray(shape_of_input)
    bestfit_distribution = None
    multiyear_sum = np.sum(yearly_input)
    zeros_in_multiyear = np.sum(yearly_input == 0)
    std_of_multiyear = np.std(yearly_input)
    if multiyear_sum != 0 and zeros_in_multiyear <= 0.5 * shape_of_input and std_of_multiyear != 0:
        arr_nozero = yearly_input[yearly_input != 0]
        distribution = select_distribution(arr_nozero)
        bestfit_distribution = distribution
        if bestfit_distribution != "404":
            fit_params = bestfit_distribution.fit(arr_nozero)
            if len(fit_params) == 2:
                nonzero_len = len(arr_nozero)
                for i in range(0, shape_of_input):
                    if yearly_input[i] != 0:
                        x = yearly_input[i]
                        output[i] = bestfit_distribution.cdf(x, fit_params[0], fit_params[1]) * nonzero_len / shape_of_input + 1 - (
                                nonzero_len / shape_of_input)
                    else:
                        output[i] = 1 - (nonzero_len / shape_of_input)
            elif len(fit_params) == 3:
                nonzero_len = len(arr_nozero)
                for i in range(0, shape_of_input):
                    if yearly_input[i] != 0:
                        x = yearly_input[i]
                        output[i] = bestfit_distribution.cdf(x, fit_params[0], fit_params[1], fit_params[2]) * nonzero_len / shape_of_input + 1 - (
                                nonzero_len / shape_of_input)
                    else:
                        output[i] = 1 - (nonzero_len / shape_of_input)
        elif bestfit_distribution == "404":
            fit_params=np.nan
            empirical_distribution = np.sort(yearly_input)
            empirical_distribution = empirical_distribution.tolist()
            for l in range(0, shape_of_input):
                output[l] = (empirical_distribution.index(yearly_input[l]) + 1) / shape_of_input
        
        cdf_mean = np.mean(output)
        cdf_st = np.std(output)
        output_normalized = (output - cdf_mean) / cdf_st
        return output,cdf_mean,cdf_st,bestfit_distribution,fit_params


def monthly_calculation_Standardized_index(monthly_input):
    #input: monthly_input is a 1D numpy array that can be divided by 12
    #output: standardized index, bestfit distribution for each month
    shape_of_input = monthly_input.shape
    shape_of_output = (int(shape_of_input[0]/12), 12)
    SDI = np.ndarray(shape_of_output)
    mean_SDI = np.ndarray(12)
    std_SDI = np.ndarray(12)
    output_distribution = []
    fit_params = []
    monthly_input = monthly_input.reshape(shape_of_output)
    for imon in range(12):
        index, mean, std, dist, params = standard_index(monthly_input[:, imon])
        SDI[:, imon] = index
        mean_SDI[imon] = mean
        std_SDI[imon] = std
        output_distribution.append(dist)
        fit_params.append(params)
    return SDI.flatten(), mean_SDI, std_SDI, output_distribution, fit_params


def apply_distribution(fut_data, mean_SDI, std_SDI, distributions, in_params):
    fut_data = fut_data.reshape((int(len(fut_data) / 12), 12))
    fut_index = np.ndarray(fut_data.shape)
    for imon in range(12):
        dist = distributions[imon]
        params = in_params[imon]
        if dist != "404":
            if len(params) == 2:
                cdf_values = dist.cdf(fut_data[:, imon], params[0], params[1])
            elif len(params) == 3:
                cdf_values = dist.cdf(fut_data[:, imon], params[0], params[1], params[2])
            fut_index[:, imon] = cdf_values
        else:
            empirical_distribution = np.sort(fut_data[:, imon])
            empirical_distribution = empirical_distribution.tolist()
            for j in range(len(fut_data[:, imon])):
                fut_index[j, imon] = (empirical_distribution.index(fut_data[j, imon]) + 1) / len(fut_data[:, imon])
    fut_normalized=np.ndarray(fut_index.shape)
    for imon in range(12):
        fut_normalized[:, imon] = (fut_index[:, imon] - mean_SDI[imon]) / std_SDI[imon]
    return fut_index.flatten()

def empirical_distribution_function(values):
    sorted_indices = np.argsort(-values)
    sorted_order = np.argsort(sorted_indices) + 1
    result = (len(values) - sorted_order + 0.5) / len(values)
    return result

def calculation_composite_index_hist(SPI_data, STI_data):
    SPI_data_em = SPI_data
    STI_data_em = STI_data
    combined_array = np.column_stack((SPI_data_em, STI_data_em))
    copula_set = [pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton,
                  pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe]
    min_bic = float('inf')
    best_copula = None
    for copula in copula_set:
        cop = pv.Bicop(copula)
        bic = cop.bic(combined_array)
        if bic < min_bic:
            min_bic = bic
            best_copula = copula
    copula = pv.Bicop(best_copula)
    copula.fit(combined_array)
    copula_cdf = copula.cdf(combined_array)
    return copula_cdf, copula

def calculation_composite_index(SPI_data, STI_data, copula):
    SPI_data_em = SPI_data
    STI_data_em = STI_data
    combined_array = np.column_stack((SPI_data_em, STI_data_em))
    copula_cdf = copula.cdf(combined_array)
    return copula_cdf





def ref_hist_obs_drought_index_SPISPEI(obs_pr_hist_data,obs_prpet_hist_data,obs_tas_hist_data,pr_hist_data, prpet_hist_data,tas_hist_data,pr_fut_data,prpet_fut_data,tas_fut_data):
    
    #if input series contains nan, return nan
    if np.isnan(pr_hist_data).any() or np.isnan(tas_hist_data).any() or np.isnan(obs_pr_hist_data).any() or np.isnan(obs_tas_hist_data).any() or np.isnan(prpet_hist_data).any() or np.isnan(obs_prpet_hist_data).any() or np.isnan(pr_fut_data).any() or np.isnan(tas_fut_data).any() or np.isnan(prpet_fut_data).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Step 0: calculate baseline period for STI
    STI_obs, STI_mean_obs, STI_std_obs, STI_distributions_obs,fit_params = monthly_calculation_Standardized_index(tas_hist_data) 
    # Step 1: apply the distribution to hist temperature data
    STI_hist = apply_distribution(tas_hist_data, STI_mean_obs, STI_std_obs, STI_distributions_obs, fit_params)
    # Step 2: apply the distribution to future temperature data
    STI_fut = apply_distribution(tas_fut_data, STI_mean_obs, STI_std_obs, STI_distributions_obs, fit_params)
    # print(STI_fut.shape)

    # Step 3: calculate baseline period for SPI
    SPI_obs, SPI_mean_obs, SPI_std_obs, SPI_distributions_obs,fit_params = monthly_calculation_Standardized_index(pr_hist_data)
    # Step 4: apply the distribution to hist precipitation data
    SPI_hist = apply_distribution(pr_hist_data, SPI_mean_obs, SPI_std_obs, SPI_distributions_obs, fit_params)
    # Step 5: apply the distribution to future precipitation data
    SPI_fut = apply_distribution(pr_fut_data, SPI_mean_obs, SPI_std_obs, SPI_distributions_obs, fit_params)
    # Adjust sign of SPI
    SPI_hist = 1 - SPI_hist
    SPI_fut = 1 - SPI_fut

    # Step 6: calculate baseline period for SPEI
    SPEI_obs, SPEI_mean_obs, SPEI_std_obs, SPEI_distributions_obs,fit_params = monthly_calculation_Standardized_index(prpet_hist_data)
    # Step 7: apply the distribution to hist PR-PET data
    SPEI_hist = apply_distribution(prpet_hist_data, SPEI_mean_obs, SPEI_std_obs, SPEI_distributions_obs, fit_params)
    # Step 8: apply the distribution to future PR-PET data
    SPEI_fut = apply_distribution(prpet_fut_data, SPEI_mean_obs, SPEI_std_obs, SPEI_distributions_obs, fit_params)
    # Adjust sign of SPEI
    SPEI_hist = 1 - SPEI_hist
    SPEI_fut = 1 - SPEI_fut


    # Step 9: calculate baseline period for CHD
    chd_hist_obs, copula_dry_hist_obs = calculation_composite_index_hist(SPEI_obs, STI_obs)
    # Step 10: apply the distribution to hist composite index data
    chd_hist = calculation_composite_index(SPEI_hist, STI_hist, copula_dry_hist_obs)
    # Step 11: apply the distribution to future composite index data
    chd_fut = calculation_composite_index(SPEI_fut, STI_fut, copula_dry_hist_obs)

    # Step 12: calculate baseline period for CHD based on SPI (discarded)
    chp_hist_obs, copula_dry_hist_obs = calculation_composite_index_hist(SPI_obs, STI_obs)
    # Step 13: apply the distribution to hist composite index data based on SPI (discarded)
    chp_hist = calculation_composite_index(SPI_hist, STI_hist, copula_dry_hist_obs)
    # Step 14: apply the distribution to future composite index data based on SPI (discarded)
    chp_fut = calculation_composite_index(SPI_fut, STI_fut, copula_dry_hist_obs)

    return chd_fut,chp_fut, SPEI_fut, SPI_fut, STI_fut, chd_hist,chp_hist,SPEI_hist, SPI_hist, STI_hist

