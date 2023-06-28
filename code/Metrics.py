

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:36:34 2019

@author: reyes-gonzalez
"""
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.stats import epps_singleton_2samp
from scipy.stats import anderson_ksamp
from statistics import mean,median
from typing import List, Tuple, Dict, Callable, Union, Optional

def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """
    Function that computes the correlation matrix from the covariance matrix.
    
    Args:
        covariance: np.ndarray, covariance matrix.
        
    Returns:
        np.ndarray, correlation matrix.
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def input_dist_dimensions(dist: Union[np.ndarray, tfp.distributions.Distribution]
                         ) -> Tuple[int, Optional[int]]:
    if isinstance(dist, np.ndarray):
        if len(dist.shape) != 2:
            raise ValueError("Input must be a 2-dimensional numpy array")
        else:
            nsamples, ndims = dist.shape
    elif isinstance(dist, tfp.distributions.Distribution):
        nsamples, ndims = None, dist.sample(2).numpy().shape[1]
    else:
        raise ValueError("Input must be either a numpy array or a tfp.distributions.Distribution object")
    return ndims, nsamples


def KS_test_1(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
              dist_2: Union[np.ndarray, tfp.distributions.Distribution],
              n_iter: int = 10,
              batch_size: int = 100000
             ) -> Tuple[List[float], List[float]]:
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a p-value that indicates whether the two distributions are the same or not. 
    The test is performed for each dimension of the distributions and for n_iter times and lists of ks_test_statistic and ks_pvalues are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in n_iter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/n_iter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/n_iter) of the numerical distribution. Data are then split in n_iter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other sampled from the symbolic one.
    The ks_test_statistic and ks_pvalue are then computed over all pairs of batches dist_1_j, dist_2_j for all dimensions and the (2D) lists are returned.   
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        n_iter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        ks_metric_list, ks_pvalue_list: Tuble[List[float], List[float]], list of ks_metric and ks_pvalue for each dimension and each iteration.
    """    
    # 
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims = ndims_1
    nsamples: Optional[int] = None
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if isinstance(dist_1, np.ndarray):
            dist_1 = dist_1[:nsamples,:]
        if isinstance(dist_2, np.ndarray):
            dist_2 = dist_2[:nsamples,:]
        batch_size = nsamples // n_iter
    
    # Define ks_list that will contain the list of ks for all dimensions and all iterations
    ks_metric_list: List[float] = []
    ks_pvalue_list: List[float] = []
    
    # Loop over all iterations
    for k in range(n_iter):
        # For both distributions if the distribution is numerical (np.ndarray), then samples are split in n_iter batches of size batch_size = nsamples/n_iter,
        # otherwise, if the distribution is symbolic (tfp.distributions.Distribution), then batch_size points are sampled at each iteration. In the latter case,
        # batch_size could be the input value, if both distributions are symbolic, or nsamples/n_iter if any is numerical.
        if isinstance(dist_1, np.ndarray):
            dist_1_k = dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k = dist_1.sample(batch_size).numpy()
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        # The ks test is computed and the p-value saved for each dimension
        for dim in range(ndims):
            metric, p_val = stats.ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim])
            ks_metric_list.append(metric)
            ks_pvalue_list.append(p_val)
    # Return the mean and std of the p-values
    return ks_metric_list, ks_pvalue_list


def KS_test_2(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
              dist_2: Union[np.ndarray, tfp.distributions.Distribution],
              n_iter: int = 10,
              batch_size: int = 100000
             ) -> Tuple[List[float], List[float]]:
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a p-value that indicates whether the two distributions are the same or not. 
    The test is performed for each dimension of the distributions and for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in np.ceil(np.sqrt(n_iter)) batches dist_1_j, dist_2_k of size batch_size=int(nsamples/np.ceil(np.sqrt(n_iter))) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_k.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """    
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    n_iter=int(np.ceil(np.sqrt(n_iter)))
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define ks_list that will contain the list of ks for all dimensions and all iterations
    ks_list=[]
    # Loop over all iterations
    for j in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_j=dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        for k in range(n_iter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k=dist_2.sample(batch_size).numpy()
            else:   
                raise ValueError("dist_1 must be either a numpy array or a distribution")
            # The ks test is computed and the p-value saved for each dimension
            for dim in range(ndims):
                p_val=stats.ks_2samp(dist_1_j[:,dim], dist_2_k[:,dim])[1]
                ks_list.append(p_val)
    # Compute the mean and std of the p-values
    ks_mean = np.mean(ks_list)
    ks_std = np.std(ks_list)
    # Return the mean and std of the p-values
    return [ks_mean,ks_std,ks_list]


def AD_test_1(dist_1,dist_2,n_iter=10,batch_size=100000):
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a p-value that indicates whether the two distributions are the same or not. 
    The test is performed for each dimension of the distributions and for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 10.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define ad_list that will contain the list of ad for all dimensions and all iterations
    ad_list=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # The ad test is computed and the p-value saved for each dimension
        for dim in range(ndims):
            p_val=anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]])[2]
            ad_list.append(p_val)
    # Compute the mean and std of the p-values
    ad_mean = np.mean(ad_list)
    ad_std = np.std(ad_list)
    # Return the mean and std of the p-values
    return [ad_mean,ad_std,ad_list]


def AD_test_2(dist_1,dist_2,n_iter=100,batch_size=100000):
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a p-value that indicates whether the two distributions are the same or not. 
    The test is performed for each dimension of the distributions and for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in np.ceil(np.sqrt(n_iter)) batches dist_1_j, dist_2_k of size batch_size=int(nsamples/np.ceil(np.sqrt(n_iter))) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_k.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    n_iter=int(np.ceil(np.sqrt(n_iter)))
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define ad_list that will contain the list of ad for all dimensions and all iterations
    ad_list=[]
    # Loop over all iterations
    for j in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_j=dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        for k in range(n_iter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k=dist_2.sample(batch_size).numpy()
            else:   
                raise ValueError("dist_1 must be either a numpy array or a distribution")
            # The ad test is computed and the p-value saved for each dimension
            for dim in range(ndims):
                p_val=anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]])[2]
                ad_list.append(p_val)
    # Compute the mean and std of the p-values
    ad_mean = np.mean(ad_list)
    ad_std = np.std(ad_list)
    # Return the mean and std of the p-values
    return [ad_mean,ad_std,ad_list]


def FN_1(dist_1,dist_2,n_iter=10,batch_size=100000):
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The norm is computed for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 10.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define fn_list that will contain the list of fn for all dimensions and all iterations
    FN_list=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # The fn test is computed and the p-value saved for each dimension
        dist_1_cov = np.cov(dist_1_k,bias=True,rowvar=False)
        dist_1_corr=correlation_from_covariance(dist_1_cov)
        dist_2_cov = np.cov(dist_2_k,bias=True,rowvar=False)
        dist_2_corr=correlation_from_covariance(dist_2_cov)    
        matrix_sum=dist_1_corr-dist_2_corr
        frob_norm=np.linalg.norm(matrix_sum, ord='fro')
        FN_list.append(frob_norm)
    # Compute the mean and std of the p-values
    FN_mean = np.mean(FN_list)
    FN_std = np.std(FN_list)
    # Return the mean and std of the p-values
    return [FN_mean,FN_std, FN_list]


def FN_2(dist_1,dist_2,n_iter=100,batch_size=100000):
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The norm is computed for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in np.ceil(np.sqrt(n_iter)) batches dist_1_j, dist_2_k of size batch_size=int(nsamples/np.ceil(np.sqrt(n_iter))) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_k.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    n_iter=int(np.ceil(np.sqrt(n_iter)))
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define fn_list that will contain the list of fn for all dimensions and all iterations
    FN_list=[]
    # Loop over all iterations
    for j in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_j=dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        for k in range(n_iter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k=dist_2.sample(batch_size).numpy()
            else:   
                raise ValueError("dist_1 must be either a numpy array or a distribution")
            # The fn test is computed and the p-value saved for each dimension
            dist_1_cov = np.cov(dist_1_j,bias=True,rowvar=False)
            dist_1_corr=correlation_from_covariance(dist_1_cov)
            dist_2_cov = np.cov(dist_2_k,bias=True,rowvar=False)
            dist_2_corr=correlation_from_covariance(dist_2_cov)    
            matrix_sum=dist_1_corr-dist_2_corr
            frob_norm=np.linalg.norm(matrix_sum, ord='fro')
            FN_list.append(frob_norm)
    # Compute the mean and std of the p-values
    FN_mean = np.mean(FN_list)
    FN_std = np.std(FN_list)
    # Return the mean and std of the p-values
    return [FN_mean,FN_std, FN_list]


def WD_1(dist_1,dist_2,n_iter=10,batch_size=100000):
    """
    The Wasserstein distance between the target distribution and the distribution of the test data.
    The distance is computed for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 10.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.

    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define ad_list that will contain the list of wd for all dimensions and all iterations
    wd_list=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # The WD test is computed and saved for each dimension
        for dim in range(ndims):
            wd=wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim])
            wd_list.append(wd)
    #print(wd_list)
    # Compute the mean and std of the p-values
    wd_mean = np.mean(wd_list)
    wd_std = np.std(wd_list)
    # Return the mean and std of the p-values
    return [wd_mean,wd_std, wd_list]


def WD_2(dist_1,dist_2,n_iter=100,batch_size=100000):
    """
    The Wasserstein distance between the target distribution and the distribution of the test data.
    The distance is computed for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in np.ceil(np.sqrt(n_iter)) batches dist_1_j, dist_2_k of size batch_size=int(nsamples/np.ceil(np.sqrt(n_iter))) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_k.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    n_iter=int(np.ceil(np.sqrt(n_iter)))
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define ad_list that will contain the list of wd for all dimensions and all iterations
    wd_list=[]
    # Loop over all iterations
    for j in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_j=dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        for k in range(n_iter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k=dist_2.sample(batch_size).numpy()
            else:   
                raise ValueError("dist_1 must be either a numpy array or a distribution")
            # The WD test is computed and saved for each dimension
            for dim in range(ndims):
                wd=wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim])
                wd_list.append(wd)
    #print(wd_list)
    # Compute the mean and std of the p-values
    wd_mean = np.mean(wd_list)
    wd_std = np.std(wd_list)
    # Return the mean and std of the p-values
    return [wd_mean,wd_std, wd_list]


def SWD_1(dist_1,dist_2,n_iter=10,batch_size=100000,n_slices=100,seed=None):
    """
    Compute the sliced Wasserstein distance between two sets of points using n_slices random directions and the p-th Wasserstein distance.
    The distance is computed for n_iter times and for n_slices directions and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    if seed is None:
        np.random.seed(np.random.randint(1000000))
    else:
        np.random.seed(int(seed))
    if n_slices is None:
        n_slices = np.max([100,ndims])
    else:
        n_slices = int(n_slices)
    # Define ad_list that will contain the list of swd for all dimensions and all iterations
    swd_list=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # Generate random directions
        directions = np.random.randn(n_slices, ndims)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        # Compute sliced Wasserstein distance
        for direction in directions:
            dist_1_proj = dist_1_k @ direction
            dist_2_proj = dist_2_k @ direction
            swd_list.append(wasserstein_distance(dist_1_proj, dist_2_proj))
    # Compute the mean and std of the p-values
    swd_mean = np.mean(swd_list)
    swd_std = np.std(swd_list)
    return [swd_mean,swd_std,swd_list]


def SWD_2(dist_1,dist_2,n_iter=100,batch_size=100000,n_slices=100,seed=None):
    """
    Compute the sliced Wasserstein distance between two sets of points using n_slices random directions and the p-th Wasserstein distance.
    The distance is computed for n_iter times and for n_slices directions and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in np.ceil(np.sqrt(n_iter)) batches dist_1_j, dist_2_k of size batch_size=int(nsamples/np.ceil(np.sqrt(n_iter))) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_k.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    n_iter=int(np.ceil(np.sqrt(n_iter)))
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    if seed is None:
        np.random.seed(np.random.randint(1000000))
    else:
        np.random.seed(int(seed))
    if n_slices is None:
        n_slices = np.max([100,ndims])
    else:
        n_slices = int(n_slices)
    # Define ad_list that will contain the list of swd for all dimensions and all iterations
    swd_list=[]
    # Loop over all iterations
    for j in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_j=dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        for k in range(n_iter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k=dist_2.sample(batch_size).numpy()
            else:   
                raise ValueError("dist_1 must be either a numpy array or a distribution")
            # Generate random directions
            directions = np.random.randn(n_slices, ndims)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            # Compute sliced Wasserstein distance
            for direction in directions:
                dist_1_proj = dist_1_j @ direction
                dist_2_proj = dist_2_k @ direction
                swd_list.append(wasserstein_distance(dist_1_proj, dist_2_proj))
    # Compute the mean and std of the p-values
    swd_mean = np.mean(swd_list)
    swd_std = np.std(swd_list)
    return [swd_mean,swd_std,swd_list]


def ComputeMetrics(dist_1,dist_2,n_iter=10,batch_size=100000,n_slices=100,seed=None):
    """
    Function that computes the metrics. The following metrics are implemented:
    
        - KL-divergence
        - Mean and median of 1D KS-test
        - Mean and median of 1D Anderson-Darling test
        - Mean and median of Wasserstein distance
        - Frobenius norm
    """
    [ks_mean, ks_std, ks_list] = KS_test_1(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    [ad_mean, ad_std, ad_list] = AD_test_1(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    [fn_mean, fn_std, fn_list] = FN_1(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    [wd_mean, wd_std, wd_list] = WD_1(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    [swd_mean,swd_std,swd_list] = SWD_1(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size, n_slices = n_slices, seed = seed)
    return ks_mean, ks_std, ks_list, ad_mean, ad_std, ad_list, wd_mean, wd_std, wd_list, swd_mean, swd_std, swd_list, fn_mean, fn_std, fn_list
