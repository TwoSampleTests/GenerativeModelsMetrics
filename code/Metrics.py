#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated June 2023

@author: Riccardo Torre (riccardo.torre@ge.infn.it)
"""
from sklearn import datasets # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy import stats # type: ignore
from scipy.stats import ks_2samp # type: ignore
from scipy.stats import wasserstein_distance # type: ignore
from scipy.stats import epps_singleton_2samp # type: ignore
from scipy.stats import anderson_ksamp # type: ignore
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
    nsamples: Optional[int]
    ndims: int
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
              niter: int = 10,
              batch_size: int = 100000
             ) -> Tuple[List[float], List[float]]:
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ks_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions KS_test_1 and KS_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, KS_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while KS_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but KS_test_1 performs independent 
    tests with less points, while KS_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ks_test_statistic_mean and ks_pvalue_mean for each iteration.
    """    
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    metric: float
    pvalue: float
    metric_list: List[float] = []
    pvalue_list: List[float] = []
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter
    
    for k in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_k: np.ndarray = dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k = dist_1.sample(batch_size).numpy()
        if isinstance(dist_2, np.ndarray):
            dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k = dist_2.sample(batch_size).numpy()
        [metric, pvalue] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
        metric_list.append(metric)
        pvalue_list.append(pvalue)
    return metric_list, pvalue_list

def KS_test_2(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
              dist_2: Union[np.ndarray, tfp.distributions.Distribution],
              niter: int = 10,
              batch_size: int = 100000
             ) -> Tuple[List[float], List[float]]:
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ks_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions KS_test_1 and KS_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, KS_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while KS_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but KS_test_1 performs independent 
    tests with less points, while KS_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ks_test_statistic_mean and ks_pvalue_mean for each iteration.
    """
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    metric: float
    pvalue: float
    metric_list: List[float] = []
    pvalue_list: List[float] = []
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter
    
    for j in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_j: np.ndarray = dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j = dist_1.sample(batch_size).numpy()
        for k in range(niter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k = dist_2.sample(batch_size).numpy()
            [metric, pvalue] = np.mean([ks_2samp(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            metric_list.append(metric)
            pvalue_list.append(pvalue)
    metric_list = np.mean(np.array(metric_list), axis=1).tolist()
    return metric_list, pvalue_list

def AD_test_1(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
              dist_2: Union[np.ndarray, tfp.distributions.Distribution],
              niter: int = 10,
              batch_size: int = 100000
             ) -> Tuple[List[float], List[float]]:
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ad_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions AD_test_1 and AD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, AD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while AD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but AD_test_1 performs independent 
    tests with less points, while AD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ad_test_statistic_mean and ad_pvalue_mean for each iteration.
    """
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    metric: float
    pvalue: float
    metric_list: List[float] = []
    pvalue_list: List[float] = []
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter

    for k in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_k: np.ndarray = dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k = dist_1.sample(batch_size).numpy()
        if isinstance(dist_2, np.ndarray):
            dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k = dist_2.sample(batch_size).numpy()
        [metric, pvalue] = np.mean([anderson_ksamp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
        metric_list.append(metric)
        pvalue_list.append(pvalue)
    metric_list = np.mean(np.array(metric_list), axis=1).tolist()
    return metric_list, pvalue_list

def AD_test_2(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
              dist_2: Union[np.ndarray, tfp.distributions.Distribution],
              niter: int = 10,
              batch_size: int = 100000
             ) -> Tuple[List[float], List[float]]:
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ad_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions AD_test_1 and AD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, AD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while AD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but AD_test_1 performs independent 
    tests with less points, while AD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ad_test_statistic_mean and ad_pvalue_mean for each iteration.
    """
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    metric: float
    pvalue: float
    metric_list: List[float] = []
    pvalue_list: List[float] = []
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter
        
    for j in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_j: np.ndarray = dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j = dist_1.sample(batch_size).numpy()
        for k in range(niter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k = dist_2.sample(batch_size).numpy()
            [metric, pvalue] = np.mean([anderson_ksamp(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            metric_list.append(metric)
            pvalue_list.append(pvalue)
    metric_list = np.mean(np.array(metric_list), axis=1).tolist()
    return metric_list, pvalue_list

def FN_1(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
         dist_2: Union[np.ndarray, tfp.distributions.Distribution],
         niter: int = 10,
         batch_size: int = 100000
        ) -> List[float]:
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The value of this metric is computed niter times and a list of the fn_values is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The fn_values are then computed over all pairs of batches dist_1_j, dist_2_j and the list is returned.
    
    NOTE: The functions FN_1 and FN_2 differ in the way batches are computed and compared. For instanve, for 10 iterations with 100000 samples, 
    FN_1 generates 10 pairs of independent batches of 10000 samples each and performs the tests pairwise (10 tests with 10000 points), while FN_2 generates 3 pairs of independent batches of 100000/3
    samples each and performs the tests for all pairs (9 tests with 100000/3 points). In the case of FN_1 tests are done with less points, but are independent, while in the case of 
    FN_2 tests are done with more points, but are not independent.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        fn_values_list: lists of floats, list of fn_values for each iteration.
    """
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    value: float
    values_list: List[float] = []
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter
        
    for k in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_k: np.ndarray = dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k = dist_1.sample(batch_size).numpy()
        if isinstance(dist_2, np.ndarray):
            dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k = dist_2.sample(batch_size).numpy()
        dist_1_cov: np.ndarray = np.cov(dist_1_k, bias = True, rowvar = False)
        dist_1_corr: np.ndarray = correlation_from_covariance(dist_1_cov)
        dist_2_cov: np.ndarray = np.cov(dist_2_k, bias = True, rowvar = False)
        dist_2_corr: np.ndarray = correlation_from_covariance(dist_2_cov)    
        matrix_sum: np.ndarray = dist_1_corr - dist_2_corr
        value = float(np.linalg.norm(matrix_sum, ord = 'fro'))
        values_list.append(value)
    return values_list


def FN_2(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
         dist_2: Union[np.ndarray, tfp.distributions.Distribution],
         niter: int = 10,
         batch_size: int = 100000
        ) -> List[float]:
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The value of this metric is computed niter times and a list of the fn_values is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The fn_values are then computed over all pairs of batches dist_1_j, dist_2_j and the list is returned.
    
    NOTE: The functions FN_1 and FN_2 differ in the way batches are computed and compared. For instanve, for 10 iterations with 100000 samples, 
    FN_1 generates 10 pairs of independent batches of 10000 samples each and performs the tests pairwise (10 tests with 10000 points), while FN_2 generates 3 pairs of independent batches of 100000/3
    samples each and performs the tests for all pairs (9 tests with 100000/3 points). In the case of FN_1 tests are done with less points, but are independent, while in the case of 
    FN_2 tests are done with more points, but are not independent.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        fn_values_list: lists of floats, list of fn_values for each iteration.
    """
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    value: float
    values_list: List[float] = []
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter
    
    for j in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_j: np.ndarray = dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j = dist_1.sample(batch_size).numpy()
        for k in range(niter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k=dist_2.sample(batch_size).numpy()
            dist_1_cov: np.ndarray = np.cov(dist_1_j, bias = True, rowvar = False)
            dist_1_corr: np.ndarray = correlation_from_covariance(dist_1_cov)
            dist_2_cov: np.ndarray = np.cov(dist_2_k, bias = True, rowvar = False)
            dist_2_corr: np.ndarray = correlation_from_covariance(dist_2_cov)    
            matrix_sum: np.ndarray = dist_1_corr - dist_2_corr
            value = float(np.linalg.norm(matrix_sum, ord = 'fro'))
            values_list.append(value)
    return values_list

def WD_1(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
         dist_2: Union[np.ndarray, tfp.distributions.Distribution],
         niter: int = 10,
         batch_size: int = 100000
        ) -> List[float]:
    """
    The Wasserstein distance between two 1D distributions.
    The value of the distance is computed for each dimension of the two distributions and an average is returned. This average is used as a test statistic whose
    distribution is estimated by computing the quantity niter times. The niter list of test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The wd_mean_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions AD_test_1 and AD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, AD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while AD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but AD_test_1 performs independent 
    tests with less points, while AD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        wd_mean_list: lists of float, list of wd_mean values for each iteration.
    """
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    value: float
    values_list: List[float] = []
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter
        
    for k in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_k: np.ndarray = dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k = dist_1.sample(batch_size).numpy()
        if isinstance(dist_2, np.ndarray):
            dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k = dist_2.sample(batch_size).numpy()
        value = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
        values_list.append(value)
    return values_list
        
def WD_2(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
         dist_2: Union[np.ndarray, tfp.distributions.Distribution],
         niter: int = 10,
         batch_size: int = 100000
        ) -> List[float]:
    """
    The Wasserstein distance between two 1D distributions.
    The value of the distance is computed for each dimension of the two distributions and an average is returned. This average is used as a test statistic whose
    distribution is estimated by computing the quantity niter times. The niter list of test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The wd_mean_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions AD_test_1 and AD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, AD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while AD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but AD_test_1 performs independent 
    tests with less points, while AD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        wd_mean_list: lists of float, list of wd_mean values for each iteration.
    """
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    value: float
    values_list: List[float] = []
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter
        
    for j in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_j: np.ndarray = dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j = dist_1.sample(batch_size).numpy()
        for k in range(niter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k = dist_2.sample(batch_size).numpy()
            value = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            values_list.append(value)
    return values_list

def SWD_1(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
          dist_2: Union[np.ndarray, tfp.distributions.Distribution],
          niter: int = 10,
          batch_size: int = 100000,
          nslices: int = 100,
          seed: Optional[int] = None
         ) -> List[float]:
    """
    Compute the sliced Wasserstein distance between two multi-dimensional distribution as the average over slices of the value of the 1D Wasserstein distances 
    between sets of points obtained projecting data along nslices random directions.
    The SWD value is used as a test statistic whose distribution is estimated by computing the quantity niter times. The niter list of SWD test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The swd_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions SWD_test_1 and SWD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, SWD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while SWD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but SWD_test_1 performs independent 
    tests with less points, while SWD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        nslices: int, number of random directions to be used. Defaults to 100.
        seed: int, seed to be used for random number generation. Defaults to None.
        
    Returns:
        swd_list: lists of float, list of swd values for each iteration.
    """
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    value: float
    values_list: List[float] = []
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter

    for k in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_k: np.ndarray = dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_k = dist_1.sample(batch_size).numpy()
        if isinstance(dist_2, np.ndarray):
            dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfp.distributions.Distribution):
            dist_2_k = dist_2.sample(batch_size).numpy()
        directions = np.random.randn(nslices, ndims)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        value = np.mean([wasserstein_distance(dist_1_k @ direction, dist_2_k @ direction) for direction in directions])
        values_list.append(value)
    return values_list

def SWD_2(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
          dist_2: Union[np.ndarray, tfp.distributions.Distribution],
          niter: int = 10,
          batch_size: int = 100000,
          nslices: int = 100,
          seed: Optional[int] = None
         ) -> List[float]:
    """
    Compute the sliced Wasserstein distance between two sets of points using nslices random directions and the p-th Wasserstein distance.
    The distance is computed for niter times and for nslices directions and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in np.ceil(np.sqrt(niter)) batches dist_1_j, dist_2_k of size batch_size=int(nsamples/np.ceil(np.sqrt(niter))) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_k.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        niter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/niter
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    value: float
    values_list: List[float] = []
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    ndims_1, nsamples_1 = input_dist_dimensions(dist_1)
    ndims_2, nsamples_2 = input_dist_dimensions(dist_2)
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
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
        batch_size = nsamples // niter
        
    for j in range(niter):
        if isinstance(dist_1, np.ndarray):
            dist_1_j: np.ndarray = dist_1[j*batch_size:(j+1)*batch_size,:]
        elif isinstance(dist_1, tfp.distributions.Distribution):
            dist_1_j = dist_1.sample(batch_size).numpy()
        for k in range(niter):
            if isinstance(dist_2, np.ndarray):
                dist_2_k: np.ndarray = dist_2[k*batch_size:(k+1)*batch_size,:]
            elif isinstance(dist_2, tfp.distributions.Distribution):
                dist_2_k = dist_2.sample(batch_size).numpy()
                # Generate random directions
            directions = np.random.randn(nslices, ndims)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            value = np.mean([wasserstein_distance(dist_1_j @ direction, dist_2_k @ direction) for direction in directions])
            values_list.append(value)
    return values_list

def ComputeMetrics(dist_1: Union[np.ndarray, tfp.distributions.Distribution],
                   dist_2: Union[np.ndarray, tfp.distributions.Distribution],
                   niter: int = 10,
                   batch_size: int = 100000,
                   nslices: int = 100,
                   seed: Optional[int] = None
                  ) -> Tuple[List[float],...]:
    """
    Function that computes the metrics. The following metrics are computed:
        - Mean of test statistics and p-values of 1D KS tests
        - Mean of test statistics and p-values of 1D AD tests
        - Values of FN metric
        - Mean of values of 1D WD
        - SWD values
    """
    ks_metric_list, ks_pvalue_list = KS_test_1(dist_1, dist_2, niter = niter, batch_size = batch_size)
    ad_metric_list, ad_pvalue_list = AD_test_1(dist_1, dist_2, niter = niter, batch_size = batch_size)
    fn_list = FN_1(dist_1, dist_2, niter = niter, batch_size = batch_size)
    wd_mean_list = WD_1(dist_1, dist_2, niter = niter, batch_size = batch_size)
    swd_list = SWD_1(dist_1, dist_2, niter = niter, batch_size = batch_size, nslices = nslices, seed = seed)
    return ks_metric_list, ks_pvalue_list, ad_metric_list, ad_pvalue_list, fn_list, wd_mean_list, swd_list
