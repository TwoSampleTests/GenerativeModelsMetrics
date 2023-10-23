__all__ = ["multivariate_ecdf_np",
           "multivariate_ecdf_tf",
           "compute_two_sample_Dn_np",
           "compute_two_sample_Dn_tf",
           "multiks_2samp_np",
           "multiks_2samp_tf",
           "MultiKSTest"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from scipy.stats import ks_2samp # type: ignore
from .utils import reset_random_seeds
from .utils import conditional_print
from .utils import conditional_tf_print
from .utils import generate_and_clean_data
from .utils import NumpyDistribution
from .base import TwoSampleTestInputs
from .base import TwoSampleTestBase
from .base import TwoSampleTestResult
from .base import TwoSampleTestResults

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

def multivariate_ecdf_np(sample, theta):
    sample = np.array(sample)
    n, k = sample.shape
    count = np.sum(np.all(sample <= theta, axis=1))
    return count / n

def compute_two_sample_Dn_np(sample1, sample2, debug=False):
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    n, k1 = sample1.shape
    m, k2 = sample2.shape
    assert k1 == k2, "Both samples must have the same dimensionality"
    
    # Combine both samples to consider all possible thetas
    combined_sample = np.vstack([sample1, sample2])
    
    # Precompute eCDF values for all points in the combined sample
    ecdf1_values = np.array([multivariate_ecdf_np(sample1, theta) for theta in combined_sample])
    ecdf2_values = np.array([multivariate_ecdf_np(sample2, theta) for theta in combined_sample])
    
    # Compute Dn+ and Dn-
    Dn_plus = np.max(ecdf1_values - ecdf2_values)
    Dn_minus = np.max(ecdf2_values - ecdf1_values)
    if debug:
        print(f"Computed Dn+ = {Dn_plus}")
        print(f"Computed Dn- = {Dn_minus}")
    
    # Compute Dn
    Dn = max(Dn_plus, Dn_minus)
    #print(f"Computed Dn = {Dn}")
    
    return Dn
    
def multiks_2samp_np(sample1, sample2, scaled = False):
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    n1, k1 = sample1.shape
    n2, k2 = sample2.shape
    n = n1*n2/(n1+n2)
    assert k1 == k2, "Both samples must have the same dimensionality"
    #print(f"The samples have dimensionality k = {k1}")
    
    Dn = compute_two_sample_Dn_np(sample1, sample2)
    normDn = np.sqrt(n)*Dn
    #p_value = 2*k1*np.exp(-2*normDn**2)
    #print(f"Computed Dn = {Dn}")
    #print(f"Computed normalized $\\sqrt(n)Dn$ = {normDn}")
    #print(f"Computed p-value = {p_value}")
    if scaled:
        return normDn#, p_value
    else:
        return Dn#, p_value
    
@tf.function(jit_compile=True, reduce_retracing = True)
def multivariate_ecdf_tf(sample, theta):
    sample = tf.convert_to_tensor(sample)
    n, k = tf.shape(sample)[0], tf.shape(sample)[1]
    bool_tensor = tf.reduce_all(sample <= theta, axis=1)
    count = tf.reduce_sum(tf.cast(bool_tensor, tf.float32))
    return tf.cast(count, tf.float32) / tf.cast(n, tf.float32) # type: ignore

@tf.function(jit_compile=True, reduce_retracing = True)
def compute_two_sample_Dn_tf(sample1, sample2):
    sample1 = tf.convert_to_tensor(sample1)
    sample2 = tf.convert_to_tensor(sample2)
    n, k1 = tf.shape(sample1)[0], tf.shape(sample1)[1]
    m, k2 = tf.shape(sample2)[0], tf.shape(sample2)[1]
    tf.debugging.assert_equal(k1, k2, message="Both samples must have the same dimensionality")
    
    combined_sample = tf.concat([sample1, sample2], axis=0)
    
    #ecdf1_values = tf.map_fn(lambda theta: multivariate_ecdf_tf(sample1, theta), combined_sample, dtype=tf.float32)
    #ecdf2_values = tf.map_fn(lambda theta: multivariate_ecdf_tf(sample2, theta), combined_sample, dtype=tf.float32)
    
    ecdf1_values = tf.vectorized_map(lambda theta: multivariate_ecdf_tf(sample1, theta), combined_sample) # type: ignore
    ecdf2_values = tf.vectorized_map(lambda theta: multivariate_ecdf_tf(sample2, theta), combined_sample) # type: ignore
    
    return tf.reduce_max(tf.abs(ecdf1_values - ecdf2_values)) # type: ignore

@tf.function(jit_compile=True, reduce_retracing = True)
def multiks_2samp_tf(sample1, sample2, scaled=True):
    sample1 = tf.convert_to_tensor(sample1)
    sample2 = tf.convert_to_tensor(sample2)
    n1, k1 = tf.shape(sample1)[0], tf.shape(sample1)[1]
    n2, k2 = tf.shape(sample2)[0], tf.shape(sample2)[1]
    n = (n1 * n2) / (n1 + n2)
    
    tf.debugging.assert_equal(k1, k2, message="Both samples must have the same dimensionality")
    
    Dn = compute_two_sample_Dn_tf(sample1, sample2) # type: ignore
    normDn = tf.sqrt(tf.cast(n, tf.float32)) * Dn
    #p_value = 2 * tf.cast(k1, tf.float32) * tf.exp(-2 * normDn ** 2) # type: ignore
    
    if scaled:
        return normDn#, p_value
    else:
        return Dn#, p_value


class MultiKSTest(TwoSampleTestBase):
    """
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False
                ) -> None:
        """
        Class constructor.
        """
        # From base class
        self._Inputs: TwoSampleTestInputs
        self._progress_bar: bool
        self._verbose: bool
        self._start: float
        self._end: float
        self._pbar: tqdm
        self._Results: TwoSampleTestResults
    
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
        
    def compute(self, max_vectorize: int = 10) -> None:
        """
        Function that computes the multivariate Kolmogorov-Smirnov test-statistic and p-value for two samples.
        selecting among the Test_np and Test_tf methods depending on the use_tf attribute.
        
        Parameters:
        ----------
        max_vectorize: int, optional, default = 10
            Maximum number of samples that can be processed by the tensorflow backend.
            If None, the total number of samples is not checked.

        Returns:
        -------
        None
        """
        if self.use_tf:
            self.Test_tf(max_vectorize = max_vectorize)
        else:
            self.Test_np()
    
    def Test_np(self) -> None:
        """
        Function that computes the multivariate Kolmogorov-Smirnov test-statistic and p-value for two samples using numpy functions.
        The calculation is based on a custom function multiks_2samp_np.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
            
        Parameters:
        ----------
        None
            
        Returns:
        -------
        None        
        """
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: DataTypeNP = self.Inputs.dist_1_num
        else:
            dist_1_num = self.Inputs.dist_1_num.numpy()
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: DataTypeNP = self.Inputs.dist_2_num
        else:
            dist_2_num = self.Inputs.dist_2_num.numpy()
        dist_1_symb: DistType = self.Inputs.dist_1_symb
        dist_2_symb: DistType = self.Inputs.dist_2_symb
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = self.get_niter_batch_size_np() # type: ignore
        if isinstance(self.Inputs.dtype, tf.DType):
            dtype: Union[type, np.dtype] = self.Inputs.dtype.as_numpy_dtype
        else:
            dtype = self.Inputs.dtype
        seed: int = self.Inputs.seed
        dist_1_k: DataTypeNP
        dist_2_k: DataTypeNP
        
        # Utility functions
        def start_calculation() -> None:
            conditional_print(self.verbose, "\n------------------------------------------")
            conditional_print(self.verbose, "Starting MultiKS tests calculation...")
            conditional_print(self.verbose, "niter = {}" .format(niter))
            conditional_print(self.verbose, "batch_size = {}" .format(batch_size))
            self._start = timer()
            
        def init_progress_bar() -> None:
            nonlocal niter
            if self.progress_bar:
                self.pbar = tqdm(total = niter, desc="Iterations")

        def update_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.update(1)

        def close_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.close()

        def end_calculation() -> None:
            self._end = timer()
            conditional_print(self.verbose, "Two-sample test calculation completed in "+str(self.end-self.start)+" seconds.")
        
        multiks_list: List[List] = []

        start_calculation()
        init_progress_bar()
            
        reset_random_seeds(seed = seed)
        
        conditional_print(self.verbose, "Running numpy MultiKS tests...")
        for k in range(niter):
            if not np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            elif not np.shape(dist_1_num[0])[0] == 0 and np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
                dist_2_k = np.array(dist_2_symb.sample(batch_size)).astype(dtype) # type: ignore
            elif np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = np.array(dist_1_symb.sample(batch_size)).astype(dtype) # type: ignore
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            else:
                dist_1_k = np.array(dist_1_symb.sample(batch_size)).astype(dtype) # type: ignore
                dist_2_k = np.array(dist_2_symb.sample(batch_size)).astype(dtype) # type: ignore
            multiks = multiks_2samp_np(dist_1_k, dist_2_k)
            multiks_list.append(multiks)
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "MultiKS Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, DataTypeNP] = {"metric_list": np.array(multiks_list)}
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
        self.Results.append(result)
        
    def Test_tf(self) -> None:
        """
        Function that computes the multivariate Kolmogorov-Smirnov test-statistic and p-value for two samples using tensorflow functions.
        The calculation is based in the custom function multiks_2samp_tf.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The calculation is parallelized over max_vectorize (out of niter).
        The results are stored in the Results attribute.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_1_num)
        else:
            dist_1_num = self.Inputs.dist_1_num # type: ignore
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_2_num)
        else:
            dist_2_num = self.Inputs.dist_2_num # type: ignore
        if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
            dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
        else:
            raise ValueError("dist_1_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            raise ValueError("dist_2_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = [int(i) for i in self.get_niter_batch_size_tf()] # type: ignore
        dtype: tf.DType = tf.as_dtype(self.Inputs.dtype)
        seed: int = self.Inputs.seed
        
        # Utility functions
        def start_calculation() -> None:
            conditional_tf_print(self.verbose, "\n------------------------------------------")
            conditional_tf_print(self.verbose, "Starting MultiKS tests calculation...")
            conditional_tf_print(self.verbose, "Running TF MultiKS tests...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "MultiKS tests calculation completed in", str(elapsed), "seconds.")
            
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                   seed: int = 0
                                  ) -> tf.Tensor:
            nonlocal dtype
            #dist_num: tf.Tensor = tf.cast(dist.sample(nsamples, seed = int(seed)), dtype = dtype) # type: ignore
            dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, 100, dtype = self.Inputs.dtype, seed = int(seed), mirror_strategy = self.Inputs.mirror_strategy) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num
        
        #@tf.function(jit_compile=True, reduce_retracing = True)
        #@tf.function(reduce_retracing=True)
        def compute_test() -> Tuple[tf.Tensor, tf.Tensor]:
            # Check if numerical distributions are empty and print a warning if so
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
            res = tf.TensorArray(dtype, size = niter)
            
            # Define unique constants for the two distributions. It is sufficient that these two are different to get different samples from the two distributions, if they are equal. 
            # There is not problem with subsequent calls to the batched_test function, since the random state is updated at each call.
            seed_dist_1  = int(1e6)  # Seed for distribution 1
            seed_dist_2  = int(1e12)  # Seed for distribution 2

            def body(i, res):
                
                # Define the loop body to vectorize over ndims*chunk_size
                dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size, seed = seed_dist_1),
                                               false_fn = lambda: return_dist_num(dist_1_num[i * batch_size: (i + 1) * batch_size, :])) # type: ignore
                dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size, seed = seed_dist_2),
                                               false_fn = lambda: return_dist_num(dist_2_num[i * batch_size: (i + 1) * batch_size, :])) # type: ignore

                multiks = multiks_2samp_tf(dist_1_k, dist_2_k, scaled = False) # type: ignore
                
                multiks = tf.cast(multiks, dtype=dtype)
                
                res = res.write(i, multiks)
                return i+1, res

            _, res = tf.while_loop(lambda i, res: i < niter, body, [0, res])
            
            return res.stack()
                
        start_calculation()
        
        reset_random_seeds(seed = seed)
        
        multiks_list = compute_test() # type: ignore
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "MultiKS Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, DataTypeNP] = {"metric_list": np.array(multiks_list)}
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
        self.Results.append(result)