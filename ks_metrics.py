__all__ = ["KSTest"]

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
from .utils import NumpyDistribution
from .utils import ks_2samp_tf
from .base import TwoSampleTestInputs
from .base import TwoSampleTestBase
from .base import TwoSampleTestResult
from .base import TwoSampleTestResults

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

class KSTest(TwoSampleTestBase):
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 method: str = "paired", #other possibility is "crossed"
                 progress_bar: bool = False,
                 verbose: bool = False
                ) -> None:
        # From base class
        self._Inputs: TwoSampleTestInputs
        self._method: str
        self._progress_bar: bool
        self._verbose: bool
        self._start: float
        self._end: float
        self._pbar: tqdm
        self._Results: TwoSampleTestResults
    
        super().__init__(data_input = data_input, 
                         method = method,
                         progress_bar = progress_bar,
                         verbose = verbose)
        
    def compute_KS(self):
        if self.use_tf:
            self.Test_tf()
        else:
            self.Test_np()
    
    def Test_np(self):
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: DataTypeNP = self.Inputs.dist_1_num
        else:
            dist_1_num = self.Inputs.dist_1_num.numpy()
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: DataTypeNP = self.Inputs.dist_2_num
        else:
            dist_2_num = self.Inputs.dist_2_num.numpy()
        is_symb_1: bool = self.Inputs.is_symb_1
        is_symb_2: bool = self.Inputs.is_symb_2
        dist_1_symb: DistType = self.Inputs.dist_1_symb
        dist_2_symb: DistType = self.Inputs.dist_2_symb
        ndims: int = self.Inputs.ndims
        method: str = self.method
        niter: int
        batch_size: int
        niter, batch_size = self.get_niter_batch_size_np(method = method) # type: ignore
        if isinstance(self.Inputs.dtype, tf.DType):
            dtype: Union[type, np.dtype] = self.Inputs.dtype.as_numpy_dtype
        else:
            dtype = self.Inputs.dtype
        seed: int = self.Inputs.seed
        dist_1_k: DataTypeNP
        dist_2_k: DataTypeNP
        dist_1_j: DataTypeNP
        
        # Utility functions
        def start_calculation() -> None:
            conditional_print(self.verbose, "Starting two-sample test calculation...")
            if method == "paired":
                conditional_print(self.verbose, "Running KS test in the 'paired' mode.")
            elif method == "crossed":
                conditional_print(self.verbose, "Running KS test in the 'crossed' mode.")
            else:
                raise ValueError("Invalid method.")    
            if method == "paired":
                conditional_print(self.verbose, "niter = {}" .format(niter))
            else:
                conditional_print(self.verbose, "niter = {}" .format(niter**2))
            conditional_print(self.verbose, "batch_size = {}" .format(batch_size))
            self._start = timer()
            
        def init_progress_bar() -> None:
            nonlocal niter
            total = niter**2 if method == "crossed" else niter
            if self.progress_bar:
                self.pbar = tqdm(total = total, desc="Iterations")

        def update_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.update(1)

        def close_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.close()

        def end_calculation() -> None:
            self._end = timer()
            conditional_print(self.verbose, "Two-sample test calculation completed in "+str(self.end-self.start)+" seconds.")
        
        if method == "paired":
            result_size: int = niter
        elif method == "crossed":
            result_size: int = niter**2
        else:
            raise ValueError("Invalid method.")
        metric_list: DataTypeNP = np.zeros(result_size).astype(dtype)
        pvalue_list: DataTypeNP = np.zeros(result_size).astype(dtype)

        start_calculation()
        init_progress_bar()
            
        reset_random_seeds(seed = seed)
        
        if method == "paired":
            conditional_print(self.verbose, "Running numpy paired")
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
                [metric_list[k], pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                update_progress_bar()
        elif method == "crossed":
            conditional_print(self.verbose, "Running numpy crossed")
            l: int = 0
            for j in range(niter):
                for k in range(niter):
                    if not np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                        dist_1_j = dist_1_num[j*batch_size:(j+1)*batch_size,:]
                        dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
                    elif not np.shape(dist_1_num[0])[0] == 0 and np.shape(dist_2_num[0])[0] == 0:
                        dist_1_j = dist_1_num[j*batch_size:(j+1)*batch_size,:]
                        dist_2_k = np.array(dist_2_symb.sample(batch_size)).astype(dtype) # type: ignore
                    elif np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                        dist_1_j = np.array(dist_1_symb.sample(batch_size)).astype(dtype) # type: ignore
                        dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
                    else:
                        dist_1_j = np.array(dist_1_symb.sample(batch_size)).astype(dtype) # type: ignore
                        dist_2_k = np.array(dist_2_symb.sample(batch_size)).astype(dtype) # type: ignore
                    [metric_list[l], pvalue_list[l]] = np.mean([ks_2samp(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                    l += 1
                    update_progress_bar()
        else:
            raise ValueError("method must be either 'paired' or 'crossed'")
        
        close_progress_bar()
        end_calculation()
        
        timestamp = datetime.now().isoformat()
        test_name = "KS Test_np"
        parameters = {**self.param_dict, **{"backend": "numpy"}}
        result_value = tf.stack([metric_list, pvalue_list], axis=1).numpy()
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = int(1e6)):
        max_vectorize = int(max_vectorize)
        # Set alias for inputs
        dist_1_num: tf.Tensor = self.Inputs.dist_1_num
        dist_2_num: tf.Tensor = self.Inputs.dist_2_num
        if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
            dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
        else:
            raise ValueError("dist_1_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            raise ValueError("dist_2_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        ndims: int = self.Inputs.ndims
        method: str = self.method
        niter: int
        batch_size: int
        niter, batch_size = [int(i) for i in self.get_niter_batch_size_tf(method = method)] # type: ignore
        dtype: tf.DType = tf.as_dtype(self.Inputs.dtype)
        seed: int = self.Inputs.seed
        
        # Utility functions
        def start_calculation() -> None:
            conditional_tf_print(self.verbose, "Starting KS tests calculation...")        
            if method == "paired":
                conditional_print(self.verbose, "Running KS test in the 'paired' mode.")
                conditional_print(self.verbose, "niter =", niter)
            elif method == "crossed":
                conditional_print(self.verbose, "Running KS test in the 'crossed' mode.")
                conditional_print(self.verbose, "niter =", niter**2)
            else:
                raise ValueError("Invalid method.")                    
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def create_progress_bar() -> None:
            nonlocal niter, method
            self.pbar = tf.cond(tf.equal(method, "paired"), true_fn = lambda: tqdm(total = int(niter), desc = "Iterations"), false_fn = lambda: tqdm(total = int(niter**2), desc = "Iterations")) # type: ignore

        def init_progress_bar() -> None:
            tf.cond(tf.equal(self.progress_bar, True), true_fn = lambda: create_progress_bar(), false_fn = lambda: None)

        def update_progress_bar() -> None:
            self.pbar.update(1)
            return None
            #tf.cond(tf.equal(self.progress_bar, True), true_fn = lambda: , false_fn = lambda: None)
            #if self.pbar:
            #    self.pbar.update(1)

        def close_progress_bar() -> None:
            tf.cond(tf.equal(self.progress_bar, True), true_fn = lambda: self.pbar.close(), false_fn = lambda: None)
            #if self.pbar:
            #    self.pbar.close()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "KS tests calculation completed in", str(elapsed), "seconds.")
            
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                   seed: int = 0
                                  ) -> tf.Tensor:
            nonlocal dtype
            dist_num: tf.Tensor = tf.cast(dist.sample(nsamples, seed = int(seed)), dtype = dtype) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num
            
        result_size = tf.cond(tf.equal(method, "paired"), true_fn = lambda: niter, false_fn = lambda: niter**2)
        l: int = 0
        
        @tf.function(reduce_retracing=True)
        def batched_compute_paired(start, end):
            conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".")
            # Define unique constants for the two distributions. It is sufficient that these two are different to get different samples from the two distributions, if they are equal. 
            # There is not problem with subsequent calls to the batch_compute_paired function, since the random state is updated at each call.
            seed_dist_1  = int(1e6)  # Seed for distribution 1
            seed_dist_2  = int(1e12)  # Seed for distribution 2
            
            # Define batched distributions
            dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size*(end-start), seed = seed_dist_1),
                                               false_fn = lambda: return_dist_num(dist_1_num[start*batch_size:end*batch_size, :])) # type: ignore
            dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size*(end-start), seed = seed_dist_2),
                                               false_fn = lambda: return_dist_num(dist_2_num[start*batch_size:end*batch_size, :])) # type: ignore

            dist_1_k = tf.reshape(dist_1_k, (batch_size, ndims*(end-start)))
            dist_2_k = tf.reshape(dist_2_k, (batch_size, ndims*(end-start)))

            #tf.print("dist_1_k.shape = ", tf.shape(dist_1_k))
            #tf.print("dist_2_k.shape = ", tf.shape(dist_2_k))
            #tf.print("dist_1_k[0,1] = ", dist_1_k[0,1])
            #tf.print("dist_2_k[0,1] = ", dist_2_k[0,1])
            
            # Define the loop body to vectorize over ndims*chunk_size
            def loop_body_vmap(idx):
                ks_result, ks_pvalue, _ = ks_2samp_tf(dist_1_k[:, idx], dist_2_k[:, idx], verbose=False) # type: ignore
                ks_result = tf.cast(ks_result, dtype=dtype)
                ks_pvalue = tf.cast(ks_pvalue, dtype=dtype)
                return ks_result, ks_pvalue

            # Vectorize over ndims*chunk_size
            ks_tests, ks_pvalues = tf.vectorized_map(loop_body_vmap, tf.range(ndims*(end-start))) # type: ignore

            # Reshape the results back to (chunk_size, ndims)
            ks_tests = tf.reshape(ks_tests, (end-start, ndims))
            ks_pvalues = tf.reshape(ks_pvalues, (end-start, ndims))     

            # Free up memory
            #del(dist_1_k, dist_2_k)
            #gc.collect()

            # Compute the mean values
            mean_ks_tests_batch = tf.cast(tf.reduce_mean(ks_tests, axis=1), dtype=dtype)
            mean_ks_pvalues_batch = tf.cast(tf.reduce_mean(ks_pvalues, axis=1), dtype=dtype)        

            # Stack and return the result
            result_value = tf.stack([mean_ks_tests_batch, mean_ks_pvalues_batch], axis=1)
            return result_value

        @tf.function(reduce_retracing=True)
        def compute_paired(max_vectorize: int = int(1e6)) -> tf.Tensor:
            conditional_tf_print(self.verbose, "Running compute_paired")
            # Check if numerical distributions are empty and print a warning if so
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
            # Ensure that max_vectorize is an integer larger than ndims
            max_vectorize = int(tf.cast(tf.maximum(max_vectorize, ndims),tf.int32)) # type: ignore

            # Compute the maximum number of iterations per chunk
            max_iter_per_chunk: int = int(tf.cast(tf.math.floor(max_vectorize / ndims), tf.int32)) # type: ignore

            # Compute the number of chunks
            nchunks: int = int(tf.cast(tf.math.ceil(niter / max_iter_per_chunk), tf.int32)) # type: ignore
            conditional_tf_print(tf.logical_and(self.verbose,tf.logical_not(tf.equal(nchunks,1))), "nchunks =", nchunks)
            
            # Run the computation in chunks
            # Initialize the result TensorArray
            res = tf.TensorArray(dtype, size=nchunks)

            def body(i, res):
                start = i * max_iter_per_chunk
                end = tf.minimum(start + max_iter_per_chunk, niter)
                chunk_result = batched_compute_paired(start, end) # type: ignore
                res = res.write(i, chunk_result)
                return i+1, res

            _, res = tf.while_loop(lambda i, res: i < nchunks, body, [0, res])
            res_stacked = res.concat()
            return res_stacked
        
        @tf.function(reduce_retracing=True)
        def compute_crossed() -> tf.Tensor:
            result_value: DataTypeTF
            res = tf.TensorArray(dtype, size=result_size)
            
            def body_large_crossed(j: int,
                                   k: int, 
                                   res: tf.TensorArray
                                  ) -> Tuple[int, int, tf.TensorArray]:
                nonlocal l
                dist_1_j: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size),
                                               false_fn = lambda: return_dist_num(dist_1_num[j * batch_size:(j + 1) * batch_size, :])) # type: ignore
                dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_2_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size),
                                               false_fn = lambda: return_dist_num(dist_2_num[k * batch_size:(k + 1) * batch_size, :])) # type: ignore
                ks_tests: tf.TensorArray = tf.TensorArray(dtype, size=ndims)
                ks_pvalues: tf.TensorArray = tf.TensorArray(dtype, size=ndims)
                def loop_body(dim: int, 
                              ks_tests: tf.TensorArray, 
                              ks_pvalues: tf.TensorArray
                             ) -> Tuple[int, tf.TensorArray, tf.TensorArray]:
                    ks_result, ks_pvalue, _ = ks_2samp_tf(dist_1_j[:, dim], dist_2_k[:, dim], verbose = False) # type: ignore
                    ks_result: tf.Tensor = tf.cast(ks_result, dtype = dtype) # type: ignore
                    ks_pvalue: tf.Tensor = tf.cast(ks_pvalue, dtype = dtype) # type: ignore
                    ks_tests = ks_tests.write(dim, ks_result)
                    ks_pvalues = ks_pvalues.write(dim, ks_pvalue)
                    return dim + 1, ks_tests, ks_pvalues
                _, ks_tests, ks_pvalues = tf.while_loop(lambda dim, ks_tests, ks_pvalues: dim < ndims,
                                                        loop_body,
                                                        [0, ks_tests, ks_pvalues])
                mean_ks_tests_batch: tf.Tensor = tf.cast(tf.reduce_mean(ks_tests.stack()), dtype = dtype) # type: ignore
                mean_ks_pvalues_batch: tf.Tensor = tf.cast(tf.reduce_mean(ks_pvalues.stack()), dtype = dtype) # type: ignore
                res = res.write(l, tf.stack([mean_ks_tests_batch, mean_ks_pvalues_batch], axis=0))
                l += 1
                tf.cond(tf.equal(self.progress_bar, True), true_fn = lambda: update_progress_bar(), false_fn = lambda: None)
                return j, k + 1, res

            def outer_loop_body(j: int,
                                outer_res: tf.TensorArray
                               ) -> Tuple[int, tf.TensorArray]:
                k = 0
                _, _, result_value = tf.while_loop(lambda j, k, res: k < niter, body_large_crossed, [j, k, outer_res])
                return j + 1, result_value

            _, result_value = tf.while_loop(lambda j, outer_res: j < niter, outer_loop_body, [0, res])
            return result_value.stack()
                
        start_calculation()
        init_progress_bar()
        
        reset_random_seeds(seed = seed)
        
        if method == "paired":
            result_value: tf.Tensor = compute_paired(max_vectorize = max_vectorize) # type: ignore
        elif method == "crossed":
            result_value: tf.Tensor = compute_crossed() # type: ignore
        else:
            raise ValueError("Invalid method.")
                             
        close_progress_bar()
        end_calculation()
        
        timestamp = datetime.now().isoformat()
        test_name = "KS Test_tf"
        parameters = {**self.param_dict, **{"backend": "tensorflow"}}
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)
        
        
        #@tf.function(reduce_retracing=True)
        #def compute_paired_best() -> tf.Tensor:
        #    conditional_tf_print(self.verbose, "Running compute_small_paired_best")
        #
        #    # Reshape the distributions
        #    dist_1_k = tf.reshape(dist_1_num[:batch_size * niter, :], (batch_size, ndims*niter)) # type: ignore
        #    dist_2_k = tf.reshape(dist_2_num[:batch_size * niter, :], (batch_size, ndims*niter)) # type: ignore
        #
        #    def loop_body_vmap(idx):
        #        ks_result, ks_pvalue, _ = ks_2samp_tf(dist_1_k[:, idx], dist_2_k[:, idx], verbose = False) # type: ignore
        #        ks_result: tf.Tensor = tf.cast(ks_result, dtype = dtype) # type: ignore
        #        ks_pvalue: tf.Tensor = tf.cast(ks_pvalue, dtype = dtype) # type: ignore
        #        return ks_result, ks_pvalue
        #
        #    # Vectorize over ndims*niter
        #    ks_tests, ks_pvalues = tf.vectorized_map(loop_body_vmap, tf.range(ndims*niter)) # type: ignore
        #
        #    # Reshape the results back to (niter, ndims)
        #    ks_tests = tf.reshape(ks_tests, (niter, ndims))
        #    ks_pvalues = tf.reshape(ks_pvalues, (niter, ndims))
        #
        #    # Compute the mean values
        #    mean_ks_tests_batch: tf.Tensor = tf.cast(tf.reduce_mean(ks_tests, axis = 1), dtype = dtype) # type: ignore
        #    mean_ks_pvalues_batch: tf.Tensor = tf.cast(tf.reduce_mean(ks_pvalues, axis = 1), dtype = dtype) # type: ignore
        #
        #    # Stack and return the result
        #    result_value = tf.stack([mean_ks_tests_batch, mean_ks_pvalues_batch], axis=1)
        #    return result_value
        
        #@tf.function(reduce_retracing=True)
        #def compute_small_paired() -> tf.Tensor:
        #    result_value: DataTypeTF
        #    res = tf.TensorArray(dtype, size=result_size)
        #    
        #    def body_small_paired(k: int, 
        #                          res: tf.TensorArray
        #                         ) -> Tuple[int, tf.TensorArray]:
        #        dist_1_k: tf.Tensor = dist_1_num[k * batch_size:(k + 1) * batch_size, :]
        #        dist_2_k: tf.Tensor = dist_2_num[k * batch_size:(k + 1) * batch_size, :]
        #        ks_tests: tf.TensorArray = tf.TensorArray(dtype, size=ndims)
        #        ks_pvalues: tf.TensorArray = tf.TensorArray(dtype, size=ndims)
        #        def loop_body(dim: int, 
        #                      ks_tests: tf.TensorArray, 
        #                      ks_pvalues: tf.TensorArray
        #                     ) -> Tuple[int, tf.TensorArray, tf.TensorArray]:
        #            ks_result, ks_pvalue, _ = ks_2samp_tf(dist_1_k[:, dim], dist_2_k[:, dim], verbose = False)
        #            ks_result: tf.Tensor = tf.cast(ks_result, dtype = dtype) # type: ignore
        #            ks_pvalue: tf.Tensor = tf.cast(ks_pvalue, dtype = dtype) # type: ignore
        #            ks_tests = ks_tests.write(dim, ks_result)
        #            ks_pvalues = ks_pvalues.write(dim, ks_pvalue)
        #            return dim + 1, ks_tests, ks_pvalues
        #        _, ks_tests, ks_pvalues = tf.while_loop(lambda dim, ks_tests, ks_pvalues: dim < ndims,
        #                                                loop_body,
        #                                                [0, ks_tests, ks_pvalues])
        #        mean_ks_tests_batch: tf.Tensor = tf.cast(tf.reduce_mean(ks_tests.stack()), dtype = dtype) # type: ignore	
        #        mean_ks_pvalues_batch: tf.Tensor = tf.cast(tf.reduce_mean(ks_pvalues.stack()), dtype = dtype) # type: ignore
        #        #tf.print("iteration = ", k)
        #        #tf.print("ks_result = ", mean_ks_tests_batch)
        #        #tf.print("ks_pvalue = ", mean_ks_pvalues_batch)
        #        res = res.write(k, tf.stack([mean_ks_tests_batch, mean_ks_pvalues_batch], axis=0))
        #        tf.cond(tf.equal(self.progress_bar, True), true_fn = lambda: update_progress_bar(), false_fn = lambda: None)
        #        return k + 1, res
        #
        #    _, result_value = tf.while_loop(lambda k, res: k < niter, body_small_paired, [0, res])
        #    return result_value.stack()

        # Best so far 8.6 sec 100 dim 100 iter 10K samples
        #@tf.function(reduce_retracing=True)
        #def compute_small_paired_best() -> tf.Tensor:
        #    tf.print("Running compute_small_paired_best")
        #    result_value: DataTypeTF
        #    res = tf.TensorArray(dtype, size=result_size)
        #            
        #    def body_small_paired(k: int, res: tf.TensorArray) -> Tuple[int, tf.TensorArray]:
        #        dist_1_k: tf.Tensor = dist_1_num[k * batch_size:(k + 1) * batch_size, :]
        #        dist_2_k: tf.Tensor = dist_2_num[k * batch_size:(k + 1) * batch_size, :]
        #
        #        def loop_body_vmap(dim):
        #            tf.print(dist_1_k[:, dim][:10])
        #            tf.print(dist_2_k[:, dim][:10])
        #            ks_result, ks_pvalue, _ = ks_2samp_tf(dist_1_k[:, dim], dist_2_k[:, dim], verbose = False)
        #            ks_result: tf.Tensor = tf.cast(ks_result, dtype = dtype)
        #            ks_pvalue: tf.Tensor = tf.cast(ks_pvalue, dtype = dtype)
        #            return ks_result, ks_pvalue     
        #
        #        ks_tests, ks_pvalues = tf.vectorized_map(loop_body_vmap, tf.range(ndims))
        #        tf.print("k = ", k)
        #        tf.print("ks_tests = ", ks_tests)
        #        tf.print("ks_pvalues = ", ks_pvalues)
        #
        #        mean_ks_tests_batch: tf.Tensor = tf.cast(tf.reduce_mean(ks_tests), dtype = dtype) 
        #        mean_ks_pvalues_batch: tf.Tensor = tf.cast(tf.reduce_mean(ks_pvalues), dtype = dtype)       
        #
        #        res = res.write(k, tf.stack([mean_ks_tests_batch, mean_ks_pvalues_batch], axis=0))
        #        tf.cond(tf.equal(self.progress_bar, True), true_fn = lambda: update_progress_bar(), false_fn = lambda: None)
        #        return k + 1, res
        #        
        #    _, result_value = tf.while_loop(lambda k, res: k < niter, body_small_paired, [0, res])
        #    return result_value.stack()