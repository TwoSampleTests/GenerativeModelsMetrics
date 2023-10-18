__all__ = ["sks_2samp_tf",
           "SKSTest"]

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
from .ks_metrics import ks_2samp_tf

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType


@tf.function(experimental_compile=True, reduce_retracing=True)
def sks_2samp_tf(data1: tf.Tensor, 
                 data2: tf.Tensor,
                 alternative: str = 'two-sided',
                 method: str = 'auto',
                 precision: int = 100,
                 nslices: int = 100
                ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Function that computes the sliced Kolmogorov-Smirnov distance between two samples using tensorflow functions.
    The sliced Kolmogorov-Smirnov distance is computed by projecting the samples onto random directions,
    computing the Kolmogorov-Smirnov distance between the projections, and
    then taking the mean and standard deviation of the Kolmogorov-Smirnov distances.
    The sliced Kolmogorov-Smirnov distances can be computed using either numpy or tensorflow.
    
    The tf.function decorator is used to speed up subsequent calls to this function and to avoid retracing.

    Parameters:
    -----------
    data1: tf.Tensor, optional, shape = (n1,)
        First sample. Sample sizes can be different.
        
    data2: tf.Tensor, optional, shape = (n2,)
        Second sample. Sample sizes can be different.
        
    alternative: str, optional, default = 'two-sided'
        Defines the alternative hypothesis.
        
        - 'two-sided': two-sided test.
        - 'less': one-sided test, that data1 is less than data2.
        - 'greater': one-sided test, that data1 is greater than data2.
        
    method: str, optional, default = 'auto'
        Defines the method used for the calculation.
        
        - 'auto': selects the best method automatically.
        - 'exact': uses the exact method.
        - 'asymptotic': uses the asymptotic method.
        
    precision: int, optional, default = 100
        Number of points to use to discretize the CDFs.

    nslices: int, optional, default = 100
        Number of random directions to use for the projection.

    Returns:
    --------
    result: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        Tuple containing the following elements:

        - sks_mean: tf.Tensor, shape = (1,)
            Mean of the sliced Kolmogorov-Smirnov distances.
            
        - sks_std: tf.Tensor, shape = (1,)
            Standard deviation of the sliced Kolmogorov-Smirnov distances.
            
        - sks_proj: tf.Tensor, shape = (nslices,)
            Sliced Kolmogorov-Smirnov distances.
    """
    @tf.function(experimental_compile=True, reduce_retracing = True)
    def simplified_ks_2samp_tf(a, b):
        return ks_2samp_tf(a, b, alternative = alternative, method = method, precision = precision)[0] # type: ignore
        #return ks_2samp(a, b, alternative='two-sided', mode='asymp')[0]
    
    # Your existing code
    # Compute ndims
    ndims = tf.shape(data1)[1]
    
    # Generate random directions
    directions = tf.random.normal((nslices, ndims))
    directions /= tf.norm(directions, axis=1)[:, None]
    directions = tf.cast(directions, dtype=data1.dtype)
    
    # Compute projections for all directions at once
    data1_proj = tf.tensordot(data1, directions, axes=[[1],[1]])
    data2_proj = tf.tensordot(data2, directions, axes=[[1],[1]])
        
    # Transpose the projection tensor to have slices on the first axis
    data1_proj = tf.transpose(data1_proj)
    data2_proj = tf.transpose(data2_proj)
    
    # Apply the ks_2samp function to each slice using tf.vectorized_map
    sks_proj = tf.vectorized_map(lambda args: simplified_ks_2samp_tf(*args), (data1_proj, data2_proj)) # type: ignore
    
    # Computing correlation matrix of the projections
    # corr_mat = tfp.stats.correlation(directions)
    
    # Compute mean and std
    sks_mean = tf.reduce_mean(sks_proj)
    sks_std = tf.math.reduce_std(sks_proj)

    return sks_mean, sks_std, sks_proj # type: ignore


class SKSTest(TwoSampleTestBase):
    """
    Class for computing the sliced Kolmogorov-Smirnov distance between two samples.
    It inherits from the TwoSampleTestBase class.
    The SKS test is computed by projecting the samples onto random directions,
    computing the Kolmogorov-Smirnov distance between the projections, and
    then taking the mean and standard deviation of the Kolmogorov-Smirnov distances.
    The sliced Kolmogorov-Smirnov distances can be computed using either numpy or tensorflow.
    The scipy implementation is used for the numpy backend.
    A custom tensorflow implementation is used for the tensorflow backend.
    The tensorflow implementation is faster than the scipy implementation, especially for large sample sizes,
    number of projections, and number of iterations.
    
    Parameters:
    ----------
    data_input: TwoSampleTestInputs
        Object containing the inputs for the two-sample test.

    progress_bar: bool, optional, default = False
        If True, display a progress bar. The progress bar is automatically disabled when running tensorflow functions.
        
    verbose: bool, optional, default = False
        If True, print additional information.

    Attributes:
    ----------
    Inputs: TwoSampleTestInputs object
        Object containing the inputs for the two-sample test.

    Results: TwoSampleTestResults object
        Object containing the results of the two-sample test.

    start: float
        Time when the two-sample test calculation started.

    end: float
        Time when the two-sample test calculation ended.

    pbar: tqdm
        Progress bar object.
        
    Methods:
    -------
    compute(nslices: int = 100) -> None
        Function that computes the sliced Kolmogorov-Smirnov distance between the two samples
        selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.
        
    Test_np(nslices: int = 100) -> None
        Function that computes the sliced Kolmogorov-Smirnov distance between the two samples using numpy functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
    Test_tf(nslices: int = 100) -> None
        Function that computes the sliced Kolmogorov-Smirnov distance between the two samples using tensorflow functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
    Examples:
    --------
    
    .. code-block:: python
    
        import numpy as np
        import tensorflow as tf
        import tensorflow_probability as tfp
        import GenerativeModelsMetrics as GMetrics

        # Set random seed
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Define inputs
        nsamples = 1000000
        ndims = 2
        dtype = tf.float32
        ndims = 100
        eps = 0.1
        dist_1_symb = tfp.distributions.Normal(loc=np.full(ndims,0.), scale=np.full(ndims,1.))
        dist_2_symb = tfp.distributions.Normal(loc=np.random.uniform(-eps, eps, ndims), scale=np.random.uniform(1-eps, 1+eps, ndims))
        dist_1_num = tf.cast(dist_1_symb.sample(nsamples),tf.float32)
        dist_2_num = tf.cast(dist_2_symb.sample(nsamples),tf.float32)
        data_input = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1_num,
                                                  dist_2_input = dist_2_num,
                                                  niter = 100,
                                                  batch_size = 10000,
                                                  dtype_input = tf.float64,
                                                  seed_input = 0,
                                                  use_tf = True,
                                                  verbose = True)

        # Compute SKS metric
        sks_test = GMetrics.SKSTest(data_input = data_input, 
                                    progress_bar = True, 
                                    verbose = True)
        sks_test.compute(nslices = 100)
        sks_test.Results[0].result_value
        
        >> {'metric_lists': [[...]]
            'metric_means': [...],
            'metric_stds': [...]}
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False
                ) -> None:
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
        
    def compute(self, 
                nslices: int = 100
               ) -> None:
        """
        Function that computes the sliced Kolmogorov-Smirnov distance between the two samples
        selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.

        Parameters:
        -----------
        nslices: int, optional, default = 100
            Number of random directions to use for the projection.

        Returns:
        --------
        None

        """
        if self.use_tf:
            self.Test_tf(nslices = nslices)
        else:
            self.Test_np(nslices = nslices)
    
    def Test_np(self,
                nslices: int = 100
               ) -> None:
        """
        Function that computes the sliced Kolmogorov-Smirnov distance between the two samples using numpy functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.

        Parameters:
        -----------
        nslices: int, optional, default = 100
            Number of random directions to use for the projection.

        Returns:
        --------
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
            conditional_print(self.verbose, "Starting SKS metric calculation...")
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
        
        metric_lists: List[List] = []
        metric_means: List[float] = []
        metric_stds: List[float] = []

        start_calculation()
        init_progress_bar()
            
        reset_random_seeds(seed = seed)
        
        conditional_print(self.verbose, "Running numpy SKS calculation...")
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
            # Generate random directions
            directions = np.random.randn(nslices, ndims)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            # Compute sliced KS distance
            list1 = []
            for direction in directions:
                dist_1_proj = dist_1_k @ direction
                dist_2_proj = dist_2_k @ direction
                list1.append(ks_2samp(dist_1_proj, dist_2_proj)[0])
            metric_lists.append(list1)
            metric_means.append(np.mean(list1)) # type: ignore
            metric_stds.append(np.std(list1)) # type: ignore
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "SKS Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, DataTypeTF] = {"metric_lists": np.array(metric_lists), # type: ignore
                                               "metric_means": np.array(metric_means), # type: ignore
                                               "metric_stds": np.array(metric_stds)} # type: ignore
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
        self.Results.append(result)
        
    def Test_tf(self,
                nslices: int = 100
               ) -> None:
        """
        Function that computes the sliced Kolmogorov-Smirnov distance between the two samples using tensorflow functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
        Parameters:
        -----------
        nslices: int, optional, default = 100
            Number of random directions to use for the projection.

        Returns:
        --------
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
            conditional_tf_print(self.verbose, "Starting SKS metric calculation...")
            conditional_tf_print(self.verbose, "Running TF SKS calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "SKS metric calculation completed in", str(elapsed), "seconds.")
            
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
            
        #@tf.function(reduce_retracing=True)
        def compute_test() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            # Check if numerical distributions are empty and print a warning if so
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
            # Initialize the result TensorArray
            res = tf.TensorArray(dtype, size = niter)
            res_sks_mean = tf.TensorArray(dtype, size = niter)
            res_sks_std = tf.TensorArray(dtype, size = niter)
            res_sks_proj = tf.TensorArray(dtype, size = niter)
            
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
                
                sks_mean, sks_std, sks_proj = sks_2samp_tf(dist_1_k, dist_2_k, nslices = nslices) # type: ignore
                sks_mean = tf.cast(sks_mean, dtype=dtype)
                sks_std = tf.cast(sks_std, dtype=dtype)
                sks_proj = tf.cast(sks_proj, dtype=dtype)
        
                # Here we add an extra dimension to `sks_mean` and `sks_std` tensors to match rank with `sks_proj`
                sks_mean = tf.expand_dims(sks_mean, axis=0)
                sks_std = tf.expand_dims(sks_std, axis=0)
        
                result_value = tf.concat([sks_mean, sks_std, sks_proj], axis=0)
        
                res = res.write(i, result_value)
                return i+1, res
    
            _, res = tf.while_loop(lambda i, _: i < niter, body, [0, res])
            
            for i in range(niter):
                res_i = res.read(i)
                res_sks_mean = res_sks_mean.write(i, res_i[0])
                res_sks_std = res_sks_std.write(i, res_i[1])
                res_sks_proj = res_sks_proj.write(i, res_i[2:])
            
            sks_means = res_sks_mean.stack()
            sks_stds = res_sks_std.stack()
            sks_lists = res_sks_proj.stack()
                            
            return sks_means, sks_stds, sks_lists # type: ignore
                
        start_calculation()
        
        reset_random_seeds(seed = seed)
        
        sks_means, sks_stds, sks_lists = compute_test() # type: ignore
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "SKS Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, Any] = {"metric_lists": sks_lists.numpy(),
                                        "metric_means": sks_means.numpy(),
                                        "metric_stds": sks_stds.numpy()}
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)