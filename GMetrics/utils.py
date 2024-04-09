__all__ = [
    'reset_random_seeds',
    'NumpyDistribution',
    'get_best_dtype_np',
    'get_best_dtype_tf',
    'conditional_print',
    'conditional_tf_print',
    'parse_input_dist_np',
    'parse_input_dist_tf'
]
import os
import inspect
import numpy as np
import pandas as pd
import random
import json
from scipy.stats import moment # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from numpy import random as npr
from timeit import default_timer as timer

from typing import Tuple, Union, Optional, Type, Callable, Dict, List, Any
from numpy import typing as npt
# For future tensorflow typing support
#import tensorflow.python.types.core as tft
#IntTensor = tft.Tensor[tf.int32]
#FloatTensor = Union[tft.Tensor[tf.float32], tft.Tensor[tf.float64]]
#BoolTensor = Type[tft.Tensor[tf.bool]]
DTypeType = Union[tf.DType, np.dtype, type]
IntTensor = Type[tf.Tensor]
FloatTensor = Type[tf.Tensor]
BoolTypeTF = Type[tf.Tensor]
BoolTypeNP = np.bool_
IntType = Union[int, IntTensor]
DataTypeTF = FloatTensor
DataTypeNP = npt.NDArray[np.float_]
DataType = Union[DataTypeNP, DataTypeTF]
DistTypeTF = tfp.distributions.Distribution

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    npr.seed(seed)
    random.seed(seed)

class NumpyDistribution:
    """
    Wrapper class for numpy.random.Generator distributions.
    Example:

    .. code-block:: python
    
        dist = NumpyDistribution('normal', loc=0, scale=1)
        dist.sample(100)
        
    """
    def __init__(self, 
                 distribution: str = "standard_normal",
                 generator_input: np.random.Generator = np.random.default_rng(),
                 dtype: type = np.float32,
                 **kwargs):
        self.generator: np.random.Generator = generator_input
        self.distribution: str = distribution
        self.dtype: type = dtype
        self.params: dict = kwargs

        # Check if the distribution is a valid method of the generator
        if not self.is_valid_distribution():
            raise ValueError(f"{distribution} is not a valid distribution of numpy.random.Generator.")

    def is_valid_distribution(self) -> bool:
        """Check if the given distribution is a valid method of the generator."""
        return self.distribution in dir(self.generator)

    def sample(self, 
               n: int,
               seed: Optional[int] = None
               ) -> DataTypeNP:
        """Generate a sample from the distribution."""
        if seed:
            reset_random_seeds(seed = seed)
        method: Callable = getattr(self.generator, self.distribution)
        if inspect.isroutine(method):
            return method(size=n, **self.params).astype(self.dtype)
        else:
            raise TypeError(f"{self.distribution} is not a callable method of numpy.random.Generator.")

DistTypeNP = NumpyDistribution
DistType = Union[tfp.distributions.Distribution, DistTypeNP]
DataDistTypeNP = Union[DataTypeNP, DistTypeNP]
DataDistTypeTF = Union[DataTypeTF, tfp.distributions.Distribution]
DataDistType = Union[DataDistTypeNP, DataDistTypeTF]
BoolType = Union[bool, BoolTypeTF, BoolTypeNP]
  
def compute_lik_ratio_statistic(dist_ref: tfp.distributions.Distribution,
                                dist_alt: tfp.distributions.Distribution,
                                sample_ref: tf.Tensor,
                                sample_alt: tf.Tensor,
                                batch_size: int = 10_000
                               ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    start_global = timer()
    
    # Compute number of samples
    n_ref = len(sample_ref)
    n_alt = len(sample_alt)
    
    print(f"Computing likelihood ratio statistic with {n_ref} reference samples and {n_alt} alternative samples.")
    
    # Compute log probabilities without reshaping
    start = timer()
    logprob_ref_ref = tf.reshape(dist_ref.log_prob(sample_ref), [-1, batch_size])
    end = timer()
    print(f"Computed logprob of ref dist on ref samples in {end - start:.2f} s.")
    start = timer()
    logprob_ref_alt = tf.reshape(dist_ref.log_prob(sample_alt), [-1, batch_size])
    end = timer()
    print(f"Computed logprob of ref dist on alt samples in {end - start:.2f} s.")
    start = timer()
    logprob_alt_alt = tf.reshape(dist_alt.log_prob(sample_alt), [-1, batch_size])
    end = timer()
    print(f"Computed logprob of alt dist on alt samples in {end - start:.2f} s.")
    
    # Reshape the log probabilities
    logprob_ref_ref_reshaped = tf.reshape(logprob_ref_ref, [-1, batch_size])
    logprob_ref_alt_reshaped = tf.reshape(logprob_ref_alt, [-1, batch_size])
    logprob_alt_alt_reshaped = tf.reshape(logprob_alt_alt, [-1, batch_size])
    
    # Create masks for finite log probabilities
    finite_indices_ref_ref = tf.math.is_finite(logprob_ref_ref_reshaped)
    finite_indices_ref_alt = tf.math.is_finite(logprob_ref_alt_reshaped)
    finite_indices_alt_alt = tf.math.is_finite(logprob_alt_alt_reshaped)
    
    # Count the number of finite samples
    n_ref_finite = tf.reduce_sum(tf.cast(finite_indices_ref_ref, tf.int32))
    n_alt_finite = tf.reduce_sum(tf.cast(tf.math.logical_and(finite_indices_alt_alt, finite_indices_ref_alt), tf.int32))

    if n_ref_finite < n_ref:
        fraction = tf.cast(n_ref - n_ref_finite, tf.float32) / tf.cast(n_ref, tf.float32) # type: ignore
        print(f"Warning: Removed a fraction {fraction} of reference samples due to non-finite log probabilities.")
        
    if n_alt_finite < n_alt:
        fraction = tf.cast(n_alt - n_alt_finite, tf.float32) / tf.cast(n_alt, tf.float32) # type: ignore
        print(f"Warning: Removed a fraction {fraction} of alternative samples due to non-finite log probabilities.")
    
    # Combined finite indices
    combined_finite_indices = tf.math.logical_and(tf.math.logical_and(finite_indices_ref_ref, finite_indices_ref_alt), finite_indices_alt_alt)
    
    # Use masks to filter the reshaped log probabilities
    logprob_ref_ref_filtered = tf.where(combined_finite_indices, logprob_ref_ref_reshaped, 0.)
    logprob_ref_alt_filtered = tf.where(combined_finite_indices, logprob_ref_alt_reshaped, 0.)
    logprob_alt_alt_filtered = tf.where(combined_finite_indices, logprob_alt_alt_reshaped, 0.)
    
    ## Filter the log probabilities using the mask
    #logprob_ref_ref_filtered = tf.boolean_mask(logprob_ref_ref_reshaped, combined_finite_indices)
    #logprob_ref_alt_filtered = tf.boolean_mask(logprob_ref_alt_reshaped, combined_finite_indices)
    #logprob_alt_alt_filtered = tf.boolean_mask(logprob_alt_alt_reshaped, combined_finite_indices)
    
    # Compute log likelihoods
    logprob_ref_ref_sum = tf.reduce_sum(logprob_ref_ref_filtered, axis=1)
    logprob_ref_alt_sum = tf.reduce_sum(logprob_ref_alt_filtered, axis=1)
    logprob_alt_alt_sum = tf.reduce_sum(logprob_alt_alt_filtered, axis=1)
    lik_ref_dist = logprob_ref_ref_sum + logprob_ref_alt_sum
    lik_alt_dist = logprob_ref_ref_sum + logprob_alt_alt_sum
    
    # Compute likelihood ratio statistic
    lik_ratio = 2 * (lik_alt_dist - lik_ref_dist)
    print(f'lik_ratio = {lik_ratio}')
    
    # Casting to float32 before performing division
    n_ref_finite_float = tf.cast(n_ref_finite, tf.float32)
    n_alt_finite_float = tf.cast(n_alt_finite, tf.float32)  

    # Compute normalized likelihood ratio statistic
    n = 2 * n_ref_finite_float * n_alt_finite_float / (n_ref_finite_float + n_alt_finite_float) # type: ignore
    
    # Compute normalized likelihood ratio statistic
    lik_ratio_norm = lik_ratio / tf.sqrt(tf.cast(n, tf.float32))

    end_global = timer()
    
    print(f"Computed likelihood ratio statistic in {end_global - start_global:.2f} s.")
    
    return logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm

    
def get_best_dtype_np(dtype_1: Union[type, np.dtype],
                      dtype_2: Union[type, np.dtype]) -> Union[type, np.dtype]:
    dtype_1_precision = np.finfo(dtype_1).eps
    dtype_2_precision = np.finfo(dtype_2).eps
    if dtype_1_precision > dtype_2_precision:
        return dtype_1
    else:
        return dtype_2
    
def get_best_dtype_tf(dtype_1: tf.DType, 
                      dtype_2: tf.DType) -> tf.DType:
    dtype_1_precision: tf.Tensor = tf.abs(tf.as_dtype(dtype_1).min)
    dtype_2_precision: tf.Tensor = tf.abs(tf.as_dtype(dtype_2).min)

    dtype_out = tf.cond(tf.greater(dtype_1_precision, dtype_2_precision), 
                        lambda: dtype_1,
                        lambda: dtype_2)
    return tf.as_dtype(dtype_out)

def conditional_print(verbose: bool = False,
                      *args) -> None:
    if verbose:
        print(*args)

#@tf.function(reduce_retracing = True)
def conditional_tf_print(verbose: bool = False,
                         *args) -> None:
    tf.cond(tf.equal(verbose, True), lambda: tf.print(*args), lambda: verbose)

def parse_input_dist_np(dist_input: DataDistTypeNP,
                        verbose: bool = False
                       ) -> Tuple[bool, DistTypeNP, DataTypeNP, int, int]:
    dist_symb: DistTypeNP
    dist_num: DataTypeNP
    nsamples: int
    ndims: int
    is_symb: bool
    if verbose:
        print("Parsing input distribution...")
    if isinstance(dist_input, np.ndarray):
        if verbose:
            print("Input distribution is a numberic numpy array or tf.Tensor")
        if len(dist_input.shape) != 2:
            raise ValueError("Input must be a 2-dimensional numpy array or a tfp.distributions.Distribution object")
        else:
            dist_symb = NumpyDistribution()
            dist_num = dist_input
            nsamples, ndims = dist_num.shape
            is_symb = False
    elif isinstance(dist_input, NumpyDistribution):
        if verbose:
            print("Input distribution is a NumpyDistribution object.")
        dist_symb = dist_input
        dist_num = np.array([[]],dtype=dist_symb.dtype)
        nsamples, ndims = 0, dist_symb.sample(2).shape[1]
        is_symb = True
    else:
        raise TypeError("Input must be either a numpy array or a NumpyDistribution object.")
    return is_symb, dist_symb, dist_num, ndims, nsamples


def parse_input_dist_tf(dist_input: DataDistType,
                        verbose: bool = False
                       ) -> Tuple[BoolType, DistTypeTF, DataTypeTF, IntType, IntType]: # type: ignore
    
    def is_ndarray_or_tensor():
        return tf.reduce_any([isinstance(dist_input, np.ndarray), tf.is_tensor(dist_input)])
    
    def is_distribution():
        return tf.reduce_all([
            tf.logical_not(is_ndarray_or_tensor()),
            tf.reduce_any([isinstance(dist_input, tfp.distributions.Distribution)])
        ])

    def handle_distribution():
        conditional_tf_print(verbose, "Input distribution is a tfp.distributions.Distribution object.")
        dist_symb: tfp.distributions.Distribution = dist_input
        nsamples, ndims = tf.constant(0), tf.shape(dist_symb.sample(2))[1]
        dist_num = tf.convert_to_tensor([[]],dtype=dist_symb.dtype)
        return tf.constant(True), dist_symb, dist_num, ndims, nsamples

    def handle_ndarray_or_tensor():
        conditional_tf_print(verbose, "Input distribution is a numeric numpy array or tf.Tensor.")
        if tf.rank(dist_input) != 2:
            tf.debugging.assert_equal(tf.rank(dist_input), 2, "Input must be a 2-dimensional numpy array or a tfp.distributions.Distribution object.")
        dist_symb = tfp.distributions.Normal(loc=tf.zeros(dist_input.shape[1]), scale=tf.ones(dist_input.shape[1])) # type: ignore
        dist_num = tf.convert_to_tensor(dist_input)
        nsamples, ndims = tf.unstack(tf.shape(dist_num))
        return tf.constant(False), dist_symb, dist_num, ndims, nsamples

    def handle_else():
        tf.debugging.assert_equal(
            tf.reduce_any([is_distribution(), is_ndarray_or_tensor()]),
            True,
            "Input must be either a numpy array or a tfp.distributions.Distribution object."
        )

    conditional_tf_print(verbose, "Parsing input distribution...")

    return tf.case([
        (is_distribution(), handle_distribution),
        (is_ndarray_or_tensor(), handle_ndarray_or_tensor)
    ], default=handle_else, exclusive=True)


def se_mean(data):
    n = len(data)
    mu_2 = moment(data, moment=2)  # second central moment (variance)
    se_mean = mu_2 / np.sqrt(n)
    return se_mean

def se_std(data):
    n = len(data)
    mu_2 = moment(data, moment=2)  # second central moment (variance)
    mu_4 = moment(data, moment=4)  # fourth central moment
    se_std = np.sqrt((mu_4 - mu_2**2) / (4 * mu_2 * n))
    return se_std

@tf.function(jit_compile=True, reduce_retracing=True)
def generate_and_clean_data_simple_1(dist: tfp.distributions.Distribution,
                                     n_samples: int,
                                     batch_size: int, 
                                     dtype: DTypeType,
                                     seed_generator: tf.random.Generator
                                    ) -> tf.Tensor:
    if dtype is None:
        dtype = tf.float32

    # Calculate maximum number of iterations
    max_iterations = n_samples // batch_size + 1  # +1 to handle the case where there's a remainder

    def sample_and_clean(dist, batch_size, seed_generator):
        new_seed = tf.cast(seed_generator.make_seeds(2)[0], tf.int32)
        batch = dist.sample(batch_size, seed=new_seed)
        finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)
        finite_batch = tf.boolean_mask(batch, finite_indices)
        return finite_batch, tf.shape(finite_batch)[0]

    total_samples = tf.constant(0, dtype=tf.int32)
    samples = tf.TensorArray(dtype, size=max_iterations)  # Fixed size TensorArray

    i = 0
    while total_samples < n_samples:
        try:
            finite_batch, finite_count = sample_and_clean(dist, batch_size, seed_generator)
            samples = samples.write(i, finite_batch)
            i += 1
            total_samples += finite_count
        except (RuntimeError, tf.errors.ResourceExhaustedError):
            batch_size = batch_size // 2
            if batch_size == 0:
                raise RuntimeError("Batch size is zero. Unable to generate samples.")

    samples = samples.concat()
    return samples[:n_samples] # type: ignore

@tf.function(jit_compile=True, reduce_retracing=True)
def generate_and_clean_data_simple_2(dist: tfp.distributions.Distribution,
                                     n_samples: int,
                                     batch_size: int, 
                                     dtype: DTypeType,
                                     seed_generator: tf.random.Generator
                                     ) -> tf.Tensor:
    if dtype is None:
        dtype = tf.float32

    # Calculate maximum number of iterations
    max_iterations = n_samples // batch_size + 1

    def sample_and_clean(dist, batch_size, seed_generator):
        new_seed = tf.cast(seed_generator.make_seeds(2)[0], tf.int32)
        batch = dist.sample(batch_size, seed=new_seed)
        finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)
        finite_batch = tf.boolean_mask(batch, finite_indices)
        return finite_batch, tf.shape(finite_batch)[0]

    def loop_cond(i, total_samples, samples, batch_size):
        return total_samples < n_samples

    def loop_body(i, total_samples, samples, batch_size):
        try:
            finite_batch, finite_count = sample_and_clean(dist, batch_size, seed_generator)
            samples = samples.write(i, finite_batch)
            i += 1
            total_samples += finite_count
        except (RuntimeError, tf.errors.ResourceExhaustedError):
            batch_size = batch_size // 2
            if batch_size == 0:
                raise RuntimeError("Batch size is zero. Unable to generate samples.")
        return i, total_samples, samples, batch_size

    total_samples = tf.constant(0, dtype=tf.int32)
    samples = tf.TensorArray(dtype, size=max_iterations)
    i = tf.constant(0, dtype=tf.int32)

    i, total_samples, samples, batch_size = tf.while_loop(
        cond=loop_cond, 
        body=loop_body, 
        loop_vars=[i, total_samples, samples, batch_size],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape(None),
            tf.TensorShape([])
        ]
    )

    samples = samples.concat()
    return samples[:n_samples]

def generate_and_clean_data_simple(*args, **kwargs):
    return generate_and_clean_data_simple_1(*args, **kwargs) # type: ignore

@tf.function(reduce_retracing=True)
def generate_and_clean_data_mirror_1(dist: tfp.distributions.Distribution,
                                     n_samples: int,
                                     batch_size: int, 
                                     dtype: DTypeType,
                                     seed_generator: tf.random.Generator,
                                     strategy: tf.distribute.Strategy
                                    ) -> tf.Tensor:
    #print("Generating data with mirrored strategy...")
    if dtype is None:
        dtype = tf.float32
    
    # Calculate maximum number of iterations
    max_iterations = n_samples // batch_size + 1  # +1 to handle the case where there's a remainder

    #jit_compile=True, 
    def sample_and_clean(dist, batch_size, seed_generator):
        new_seed = tf.cast(seed_generator.make_seeds(2)[0], tf.int32)
        batch = dist.sample(batch_size, seed=new_seed)
        finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)
        finite_batch = tf.boolean_mask(batch, finite_indices)
        return finite_batch, tf.shape(finite_batch)[0]

    total_samples = tf.constant(0, dtype=tf.int32)
    samples = tf.TensorArray(dtype, size=max_iterations)  # Fixed size TensorArray

    i = 0
    #with strategy.scope():
    while total_samples < n_samples:
        #try:
        per_replica_samples, per_replica_sample_count = strategy.run(sample_and_clean, args=(dist, batch_size, seed_generator))
        per_replica_samples_concat = tf.concat(strategy.experimental_local_results(per_replica_samples), axis=0)
        samples = samples.write(i, per_replica_samples_concat)
        i += 1
        total_samples += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_sample_count, axis=None)
        #print(f"Generated {total_samples} samples")
        #except (RuntimeError, tf.errors.ResourceExhaustedError):
        #    # If a RuntimeError or a ResourceExhaustedError occurs (possibly due to OOM), halve the batch size
        #    batch_size = batch_size // 2
        #    print(f"Warning: Batch size too large. Halving batch size to {batch_size} and retrying.")
        #    if batch_size == 0:
        #        raise RuntimeError("Batch size is zero. Unable to generate samples.")

    samples = samples.concat()
    return samples[:n_samples] # type: ignore

@tf.function(jit_compile=True, reduce_retracing=True)
def generate_and_clean_data_mirror_2(dist: tfp.distributions.Distribution,
                                     n_samples: int,
                                     batch_size: int, 
                                     dtype: DTypeType,
                                     seed_generator: tf.random.Generator,
                                     strategy: tf.distribute.Strategy
                                    ) -> tf.Tensor:
    if dtype is None:
        dtype = tf.float32

    # Calculate maximum number of iterations
    max_iterations = n_samples // batch_size + 1

    def sample_and_clean(dist, batch_size, seed_generator):
        new_seed = tf.cast(seed_generator.make_seeds(2)[0], tf.int32)
        batch = dist.sample(batch_size, seed=new_seed)
        finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)
        finite_batch = tf.boolean_mask(batch, finite_indices)
        return finite_batch, tf.shape(finite_batch)[0]

    def loop_cond(i, total_samples, samples, batch_size):
        return total_samples < n_samples

    def loop_body(i, total_samples, samples, batch_size):
        try:
            per_replica_samples, per_replica_sample_count = strategy.run(sample_and_clean, args=(dist, batch_size, seed_generator))
            per_replica_samples_concat = tf.concat(strategy.experimental_local_results(per_replica_samples), axis=0)
            samples = samples.write(i, per_replica_samples_concat)
            i += 1
            total_samples += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_sample_count, axis=None)
        except (RuntimeError, tf.errors.ResourceExhaustedError):
            batch_size = batch_size // 2
            if batch_size == 0:
                raise RuntimeError("Batch size is zero. Unable to generate samples.")
        return i, total_samples, samples, batch_size

    total_samples = tf.constant(0, dtype=tf.int32)
    samples = tf.TensorArray(dtype, size=max_iterations)
    i = tf.constant(0, dtype=tf.int32)

    #with strategy.scope():
    i, total_samples, samples, batch_size = tf.while_loop(
        cond=loop_cond, 
        body=loop_body, 
        loop_vars=[i, total_samples, samples, batch_size],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape(None),
            tf.TensorShape([])
        ]
    )

    samples = samples.concat()
    return samples[:n_samples]

def generate_and_clean_data_mirror(*args, **kwargs):
    return generate_and_clean_data_mirror_1(*args, **kwargs) # type: ignore

def generate_and_clean_data(dist: tfp.distributions.Distribution,
                            n_samples: int,
                            batch_size: int, 
                            dtype: DTypeType,
                            seed_generator: tf.random.Generator,
                            strategy: Optional[tf.distribute.Strategy] = None
                            ) -> tf.Tensor:
    if batch_size > n_samples:
        batch_size = n_samples
        #print("Warning: batch_size > n_samples. Setting batch_size = n_samples and proceeding.")
    #gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if strategy:
        return generate_and_clean_data_mirror(dist = dist, 
                                              n_samples = n_samples, 
                                              batch_size = batch_size, 
                                              dtype = dtype, 
                                              seed_generator = seed_generator,
                                              strategy = strategy)
    else:
        return generate_and_clean_data_simple(dist = dist, 
                                              n_samples = n_samples, 
                                              batch_size = batch_size, 
                                              dtype = dtype, 
                                              seed_generator = seed_generator)
        
def flatten_list(lst):
    out = []
    for item in lst:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(flatten_list(item))
        else:
            out.append(item)
    return out

def convert_types_dict(d):
    dd = {}
    for k, v in d.items():
        if isinstance(v, dict):
            dd[k] = convert_types_dict(v)
        elif type(v) == np.ndarray:
            dd[k] = v.tolist()
        elif type(v) == list:
            if str in [type(q) for q in flatten_list(v)]:
                dd[k] = np.array(v, dtype=object).tolist()
            else:
                dd[k] = np.array(v).tolist()
        else:
            dd[k] = np.array(v).tolist()
    return dd

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        elif isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalars to Python scalars
        elif isinstance(obj, tf.DType):
            return obj.name  # Assuming 'DType' objects have a 'name' attribute for their string representation
        elif isinstance(obj, np.dtype):
            return np.dtype(obj).name
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif callable(obj):
            return "callable: {}".format(obj.__name__)
        
        # Fall back to the default behavior
        return json.JSONEncoder.default(self, obj)

def save_update_metrics_config(metrics_config: Dict[str,Any],
                               metrics_config_file: str
                              ) -> pd.DataFrame:
    # Step 1: Read the existing content if the file exists
    existing_data = {}
    if os.path.exists(metrics_config_file):
        try:
            with open(metrics_config_file, "r") as file:
                existing_data = json.load(file)
        except json.JSONDecodeError:
            # File is empty or corrupted, start with an empty dictionary
            existing_data = {}

    # Step 2: Update the dictionary with new results
    existing_data.update(metrics_config)

    # Step 3: Write the updated dictionary back to the file
    # Use this custom encoder when dumping your JSON data
    with open(metrics_config_file, "w") as file:
        json.dump(existing_data, file, cls=CustomEncoder, indent=4)
        
    dict_values_list = [x.values() for x in metrics_config.values()]
    flat_list_of_dicts = [item for sublist in dict_values_list for item in sublist]
    
    return pd.DataFrame(flat_list_of_dicts)

import sys
from datetime import datetime
sys.path.insert(0,'../utils_func/')
import MixtureDistributions # type: ignore
sys.path.insert(0,'../')
import GMetrics # type: ignore

def compute_exclusion_adaptive_bisection(metric_config: Dict[str,Any],
                                         test_kwargs: Dict[str,Any],
                                         model_dir: str,
                                         deformation: str = "mean", # could be mean, std, or both
                                         seed_dist: int = 0,
                                         x_tol: float = 0.01,
                                         fn_tol: float = 0.01,
                                         eps_min: float = 0.,
                                         eps_max: float = 1.,
                                         initial_division_factor: float = 1 / 2.,
                                         reduce_division_factor: float = 1.,
                                         max_iterations: int = 100,
                                         save: bool = True,
                                         verbose: bool = True
                                        ) -> Dict[str,Any]:
    # Generate timestamp for result
    timestamp: str = datetime.now().isoformat()
    
    if verbose:
        print("\n======================================================")
    if deformation == "mean":
        if verbose:
            print(f"=============== {metric_config['name']} - only mean ===============")
    elif deformation == "std":
        if verbose:
            print(f"=============== {metric_config['name']} - only std ===============")
    elif deformation == "both":
        if verbose:
            print(f"=============== {metric_config['name']} - mean+std ===============")
    else:
        raise ValueError(f"Invalid value for mean_std_both: {deformation}")
    if verbose:
        print("======================================================") 
    
    # Define metric name:
    metric_name = metric_config["name"]
    metric_class = eval(metric_config["class_name"])
    metric_kwargs = metric_config["kwargs"]
    metric_result_key = metric_config["result_key"]
    metric_scale_func = metric_config["scale_func"]
    max_vectorize = metric_config["max_vectorize"]
    
    # Define ncomp and ndims
    ncomp = metric_config["test_config"]["ncomp"]
    ndims = metric_config["test_config"]["ndims"]
    
    # Compute metric scaling factor
    nsamples = test_kwargs["batch_size_test"]
    ns = nsamples**2 / (2 * nsamples)

    metrics_list = []
    eps_list = []
    exclusion_list = []

    metric_thresholds = metric_config["thresholds"][-2:]
    metric_threshold_number = 0
    eps_min_start = eps_min
    eps_max_start = eps_max
    division_factor = initial_division_factor
    initial_relative_error_eps = 2 * (eps_max - eps_min) / (eps_max + eps_min)
    relative_error_eps_threshold = initial_relative_error_eps
    initial_relative_error_metric = 2 * (eps_max - eps_min) / (eps_max + eps_min)
    relative_error_metric_threshold = initial_relative_error_metric
    eps = (eps_max + eps_min) / 2.

    start_global = timer()
    start = timer()
    
    dist_1 = MixtureDistributions.MixtureGaussian(ncomp, ndims, 0., 0., seed_dist)

    iteration = 0

    while metric_threshold_number < len(metric_thresholds) and iteration < max_iterations:
        iteration += 1
        
        if deformation == "mean":
            eps_mean = eps
            eps_std = 0.
        elif deformation == "std":
            eps_mean = 0.
            eps_std = eps
        else:
            eps_mean = eps
            eps_std = eps
        
        print(f"\n------------ {iteration} ------------")
        print(f"eps = {eps}")
        
        dist_2 = MixtureDistributions.MixtureGaussian(ncomp, ndims, eps_mean, eps_std, seed_dist)

        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs)

        Metric = metric_class(TwoSampleTestInputs, **metric_kwargs) # type: ignore
        Metric.Test_tf(max_vectorize=max_vectorize)
        metric = np.mean(Metric.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims) # type: ignore

        metrics_list.append(metric)
        eps_list.append(eps)

        # Determine direction of adjustment based on overshooting or undershooting
        if metric > metric_thresholds[metric_threshold_number][2]: # type: ignore
            #direction = -1
            eps_max = eps  # Update the maximum bound
            eps = eps_max - (eps_max - eps_min) * division_factor
        else:
            #direction = 1
            eps_min = eps  # Update the minimum bound
            eps = eps_min + (eps_max - eps_min) * division_factor
                        
        if verbose:
            print(f"statistic = {metric} - next threshold = {metric_thresholds[metric_threshold_number][2]} at {metric_thresholds[metric_threshold_number][0]} CL")

        relative_error_eps = 2 * (eps_max - eps_min) / (eps_max + eps_min)
        relative_error_metric = 2 * np.abs(metric_thresholds[metric_threshold_number][2] - metric) / (metric_thresholds[metric_threshold_number][2] + metric)
        if verbose:
            print(f"relative_error_eps = {relative_error_eps}")
            print(f"relative_error_metric = {relative_error_metric}")

        if division_factor / reduce_division_factor <= 1 / 2.:
            division_factor = 1 / 2.
        else:
            if relative_error_eps < relative_error_eps_threshold  / 2 and relative_error_metric < relative_error_metric_threshold / 2:
                print(f"Relative error halved. Reducing division factor from {division_factor} to {division_factor / reduce_division_factor}")
                relative_error_eps_threshold = relative_error_eps
                relative_error_metric_threshold = relative_error_metric
                division_factor = division_factor / reduce_division_factor
            
        
        # Check if the fn value is within the required accuracy of the threshold
        if relative_error_eps < x_tol and relative_error_metric < fn_tol:
            end = timer()
            if verbose:
                print(f"=======> statistic within required accuracy at {metric_thresholds[metric_threshold_number][0]} CL in {end - start} seconds")
            exclusion_list.append([metric_thresholds[metric_threshold_number][0], metric_name, eps, metric, end - start])
            metric_threshold_number += 1
            print("\n======================================================")
            print("New threshold. Resetting eps_min and eps_max.")
            start = timer() # Reset the timer
            iteration = 0
            eps_min, eps_max = eps, eps_max_start # Initialize the bounds
            division_factor = initial_division_factor
            relative_error_eps_threshold = initial_relative_error_eps
            relative_error_metric_threshold = initial_relative_error_metric
        
    end = timer()
    if verbose:
        print("Time elapsed:", end - start_global, "seconds.")
    result = {timestamp: {"test_config": test_kwargs,
                          "null_config": metric_config,
                          "deformation": deformation,
                          "parameters": {"ncomp": ncomp,
                                         "seed_dist": seed_dist,
                                         "x_tol": x_tol,
                                         "fn_tol": fn_tol,
                                         "eps_min": eps_min_start,
                                         "eps_max": eps_max_start,
                                         "max_iterations": max_iterations,
                                         "save": save,
                                         "verbose": verbose},
                          "exclusion_list": exclusion_list,
                          "eps_list": eps_list,
                          "metrics_list": metrics_list,
                          "time_elapsed": end - start_global}}
    
    # Saving if required
    if save:
        file_path = model_dir + "exclusion_limits.json"
        if verbose:
            print(f"Saving results in the file {file_path}")
        # Step 1: Read the existing content if the file exists
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                # File is empty or corrupted, start with an empty dictionary
                existing_data = {}
        
        # Step 2: Update the dictionary with new results
        existing_data.update(result)
        
        # Step 3: Write the updated dictionary back to the file
        # Use this custom encoder when dumping your JSON data
        with open(file_path, "w") as file:
            json.dump(existing_data, file, cls=GMetrics.utils.CustomEncoder, indent=4) # type: ignore
    
    return result

def compute_exclusion_bisection(metric_config: Dict[str,Any],
                                test_kwargs: Dict[str,Any],
                                model_dir: str,
                                deformation: str = "mean", # could be mean, std, or both
                                seed_dist: int = 0,
                                x_tol: float = 0.01,
                                fn_tol: float = 0.01,
                                eps_min: float = 0.,
                                eps_max: float = 1.,
                                max_iterations: int = 100,
                                save: bool = True,
                                verbose: bool = True
                               ) -> Dict[str,Any]:
    # Generate timestamp for result
    timestamp: str = datetime.now().isoformat()
    
    if verbose:
        print("\n======================================================")
    if deformation == "mean":
        if verbose:
            print(f"=============== {metric_config['name']} - only mean ===============")
    elif deformation == "std":
        if verbose:
            print(f"=============== {metric_config['name']} - only std ===============")
    elif deformation == "both":
        if verbose:
            print(f"=============== {metric_config['name']} - mean+std ===============")
    else:
        raise ValueError(f"Invalid value for mean_std_both: {deformation}")
    if verbose:
        print("======================================================") 
    
    # Define metric name:
    metric_name = metric_config["name"]
    metric_class = eval(metric_config["class_name"])
    metric_kwargs = metric_config["kwargs"]
    metric_result_key = metric_config["result_key"]
    metric_scale_func = metric_config["scale_func"]
    max_vectorize = metric_config["max_vectorize"]
    
    # Define ncomp and ndims
    ncomp = metric_config["test_config"]["ncomp"]
    ndims = metric_config["test_config"]["ndims"]
    
    # Compute metric scaling factor
    nsamples = test_kwargs["batch_size_test"]
    ns = nsamples**2 / (2 * nsamples)

    metrics_list = []
    eps_list = []
    exclusion_list = []

    metric_thresholds = metric_config["thresholds"][-2:]
    metric_threshold_number = 0
    eps_min_start = eps_min
    eps_max_start = eps_max
    eps = (eps_max + eps_min) / 2.

    start_global = timer()
    start = timer()
    
    dist_1 = MixtureDistributions.MixtureGaussian(ncomp, ndims, 0., 0., seed_dist)

    iteration = 0

    while metric_threshold_number < len(metric_thresholds) and iteration < max_iterations:
        iteration += 1
        
        if deformation == "mean":
            eps_mean = eps
            eps_std = 0.
        elif deformation == "std":
            eps_mean = 0.
            eps_std = eps
        else:
            eps_mean = eps
            eps_std = eps
        
        print(f"\n------------ {iteration} ------------")
        print(f"eps = {eps}")
        
        dist_2 = MixtureDistributions.MixtureGaussian(ncomp, ndims, eps_mean, eps_std, seed_dist)

        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs)

        Metric = metric_class(TwoSampleTestInputs, **metric_kwargs) # type: ignore
        Metric.Test_tf(max_vectorize=max_vectorize)
        metric = np.mean(Metric.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims) # type: ignore

        metrics_list.append(metric)
        eps_list.append(eps)

        # Determine direction of adjustment based on overshooting or undershooting
        if metric > metric_thresholds[metric_threshold_number][2]: # type: ignore
            #direction = -1
            eps_max = eps  # Update the maximum bound
            eps = eps_max - (eps_max - eps_min) / 2.
        else:
            #direction = 1
            eps_min = eps  # Update the minimum bound
            eps = eps_min + (eps_max - eps_min) / 2.
                        
        if verbose:
            print(f"statistic = {metric} - next threshold = {metric_thresholds[metric_threshold_number][2]} at {metric_thresholds[metric_threshold_number][0]} CL")

        relative_error_eps = 2 * (eps_max - eps_min) / (eps_max + eps_min)
        relative_error_metric = 2 * np.abs(metric_thresholds[metric_threshold_number][2] - metric) / (metric_thresholds[metric_threshold_number][2] + metric)
        if verbose:
            print(f"relative_error_eps = {relative_error_eps}")
            print(f"relative_error_metric = {relative_error_metric}")
        
        # Check if the fn value is within the required accuracy of the threshold
        if relative_error_eps < x_tol and relative_error_metric < fn_tol:
            end = timer()
            if verbose:
                print(f"=======> statistic within required accuracy at {metric_thresholds[metric_threshold_number][0]} CL in {end - start} seconds")
            exclusion_list.append([metric_thresholds[metric_threshold_number][0], metric_name, eps, metric, end - start])
            metric_threshold_number += 1
            print("\n======================================================")
            print("New threshold. Resetting eps_min and eps_max.")
            start = timer() # Reset the timer
            iteration = 0
            eps_min, eps_max = eps, eps_max_start # Initialize the bounds
        
    end = timer()
    if verbose:
        print("Time elapsed:", end - start_global, "seconds.")
    result = {timestamp: {"test_config": test_kwargs,
                          "null_config": metric_config,
                          "deformation": deformation,
                          "parameters": {"ncomp": ncomp,
                                         "seed_dist": seed_dist,
                                         "x_tol": x_tol,
                                         "fn_tol": fn_tol,
                                         "eps_min": eps_min_start,
                                         "eps_max": eps_max_start,
                                         "max_iterations": max_iterations,
                                         "save": save,
                                         "verbose": verbose},
                          "exclusion_list": exclusion_list,
                          "eps_list": eps_list,
                          "metrics_list": metrics_list,
                          "time_elapsed": end - start_global}}
    
    # Saving if required
    if save:
        file_path = model_dir + "exclusion_limits.json"
        if verbose:
            print(f"Saving results in the file {file_path}")
        # Step 1: Read the existing content if the file exists
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                # File is empty or corrupted, start with an empty dictionary
                existing_data = {}
        
        # Step 2: Update the dictionary with new results
        existing_data.update(result)
        
        # Step 3: Write the updated dictionary back to the file
        # Use this custom encoder when dumping your JSON data
        with open(file_path, "w") as file:
            json.dump(existing_data, file, cls=GMetrics.utils.CustomEncoder, indent=4) # type: ignore
    
    return result


def compute_exclusion_LR_adaptive_bisection(metric_config: Dict[str,Any],
                                            test_kwargs: Dict[str,Any],
                                            model_dir: str,
                                            cl_list = [0.95, 0.99],
                                            deformation: str = "mean", # could be mean, std, or both
                                            seed_dist: int = 0,
                                            x_tol: float = 0.01,
                                            fn_tol: float = 0.01,
                                            eps_min: float = 0.,
                                            eps_max: float = 1.,
                                            initial_division_factor: float = 1 / 2.,
                                            reduce_division_factor: float = 1.,
                                            max_iterations: int = 100,
                                            save: bool = True,
                                            verbose: bool = True
                                           ) -> Dict[str,Any]:
    # Generate timestamp for result
    timestamp: str = datetime.now().isoformat()
    
    if verbose:
        print("\n======================================================")
    if deformation == "mean":
        if verbose:
            print(f"=============== {metric_config['name']} - only mean ===============")
    elif deformation == "std":
        if verbose:
            print(f"=============== {metric_config['name']} - only std ===============")
    elif deformation == "both":
        if verbose:
            print(f"=============== {metric_config['name']} - mean+std ===============")
    else:
        raise ValueError(f"Invalid value for mean_std_both: {deformation}")
    if verbose:
        print("======================================================")
        
    test_kwargs_null = dict(test_kwargs)
    test_kwargs_alt = dict(test_kwargs)
    test_kwargs_alt["niter"] = 10
    
    metric_kwargs_null = dict(metric_config["kwargs"])
    metric_kwargs_alt = dict(metric_config["kwargs"])
    metric_kwargs_null["null_test"] = True
    metric_kwargs_alt["null_test"] = False
    #metric_kwargs_null["verbose"] = True
    #metric_kwargs_alt["verbose"] = True
    
    metric_name = metric_config["name"]
    metric_result_key = metric_config["result_key"]
    metric_scale_func = metric_config["scale_func"]
    max_vectorize = metric_config["max_vectorize"]
    null_file_base = metric_config["null_file"]
    metrics_config_file = model_dir + "metrics_config.json"
    
    # Define ncomp and ndims
    ncomp = metric_config["test_config"]["ncomp"]
    ndims = metric_config["test_config"]["ndims"]
    
    # Compute metric scaling factor
    nsamples = test_kwargs["batch_size_test"]
    ns = nsamples**2 / (2 * nsamples)
    
    metrics_list = []
    eps_list = []
    exclusion_list = []
    
    metric_threshold_number = 0
    eps_min_start = eps_min
    eps_max_start = eps_max
    division_factor = initial_division_factor
    initial_relative_error_eps = 2 * (eps_max - eps_min) / (eps_max + eps_min)
    relative_error_eps_threshold = initial_relative_error_eps
    initial_relative_error_metric = 2 * (eps_max - eps_min) / (eps_max + eps_min)
    relative_error_metric_threshold = initial_relative_error_metric
    eps = (eps_max + eps_min) / 2.
    
    start_global = timer()
    start = timer()
    
    dist_1 = MixtureDistributions.MixtureGaussian(ncomp, ndims, 0., 0., seed_dist)

    iteration = 0

    while metric_threshold_number < len(cl_list) and iteration < max_iterations:
        iteration += 1

        if deformation == "mean":
            eps_mean = eps
            eps_std = 0.
        elif deformation == "std":
            eps_mean = 0.
            eps_std = eps
        else:
            eps_mean = eps
            eps_std = eps
            
        print(f"\n------------ {iteration} ------------")
        print(f"eps = {eps}")

        print(f"Computing null distribution")
        start_null = timer()
        dist_2 = MixtureDistributions.MixtureGaussian(ncomp, ndims, eps_mean, eps_std, seed_dist)
        
        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs_null)
        
        LRMetric_null = GMetrics.LRMetric(TwoSampleTestInputs, **metric_kwargs_null)
        LRMetric_null.Test_tf(max_vectorize = max_vectorize)
        
        null_file = null_file_base.replace(".json", "_" + deformation + "_" + str(format(eps, '.6f')) + ".json")
        print("Saving", metric_name, "to", null_file)
        LRMetric_null.Results.save_to_json(null_file)
        
        dist_null = np.array(LRMetric_null.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims)
        metric_thresholds = [[cl, 
                              [int(cl*len(dist_null)), 
                               int((1-cl)*len(dist_null))], 
                              np.sort(dist_null)[int(len(dist_null)*cl)]] for cl in cl_list]
        print(f"ThresholdS: {metric_thresholds}")
        metric_config["thresholds"].append([eps, deformation, metric_thresholds])
        save_update_metrics_config(metrics_config = metric_config, 
                                   metrics_config_file = metrics_config_file) # type: ignore
        end_null = timer()
        print(f"Null distribution computed in {end_null - start_null} seconds")
        
        print(f"Evaluating alternative distribution")
        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs_alt)
        LRMetric_alt = GMetrics.LRMetric(TwoSampleTestInputs, **metric_kwargs_alt)
        LRMetric_alt.Test_tf(max_vectorize = max_vectorize)
        metric = np.mean(LRMetric_alt.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims) # type: ignore
        
        metrics_list.append(metric)
        eps_list.append(eps)

        # Determine direction of adjustment based on overshooting or undershooting
        if metric > metric_thresholds[metric_threshold_number][2]: # type: ignore
            #direction = -1
            eps_max = eps  # Update the maximum bound
            eps = eps_max - (eps_max - eps_min) * division_factor
        else:
            #direction = 1
            eps_min = eps  # Update the minimum bound
            eps = eps_min + (eps_max - eps_min) * division_factor
                        
        if verbose:
            print(f"statistic = {metric} - next threshold = {metric_thresholds[metric_threshold_number][2]} at {metric_thresholds[metric_threshold_number][0]} CL")

        relative_error_eps = 2 * (eps_max - eps_min) / (eps_max + eps_min)
        relative_error_metric = 2 * np.abs(metric_thresholds[metric_threshold_number][2] - metric) / (metric_thresholds[metric_threshold_number][2] + metric)
        if verbose:
            print(f"relative_error_eps = {relative_error_eps}")
            print(f"relative_error_metric = {relative_error_metric}")

        if division_factor / reduce_division_factor <= 1 / 2.:
            division_factor = 1 / 2.
        else:
            if relative_error_eps < relative_error_eps_threshold  / 2 and relative_error_metric < relative_error_metric_threshold / 2:
                print(f"Relative error halved. Reducing division factor from {division_factor} to {division_factor / reduce_division_factor}")
                relative_error_eps_threshold = relative_error_eps
                relative_error_metric_threshold = relative_error_metric
                division_factor = division_factor / reduce_division_factor
            
        
        # Check if the fn value is within the required accuracy of the threshold
        if relative_error_eps < x_tol and relative_error_metric < fn_tol:
            end = timer()
            if verbose:
                print(f"=======> statistic within required accuracy at {metric_thresholds[metric_threshold_number][0]} CL in {end - start} seconds")
            exclusion_list.append([metric_thresholds[metric_threshold_number][0], metric_name, eps, metric, end - start])
            metric_threshold_number += 1
            print("\n======================================================")
            print("New threshold. Resetting eps_min and eps_max.")
            start = timer() # Reset the timer
            iteration = 0
            eps_min, eps_max = eps, eps_max_start # Initialize the bounds
            division_factor = initial_division_factor
            relative_error_eps_threshold = initial_relative_error_eps
            relative_error_metric_threshold = initial_relative_error_metric
            
    end = timer()
    if verbose:
        print("Time elapsed:", end - start_global, "seconds.")
    result = {timestamp: {"test_config": test_kwargs,
                          "null_config": metric_config,
                          "deformation": deformation,
                          "parameters": {"ncomp": ncomp,
                                         "seed_dist": seed_dist,
                                         "x_tol": x_tol,
                                         "fn_tol": fn_tol,
                                         "eps_min": eps_min_start,
                                         "eps_max": eps_max_start,
                                         "max_iterations": max_iterations,
                                         "save": save,
                                         "verbose": verbose},
                          "exclusion_list": exclusion_list,
                          "eps_list": eps_list,
                          "metrics_list": metrics_list,
                          "time_elapsed": end - start_global}}
    
    # Saving if required
    if save:
        file_path = model_dir + "exclusion_limits.json"
        if verbose:
            print(f"Saving results in the file {file_path}")
        # Step 1: Read the existing content if the file exists
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                # File is empty or corrupted, start with an empty dictionary
                existing_data = {}
        
        # Step 2: Update the dictionary with new results
        existing_data.update(result)
        
        # Step 3: Write the updated dictionary back to the file
        # Use this custom encoder when dumping your JSON data
        with open(file_path, "w") as file:
            json.dump(existing_data, file, cls=GMetrics.utils.CustomEncoder, indent=4) # type: ignore
    
    return result

def compute_exclusion_LR_bisection(metric_config: Dict[str,Any],
                                   test_kwargs: Dict[str,Any],
                                   model_dir: str,
                                   cl_list = [0.95, 0.99],
                                   deformation: str = "mean", # could be mean, std, or both
                                   seed_dist: int = 0,
                                   x_tol: float = 0.01,
                                   fn_tol: float = 0.01,
                                   eps_min: float = 0.,
                                   eps_max: float = 1.,
                                   max_iterations: int = 100,
                                   save: bool = True,
                                   verbose: bool = True
                                  ) -> Dict[str,Any]:
    # Generate timestamp for result
    timestamp: str = datetime.now().isoformat()
    
    if verbose:
        print("\n======================================================")
    if deformation == "mean":
        if verbose:
            print(f"=============== {metric_config['name']} - only mean ===============")
    elif deformation == "std":
        if verbose:
            print(f"=============== {metric_config['name']} - only std ===============")
    elif deformation == "both":
        if verbose:
            print(f"=============== {metric_config['name']} - mean+std ===============")
    else:
        raise ValueError(f"Invalid value for mean_std_both: {deformation}")
    if verbose:
        print("======================================================")
        
    test_kwargs_null = dict(test_kwargs)
    test_kwargs_alt = dict(test_kwargs)
    test_kwargs_alt["niter"] = 10
    
    metric_kwargs_null = dict(metric_config["kwargs"])
    metric_kwargs_alt = dict(metric_config["kwargs"])
    metric_kwargs_null["null_test"] = True
    metric_kwargs_alt["null_test"] = False
    #metric_kwargs_null["verbose"] = True
    #metric_kwargs_alt["verbose"] = True
    
    metric_name = metric_config["name"]
    metric_result_key = metric_config["result_key"]
    metric_scale_func = metric_config["scale_func"]
    max_vectorize = metric_config["max_vectorize"]
    null_file_base = metric_config["null_file"]
    metrics_config_file = model_dir + "metrics_config.json"
    
    # Define ncomp and ndims
    ncomp = metric_config["test_config"]["ncomp"]
    ndims = metric_config["test_config"]["ndims"]
    
    # Compute metric scaling factor
    nsamples = test_kwargs["batch_size_test"]
    ns = nsamples**2 / (2 * nsamples)
    
    metrics_list = []
    eps_list = []
    exclusion_list = []
    
    metric_threshold_number = 0
    eps_min_start = eps_min
    eps_max_start = eps_max
    eps = (eps_max + eps_min) / 2.
    
    start_global = timer()
    start = timer()
    
    dist_1 = MixtureDistributions.MixtureGaussian(ncomp, ndims, 0., 0., seed_dist)

    iteration = 0

    while metric_threshold_number < len(cl_list) and iteration < max_iterations:
        iteration += 1

        if deformation == "mean":
            eps_mean = eps
            eps_std = 0.
        elif deformation == "std":
            eps_mean = 0.
            eps_std = eps
        else:
            eps_mean = eps
            eps_std = eps
            
        print(f"\n------------ {iteration} ------------")
        print(f"eps = {eps}")

        print(f"Computing null distribution")
        start_null = timer()
        dist_2 = MixtureDistributions.MixtureGaussian(ncomp, ndims, eps_mean, eps_std, seed_dist)
        
        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs_null)
        
        LRMetric_null = GMetrics.LRMetric(TwoSampleTestInputs, **metric_kwargs_null)
        LRMetric_null.Test_tf(max_vectorize = max_vectorize)
        
        null_file = null_file_base.replace(".json", "_" + deformation + "_" + str(format(eps, '.6f')) + ".json")
        print("Saving", metric_name, "to", null_file)
        LRMetric_null.Results.save_to_json(null_file)
        
        dist_null = np.array(LRMetric_null.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims)
        metric_thresholds = [[cl, 
                              [int(cl*len(dist_null)), 
                               int((1-cl)*len(dist_null))], 
                              np.sort(dist_null)[int(len(dist_null)*cl)]] for cl in cl_list]
        print(f"ThresholdS: {metric_thresholds}")
        metric_config["thresholds"].append([eps, deformation, metric_thresholds])
        save_update_metrics_config(metrics_config = metric_config, 
                                   metrics_config_file = metrics_config_file) # type: ignore
        end_null = timer()
        print(f"Null distribution computed in {end_null - start_null} seconds")
        
        print(f"Evaluating alternative distribution")
        TwoSampleTestInputs = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1,
                                                           dist_2_input = dist_2,
                                                           **test_kwargs_alt)
        LRMetric_alt = GMetrics.LRMetric(TwoSampleTestInputs, **metric_kwargs_alt)
        LRMetric_alt.Test_tf(max_vectorize = max_vectorize)
        metric = np.mean(LRMetric_alt.Results[-1].result_value[metric_result_key]) * metric_scale_func(ns, ndims) # type: ignore
        
        metrics_list.append(metric)
        eps_list.append(eps)

        # Determine direction of adjustment based on overshooting or undershooting
        if metric > metric_thresholds[metric_threshold_number][2]: # type: ignore
            #direction = -1
            eps_max = eps  # Update the maximum bound
            eps = eps_max - (eps_max - eps_min) / 2.
        else:
            #direction = 1
            eps_min = eps  # Update the minimum bound
            eps = eps_min + (eps_max - eps_min) / 2.
                        
        if verbose:
            print(f"statistic = {metric} - next threshold = {metric_thresholds[metric_threshold_number][2]} at {metric_thresholds[metric_threshold_number][0]} CL")

        relative_error_eps = 2 * (eps_max - eps_min) / (eps_max + eps_min)
        relative_error_metric = 2 * np.abs(metric_thresholds[metric_threshold_number][2] - metric) / (metric_thresholds[metric_threshold_number][2] + metric)
        if verbose:
            print(f"relative_error_eps = {relative_error_eps}")
            print(f"relative_error_metric = {relative_error_metric}")
         
        # Check if the fn value is within the required accuracy of the threshold
        if relative_error_eps < x_tol and relative_error_metric < fn_tol:
            end = timer()
            if verbose:
                print(f"=======> statistic within required accuracy at {metric_thresholds[metric_threshold_number][0]} CL in {end - start} seconds")
            exclusion_list.append([metric_thresholds[metric_threshold_number][0], metric_name, eps, metric, end - start])
            metric_threshold_number += 1
            print("\n======================================================")
            print("New threshold. Resetting eps_min and eps_max.")
            start = timer() # Reset the timer
            iteration = 0
            eps_min, eps_max = eps, eps_max_start # Initialize the bounds
            
    end = timer()
    if verbose:
        print("Time elapsed:", end - start_global, "seconds.")
    result = {timestamp: {"test_config": test_kwargs,
                          "null_config": metric_config,
                          "deformation": deformation,
                          "parameters": {"ncomp": ncomp,
                                         "seed_dist": seed_dist,
                                         "x_tol": x_tol,
                                         "fn_tol": fn_tol,
                                         "eps_min": eps_min_start,
                                         "eps_max": eps_max_start,
                                         "max_iterations": max_iterations,
                                         "save": save,
                                         "verbose": verbose},
                          "exclusion_list": exclusion_list,
                          "eps_list": eps_list,
                          "metrics_list": metrics_list,
                          "time_elapsed": end - start_global}}
    
    # Saving if required
    if save:
        file_path = model_dir + "exclusion_limits.json"
        if verbose:
            print(f"Saving results in the file {file_path}")
        # Step 1: Read the existing content if the file exists
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                # File is empty or corrupted, start with an empty dictionary
                existing_data = {}
        
        # Step 2: Update the dictionary with new results
        existing_data.update(result)
        
        # Step 3: Write the updated dictionary back to the file
        # Use this custom encoder when dumping your JSON data
        with open(file_path, "w") as file:
            json.dump(existing_data, file, cls=GMetrics.utils.CustomEncoder, indent=4) # type: ignore
    
    return result