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
import random
from scipy.stats import moment # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from numpy import random as npr

from typing import Tuple, Union, Optional, Type, Callable, Dict
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
DistTypeTF = Type[tfp.distributions.Distribution]

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
            raise ValueError(f"{self.distribution} is not a callable method of numpy.random.Generator.")

DistTypeNP = NumpyDistribution
DistType = Union[DistTypeTF, DistTypeNP]
DataDistTypeNP = Union[DataTypeNP, DistTypeNP]
DataDistTypeTF = Union[DataTypeTF, DistTypeTF]
DataDistType = Union[DataDistTypeNP, DataDistTypeTF]
BoolType = Union[bool, BoolTypeTF, BoolTypeNP]
    
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
        raise ValueError("Input must be either a numpy array or a NumpyDistribution object.")
    return is_symb, dist_symb, dist_num, ndims, nsamples


def parse_input_dist_tf(dist_input: DataDistType,
                        verbose: bool = False
                       ) -> Tuple[BoolType, DistTypeTF, DataTypeTF, IntType, IntType]:
    
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