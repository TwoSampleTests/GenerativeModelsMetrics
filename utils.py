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
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from numpy import random as npr

from typing import Tuple, Union, Optional, Type, TypeAlias, TypeVar, Callable, Dict
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

@tf.function
def conditional_tf_print(verbose: tf.Tensor = tf.Tensor(False),
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
    
@tf.function#(jit_compile=True)
def ks_2samp_tf(data1: tf.Tensor, 
                data2: tf.Tensor,
                alternative: str = 'two-sided',
                method: str = 'auto',
                precision: int = 100,
                verbose: bool = False,
                debug: bool = False
               ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute the Kolmogorov-Smirnov statistic on 2 samples.

    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution. 

    Parameters:
    data1, data2: tf.Tensor
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different.
        
    alternative: str, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
        - 'two-sided'
        - 'less': one-sided, see explanation in scipy.stats.ks_2samp function
        - 'greater': one-sided, see explanation in scipy.stats.ks_2samp function

    method: str, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):
        - 'auto': use 'exact' for small size arrays, 'asymp' for larger
        - 'exact': use exact distribution of test statistic
        - 'asymp': use asymptotic distribution of test statistic

    Returns: 
    d: tf.Tensor
        KS statistic.

    prob: tf.Tensor
        Two-tailed p-value.

    d_location: tf.Tensor
        The x-location of the maximum difference of the cumulative distribution function.

    Note: The 'exact' method is not yet implemented. If selected, the function 
    will fall back to 'asymp'. 
    """
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Invalid alternative.")
    if method not in ['auto', 'exact', 'asymp']:
        raise ValueError("Invalid method.")
    alternative_dict: Dict[str,int] = {'two-sided': 0, 'less': 1, 'greater': 2}
    method_dict: Dict[str,int] = {'auto': 0, 'exact': 1, 'asymp': 2}
    
    # Convert string input to integer codes.
    alternative_int: int = alternative_dict.get(alternative, 0)
    method_int: int = method_dict.get(method, 0)
    d: tf.Tensor
    prob: tf.Tensor
    d_location: tf.Tensor
    d, prob, d_location = _ks_2samp_tf_internal(data1 = data1, 
                                                data2 = data2, 
                                                alternative_int = alternative_int, 
                                                method_int = method_int,
                                                precision = precision,
                                                verbose = verbose,
                                                debug = debug) # type: ignore
    return d, prob, d_location
    
        
def _ks_2samp_tf_internal(data1: tf.Tensor, 
                          data2: tf.Tensor,
                          #n1: tf.Tensor,
                          #n2: tf.Tensor,
                          #g: tf.Tensor,
                          #n1g: tf.Tensor,
                          #n2g: tf.Tensor,
                          alternative_int: int = 0,
                          method_int: int = 0,
                          precision: int = 100,
                          verbose: bool = False,
                          debug: bool = False
                         ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    alternative_dict = {'two-sided': 0, 'less': 1, 'greater': 2}
    method_dict = {'auto': 0, 'exact': 1, 'asymp': 2}
    """
    #alternative: tf.Tensor = tf.convert_to_tensor(alternative_int, dtype=tf.int32)
    #method: tf.Tensor = tf.convert_to_tensor(method_int, dtype=tf.int32)
    alternative: int = alternative_int
    method: int = method_int
    mode: int = method
    #if debug:
    #    # Check the correct alternatives
    #    tf.debugging.assert_less_equal(alternative, 2, "Invalid alternative.")
    #    tf.debugging.assert_greater_equal(alternative, 0, "Invalid alternative.")
    #    # Check the correct methods
    #    tf.debugging.assert_less_equal(mode, 2, "Invalid method.")
    #    tf.debugging.assert_greater_equal(mode, 0, "Invalid method.")
    #    # Check the correct input dimensions
    #    tf.debugging.assert_equal(tf.rank(data1), 1, message="data1 must be 1-dimensional.")
    #    tf.debugging.assert_equal(tf.rank(data2), 1, message="data2 must be 1-dimensional.")
    #    # Check that there are no `NaN` or `Inf` values
    #    tf.debugging.assert_none_equal(tf.math.is_nan(data1), tf.constant(True, dtype=tf.bool), message="data1 must not contain NaN values.")
    #    tf.debugging.assert_none_equal(tf.math.is_nan(data2), tf.constant(True, dtype=tf.bool), message="data2 must not contain NaN values.")
    #    tf.debugging.assert_none_equal(tf.math.is_inf(data1), tf.constant(True, dtype=tf.bool), message="data1 must not contain Inf values.")
    #    tf.debugging.assert_none_equal(tf.math.is_inf(data2), tf.constant(True, dtype=tf.bool), message="data2 must not contain Inf values.")
    
    def greatest_common_divisor_tf(x, y):
        while tf.not_equal(y, 0):
            x, y = y, tf.math.floormod(x, y)
        return x

    MAX_AUTO_N: tf.Tensor = tf.constant(10000, dtype=tf.float32) # type: ignore
    
    data1 = tf.sort(data1)
    data2 = tf.sort(data2)
    
    n1: tf.Tensor = tf.cast(tf.shape(data1)[0], tf.float32) # type: ignore
    n2: tf.Tensor = tf.cast(tf.shape(data2)[0], tf.float32) # type: ignore
    
    data_all: tf.Tensor = tf.concat([data1, data2], axis=0) # type: ignore

    # using searchsorted solves equal data problem
    cdf1: tf.Tensor = tf.cast(tf.searchsorted(data1, data_all, side = 'right'), tf.float32) / n1
    cdf2: tf.Tensor = tf.cast(tf.searchsorted(data2, data_all, side = 'right'), tf.float32) / n2
    cddiffs: tf.Tensor = cdf1 - cdf2
    
    # Identify the location of the statistic
    argminS: tf.Tensor = tf.argmin(cddiffs)
    argmaxS: tf.Tensor = tf.argmax(cddiffs)
    loc_minS: tf.Tensor = data_all[argminS]
    loc_maxS: tf.Tensor = data_all[argmaxS]
    
    # Ensure sign of minS is not negative.
    minS: tf.Tensor = tf.clip_by_value(-cddiffs[argminS], clip_value_min = 0, clip_value_max = 1) # type: ignore
    maxS: tf.Tensor = cddiffs[argmaxS]
    
    max_abs_diff: tf.Tensor = tf.maximum(minS, maxS) # type: ignore
    less_max: tf.Tensor = tf.greater_equal(minS, maxS) # type: ignore

    location: tf.Tensor = tf.where(less_max, loc_minS, loc_maxS)
    #sign: tf.Tensor = tf.where(less_max, -1, 1)
    
    d: tf.Tensor = tf.where(tf.equal(alternative, 0), x=max_abs_diff, y=tf.where(tf.equal(alternative, 1), x=minS, y=maxS))
    d_location: tf.Tensor = tf.where(tf.equal(alternative, 0), x=location, y=tf.where(tf.equal(alternative, 1), x=loc_minS, y=loc_maxS))
    #d_sign: tf.Tensor = tf.where(tf.equal(alternative, 0), x=sign, y=tf.where(tf.equal(alternative, 1), x=-1, y=1))

    g: tf.Tensor = greatest_common_divisor_tf(n1, n2)
    n1g: tf.Tensor = tf.math.floordiv(n1, g) # type: ignore
    n2g: tf.Tensor = tf.math.floordiv(n2, g) # type: ignore
    prob: tf.Tensor = -tf.float32.max # type: ignore
        
    #def switch_to_asymp_large_sample(n1: tf.Tensor,
    #                                 n2: tf.Tensor
    #                                ) -> int:
    #    conditional_tf_print(tf.constant(verbose), "Exact ks_2samp calculation not possible with sample sizes "+str(n1)+" and "+str(n2)+". Switching to 'asymp' method.")
    #    result: int = 2
    #    return result
    
    #def switch_to_asymp_not_implemented():
    #    conditional_tf_print(tf.constant(verbose), "Exact ks_2samp calculation not yet implemented. Switching to 'asymp' method.")
    #    result: int = 2
    #    return result

    # If mode is 'auto' (0), decide between 'exact' (1) and 'asymp'  (2) based on n1, n2
    mode = tf.where(tf.equal(mode, 0),
                tf.where(tf.less_equal(tf.reduce_max([n1, n2]), MAX_AUTO_N), 
                         x=1,
                         y=2),
                mode)

    
    # If lcm(n1, n2) is too big, switch from 'exact' (1) to 'asymp' (2)
    mode = tf.where(tf.logical_and(tf.equal(mode, 1), tf.greater_equal(n1g, tf.int32.max / n2g)),
                x=2,
                y=mode)


    # Exact calculation is not yet implemented, so switch from 'exact' (1) to 'asymp' (2)
    mode = tf.where(tf.equal(mode, 1), x=2, y=mode)
    
    def asymp_ks_2samp(n1: tf.Tensor,
                       n2: tf.Tensor,
                       d: tf.Tensor,
                       alternative: int,
                       precision: int
                      ) -> Tuple[tf.Tensor, tf.Tensor]:
        #tf.print("Executing asymp_ks_2samp")
        sorted_values: tf.Tensor = tf.sort(tf.stack([tf.cast(n1, tf.float32), tf.cast(n2, tf.float32)]), direction='DESCENDING')
        m: tf.Tensor = sorted_values[0]
        n: tf.Tensor = sorted_values[1]
        en: tf.Tensor = m * n / (m + n)
        
        def kolmogorov_cdf(x: tf.Tensor, 
                           precision: int
                          ) -> tf.Tensor:
            k_values: tf.Tensor = tf.range(-precision, precision + 1, dtype=tf.float32)
            terms: tf.Tensor = (-1.)**k_values * tf.exp(-2. * k_values**2 * x**2)
            prob: tf.Tensor = tf.reduce_sum(terms)
            return prob
        
        def two_sided_p_value(d: tf.Tensor,
                              en: tf.Tensor,
                              precision: int
                             ) -> tf.Tensor:
            z: tf.Tensor = tf.sqrt(en) * d
            prob: tf.Tensor = 1 - kolmogorov_cdf(z, precision) # type: ignore
            return prob

        def one_sided_p_value() -> tf.Tensor:
            z = tf.sqrt(en) * d
            expt = -2 * z**2 - 2 * z * (m + 2*n)/tf.sqrt(m*n*(m+n))/3.0
            prob = tf.exp(expt)
            return prob
        
        prob = tf.where(tf.equal(alternative, 0), x=two_sided_p_value(d, en, precision), y=one_sided_p_value())

        return d, prob
        
    #def exact_ks_2samp(n1: tf.Tensor,
    #                   n2: tf.Tensor,
    #                   g: tf.Tensor,
    #                   d: tf.Tensor,
    #                   alternative: int
    #                  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    #    #conditional_tf_print(tf.constant(verbose), "Exact ks_2samp calculation not yet implemented. Switching to 'asymp' method.")
    #    success: tf.Tensor = tf.convert_to_tensor(False, tf.bool)
    #    prob: tf.Tensor = -tf.float32.max # type: ignore
    #    return success, d, prob
    #    
    #def attempt_exact_ks_2samp(n1: tf.Tensor,
    #                           n2: tf.Tensor,
    #                           g: tf.Tensor,
    #                           d: tf.Tensor,
    #                           alternative: int,
    #                           precision: int = 1000
    #                          ) -> Tuple[tf.Tensor, tf.Tensor]:
    #    success: tf.Tensor
    #    prob: tf.Tensor
    #    success, d, prob = exact_ks_2samp(n1, n2, g, d, alternative)
    #    
    #    d, prob = tf.cond(tf.equal(success, True),
    #               true_fn=lambda: (d, prob),
    #               false_fn=lambda: asymp_ks_2samp(n1, n2, d, alternative, precision))
    #
    #    
    #    return d, prob
    #
    #d, prob = tf.cond(tf.equal(mode, 1),
    #                      true_fn = lambda: attempt_exact_ks_2samp(n1, n2, g, d, alternative, precision),
    #                      false_fn = lambda: asymp_ks_2samp(n1, n2, d, alternative, precision)) # type: ignore
    
    d, prob = asymp_ks_2samp(n1, n2, d, alternative, precision)

    prob = tf.clip_by_value(prob, 0, 1) # type: ignore
    
    return d, prob, d_location
