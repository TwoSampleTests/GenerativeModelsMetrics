# GenerativeModelsMetrics
Non-parametric two sample tests in Numpy/Scipy and TensorFlow2

Overleaf: https://www.overleaf.com/project/6513e9417b566d2c4e32709a

The module supports the following non-parammetric two sample test statistics:

    - Dimension averaged Kolmogorov-Smirnov (KS) test-statistic and p-value
    - Sliced KS test-statistic
    - Multivariate KS test-statistic
    - Sliced Wasserstein Distance (SWD) test-statistic
    - Frobenius norm of the difference between the correlation matrices (FN) test-statistic

The module also includes the parametric test-statistic for the likelihood ratio 
(computed from the symbolic PDFs), that can be used as proxy of "best estimator" by the 
Neyman-Pearson lemma.

The module includes the following files and folders:

    - __init__.py 
        Package initialization

    - base.py
        Base classes for input data parsing and results

    - fn_metrics.py
        Implementation of the FN metric

    - ks_metrics.py
        Implementation of the KS metric

    - ls_metrics.py
        Implementation of the Likelihood-ratio metric

    - multiks_metrics.py
        Implementation of the MultiKS metric

    - sks_metrics.py
        Implementation of the SKS metric

    - swd_metrics.py
        Implementation of the SKS metric

    - ultils.py
        File including utility functions

    - notebooks/
        Folder including sample Jupyter notebooks

    - utils_func/
        Old implementation of the metrics as functions needed for some backward compatibility
        It also includes the file
    
    - utils_func/MixtureDistribution.py
        File including different definitions of mixture of Gaussians tensorflow-probability distributions