import numpy as np
from jetnet.evaluation import gen_metrics as JMetrics

# Mean and covariance of the distribution
mean = [0, 0]
cov = [[1, 0.25], [0.25, 1]]

# Number of samples to draw
nsamples = 50000

# Generate random samples
s1 = np.random.multivariate_normal(mean, cov, nsamples)
s2 = np.random.multivariate_normal(mean, cov, nsamples)

# Compute values of FGD for different values of samples/nmax
for max_samples in [25_000,30_000,35_000,40_000,45_000,50_000]:
    fgd = JMetrics.fgd(real_features = s1[:max_samples],
                       gen_features = s2[:max_samples],
                       min_samples = 20_000,
                       max_samples = max_samples,
                       num_batches = 20,
                       num_points = 10,
                       normalise = True,
                       seed = 42)
    print(f"nmax = {max_samples} -> fgd = {fgd}")
    
# Compute values of MMD for different values of batch_size
for batch_size in [100,1_000,1_500,2_000,3_000,5_000]:
    mmd = JMetrics.mmd(real_features = s1,
                       gen_features = s2,
                       num_batches = 10,
                       batch_size = batch_size,
                       normalise = True,
                       seed = 42,
                       num_threads = None)
    print(f"batch_size = {batch_size} -> mmd = {mmd}")