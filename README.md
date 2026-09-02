# parametricHawkes_OLS

Matlab implementation of parametric Hawkes process by ordinary least squares model based on the paper:

*TBA* by Benjamin Poignard and Yoann Potiron.

Link: TBA

# Overview

The code in this replication includes:

- The different DGP processes considered in the simulated experiments: the replicator should execute programs *runcode_sim_dist* (illustration of the asymptotic distribution), *runcode_l2error.m* ($\ell_2$-error), *runcode_test.m* (illustration of the Wald test statistic).
- The real data experiment (transactions): the replicator should execute program *runcode_fitR2.m* (R-square analysis), *runcode_prediction.m* (out-of-sample prediction of transactions), *runcode_test.m* (Wald test for trader heterogeneity).

# Data availability

The data correspond to WRDS real financial data (requiring a license). The data will thus not be publicly distributed.

# Software requirements

The Matlab code was run on a Windows-Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz (3.09 GHz) and 128 GB Memory. The version of the Matlab software on which the code was run is a follows: 23.2.0.2859533 (R2023b) Update 10.

The following toolboxes should be installed:

- Global Optimization Toolbox, Version 23.2
- Optimization Toolbox, Version 23.2
- Parallel Computing Toolbox, Version 23.2

The Parallel Computing Toolbox is recommended to run the code in the case of 100 batches-based experiments.
