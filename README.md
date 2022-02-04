# SynCop: Using Gaussian Copula and Conditional Probabilities for Generating Synthetic Microdata from Macro Data Sources

The overall framework of SynCop is depicted here:
![image](https://user-images.githubusercontent.com/30096335/152444600-4f26aa8d-33e2-46ac-9974-f7c1dd53078f.png)

As shown, the framework consists of three main components: conditional probabilties, copula sampling, and maximum entropy optimization.
This repository contains an implementation of SynCop and is based on both R and Python. R is used for implementing Conditional Probabilities and Copula Sampling. And, python is used for maximum entropy optimization using the L-BFGS-B algorithm. The required dependencies and the corresponding source codes of these methods are detailed separately:

# Conditional Probabilities
The conditional probabilities method is implemented on top of the SynthACS framework. Therefore, the details on the functions implemented in the code can be found in [1]. We implement the conditional probabilties method for two different datasets (i) One that only consists of ACS variables - provided in r_src/conditional_probabilities_acs.R (ii) One that combines ACS variables with variables from other sources - provided in r_src/conditional_probabilities_combined.R. 

## Dependencies:
* data.table
* synthACS
* dplyr
* acs
* purrr
* stringr
* hash
* jsonlite

# Gaussian copula sampling
The Gaussian copula sampling code is motivated by [2]. As with the conditional probabilties method, it consists of codes for two different datasets and are provided in r_src/copula_sampling_acs.R and r_src/copula_sampling_combined.R.

## Dependencies:
* mvtnorm
* fitdistrplus
* dplyr

# Maximum entropy optimization
The maximum entropy optimization is implemented in python. Here, python_src/maxentropy_all_counties.py and python_src/maxentropy_all_counties_acs.py represent the codes for optimization using maximum entropy for the two datasets. python_src/get_puma.py and python_src/get_puma_acs.py are implemented for obtaining corresponding pums sample data.

## Dependencies:
* statistics
* scipy
* matplotlib
* scipy
* dython
* numpy
* pypums
* pandas
* os
* copy

# Dataset
The demographic data is obtained from the American Community Survey (ACS) using the SythACS framework. Data on opioid overdose is obtained from CDC Wonder [3], data on insurance is obtained from Small Area Health Insurance Estimates (SAHIE) [4], and data on veteran population is obtained from Opioid Environment Policy Scan (OEPS) [5]. 

All the necessary data are adjusted according to the required formats and are made available inside the "data" directory

# REFERENCES
[1] Whitworth A (Forthcoming). “synthACS: Spatial MicroSimulation Modeling with Synthetic American Community Survey Data.” Journal of Statistical Software.

[2] Wan, C., Li, Z., Guo, A., and Zhao, Y., "SynC: A Unified Framework for Generating Synthetic Population with Gaussian Copula," Workshops at the Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020. Accepted, to appear.

[3] Wide-ranging online data for epidemiologic research (WONDER). Atlanta, GA: CDC, National Center for Health Statistics; 2020. Available at http://wonder.cdc.gov

[4] US Census Bureau. (2021, October 8). 2008 - 2019 Small Area Health Insurance Estimates (SAHIE) using the American Community Survey (ACS) [Dataset]. https://www.census.gov/data/datasets/time-series/demo/sahie/estimates-acs.html

[5] Marynia Kolak, Qinyun Lin, Susan Paykin, Moksha Menghaney, & Angela Li. (2021, May 11). GeoDaCenter/opioid-policy-scan: Opioid Environment Policy Scan Data Warehouse (Version v0.1-beta). Zenodo. http://doi.org/10.5281/zenodo.4747876
