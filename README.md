# SynCop: Using Gaussian Copula and Conditional Probabilities forGenerating Synthetic Microdata from Macro Data Sources

The overall framework of SynCop is presented here:
![image](https://user-images.githubusercontent.com/30096335/152444600-4f26aa8d-33e2-46ac-9974-f7c1dd53078f.png)

As shown, the framework consists of three important components: conditional probabilties, copula sampling, and maximum entropy optimization.
This repository contains an implementation of SynCop and is based on both R and Python. R is used for implementing Conditional Probabilities and Copula Sampling. And, python is used for maximum entropy optimization using L-BFGS-B. The required dependencies and the corresponsing source codes for these frameworks are detailed separately:

# Conditional Probabilties
The conditional probabilities method is implemented on top of the SynthACS framework. Therefore, the details on the functions implemented in the code can be found in [1]. We implement the conditional probabilties method for two different datasets (i) One that only consists of ACS variables - provided in r_src/conditional_probabilities_acs.R (ii) One that combines ACS variables with variables from other sources - provided in r_src/conditional_probabilities_combined.R. 

## Depedendencies:
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
The maximum entropy optimization is implemented in python. Here, python_src/maxentropy_all_counties.py and python_src/maxentropy_all_counties_acs.py represent the codes for optimizing using maximum entropy for the two datasets. python_src/get_puma.py and python_src/get_puma_acs.py are for obtaining corresponsing pums sample data.

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
The demographic data are obtained from the American Community Survey (ACS) using the SythACS framework. Data on opioid overdose is obtained from CDC Wonder, data on insurance is obtained from Small Area Health Estimates, and data on veteran population is obtained from Opioid Environment Policy Scan (OEPS) [5]. 

# REFERENCES
[1] Whitworth A (Forthcoming). “synthACS: Spatial MicroSimulation Modeling with Synthetic American Community Survey Data.” Journal of Statistical Software.

[2] Wan, C., Li, Z., Guo, A., and Zhao, Y., "SynC: A Unified Framework for Generating Synthetic Population with Gaussian Copula," Workshops at the Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020. Accepted, to appear.

[5] Marynia Kolak, Qinyun Lin, Susan Paykin, Moksha Menghaney, & Angela Li. (2021, May 11). GeoDaCenter/opioid-policy-scan: Opioid Environment Policy Scan Data Warehouse (Version v0.1-beta). Zenodo. http://doi.org/10.5281/zenodo.4747876
