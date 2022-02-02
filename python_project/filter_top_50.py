import pandas as pd
import os
import numpy as np
import json
import statistics
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import copy
from dython.nominal import associations
from sklearn.preprocessing import OneHotEncoder
from get_puma import get_pums
from csv import DictWriter
from pypums import ACS


data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data"


# read_top_50 = pd.read_csv(os.path.join(data_path, "top_50_od.csv"))
read_od = pd.read_csv(os.path.join(data_path, "Overdose_by_age_all_counties.csv"))
county_pums = pd.read_csv(os.path.join(data_path, "puma_county.csv"))

print(read_od)
