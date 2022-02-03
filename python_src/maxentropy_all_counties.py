# This method runs the max entropy optimization after obtaining prior set of probabilities
# from the copula sampling and conditional probabilities algorithm

import pandas as pd
import os
import json
import statistics
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import copy
from get_puma import get_pums
from dython.nominal import associations
import numpy as np

np.random.seed(7)

# List of variables to constraint the data upon
constraint_list = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status', 'drug_overdose', 'Insurance', 'Veteran']
# List of variables that are in ACS
req_columns = ["age", "gender", "marital_status", "pov_status", "emp_status",
               "Insurance", "edu_attain"]

# Read the necessary data
data_path = os.path.join(".", "data")
read_top_50 = pd.read_csv(os.path.join(data_path, "top_50_od.csv"))
county_pums = pd.read_csv(os.path.join(data_path, "puma_county.csv"))


# Function for finding weights (wi) in each iteration of the max entropy model
def weight_intermediate(theta, data, total_fc):
    theta_f_repeat = np.tile(theta, (len(data), 1))
    sum_j = np.sum(total_fc * theta_f_repeat, axis=1)
    exp_j = np.exp(sum_j - 1)
    return exp_j


def find_weights_entropy_conditional(theta, u_i, data, total_fc):
    exp_j = weight_intermediate(theta, data, total_fc)
    w_i = u_i * exp_j
    return w_i


# Function that implements the max entropy loss function
def max_entropy(theta, u_i, data, total_fc):
    global t
    global f_values
    global weight_values
    print('Iteration', t, flush=True)
    theta_repeat = np.tile(theta, (len(data), 1))
    sum_j = np.sum(total_fc * theta_repeat, axis=1)
    exp_j = np.exp(sum_j - 1)
    second_term = sum(exp_j * u_i)
    first_term = sum(mew_j * theta)
    f = - first_term + second_term
    weights = find_weights_entropy_conditional(theta, u_i, data, total_fc)
    sum_weights = sum(weights)
    print("Sum of weights:", sum_weights, flush=True)
    weight_values.append(sum_weights)
    t += 1
    print("Value of f:", f)
    f_values.append(f)
    return f


# Finding the jacobian matrix of the loss function
def jacobian_m(theta, u_i, data, total_fc):
    j_jac = []
    for j in range(len(mew_j)):
        theta_repeat = np.tile(theta, (len(data), 1))
        sum_j = np.sum(total_fc * theta_repeat, axis=1)
        exp_j = np.exp(sum_j - 1)
        f_j_ = total_fc[:, j]
        second_term = sum(exp_j * u_i * f_j_)
        first_term = mew_j[j]
        f = -first_term + second_term
        j_jac.append(f)
    return np.array(j_jac)


# Fit the maximum entropy model
def fit(u_i, dt, total_fc, theta_0):
    data = dt.copy()
    global t
    global weight_values
    global f_values
    t = 1
    weight_values = []
    f_values = []
    res = minimize(max_entropy, jac=jacobian_m, x0=theta_0, args=(u_i, data, total_fc), method='L-BFGS-B',
                   options={'disp': True})
    theta_f = res.x
    print("Final theta", theta_f, flush=True)
    f_values = np.array(f_values)
    f_values = np.nan_to_num(f_values)
    final_weights = find_weights_entropy_conditional(theta_f, u_i, data, total_fc)
    data["w"] = final_weights
    data["weights"] = np.array([round(i * total) for i in final_weights])
    final_population = data.reindex(data.index.repeat(data.weights))
    return final_population, data


# Calculations related to the constraints - check if each row has the required value or not
def constraints(data):
    total_f_c = []
    for ind, i in data.iterrows():
        f_c = np.zeros(total_constraint)
        val = 0
        for con in constraint_list:
            for j in dict_constraint[con]:
                if i[con] == j:
                    f_c[val] = 1
                val += 1
        f_c[val] = 1
        total_f_c.append(f_c)
    total_f_c = np.array(total_f_c)
    return total_f_c


# Calculate the total absolute error based on the constraints
def tae_calculate(data):
    predicted_values = []
    for con in constraint_list:
        for j in dict_constraint[con]:
            dat = data[data[con] == j]
            predicted_values.append(len(dat))
    print(mew_j_copy)
    print(predicted_values)
    difference = np.sum(np.absolute(np.array(mew_j_copy)[:-1] - np.array(predicted_values)))
    return difference


# Calculate the KL divergence
def kl(p, q):
    vec = scipy.special.rel_entr(p, q)
    kl_div = np.sum(vec)
    return kl_div


# Association plots
def association_plot(df, name):
    fig, ax = plt.subplots(figsize=(10, 20))
    varLabels = ["age", "gender", "marital_sts", "pov_sts", "emp_sts",
                 "Insurance", "edu"]
    assoc = associations(df, ax=ax, plot=False, cbar=False)
    ax.set_xticks(np.arange(0, 7))
    ax.set_yticks(np.arange(0, 7))
    ax.set_xticklabels(varLabels, rotation=45)
    ax.set_yticklabels(varLabels, rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(data_path, 'Categorical Correlation matrix - ' + name))
    corr = assoc["corr"]
    corr.to_csv(os.path.join(data_path, 'Categorical Correlation matrix - ' + name + "_.csv"))

    flatten_assoc = corr.values.flatten()
    return flatten_assoc


# Filter the joint probabilities based on tau and send it to the max entropy model for optimization
def prior_calculations(prior, tau):
    prior_cp = prior.copy()
    group_by_conditional = prior_cp.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
    county_prior_data = prior[prior["p"] >= tau].reset_index(drop=True)
    county_prior_cp = county_prior_data.copy()
    county_prior_cp["weights"] = np.array([round(i) for i in county_prior_cp['p'].to_numpy() * total])
    prior_intermediate = county_prior_cp.reindex(county_prior_cp.index.repeat(county_prior_cp.weights))

    u_i_county = county_prior_data["p"].to_numpy()
    total_fc_county = constraints(county_prior_data)
    final_county_population_entropy_conditional, data_en_conditional = fit(u_i_county, county_prior_data,
                                                                           total_fc_county, theta_0)
    return final_county_population_entropy_conditional, data_en_conditional, group_by_conditional, prior_intermediate


# Distance between association matrices
def matrix_distance(a, b):
    temp = a - b
    dist = np.sqrt(np.dot(temp.T, temp))
    return dist


# Iterate through the counties
for ind, row in read_top_50.iterrows():
    fips = row["County.Code"]
    county_name = row["County"]
    state_name = row["State_name"].strip()

    # Get priors from copula and conditional probabilities
    copula_prior = pd.read_csv(os.path.join(data_path, "dataprior_copula_" + str(fips) + ".csv"))
    prior_copula = copula_prior.iloc[:, 1:]

    conditional_prior = pd.read_csv(os.path.join(data_path, "prior_" + str(fips) + "_" + county_name + ".csv"))

    # Combine both probabilities
    prior_all = pd.merge(conditional_prior, prior_copula, on=constraint_list, how="outer")
    prior_all = prior_all.fillna(0)
    prior_all["p"] = (prior_all["p_x"] + prior_all["p_y"]) / 2
    prior_all["p"] = prior_all["p"] / sum(prior_all["p"])
    prior_all = prior_all[['age', 'gender', 'edu_attain',
                           'marital_status', 'pov_status', 'emp_status', 'drug_overdose', 'Insurance', 'Veteran', 'p']]

    # get constraints
    with open(os.path.join(data_path, "json_constraint_" + str(fips) + "_" + county_name + ".json")) as json_data:
        j_data = json.load(json_data)
    dict_constraint = {}
    j_data = json.loads(j_data[0])
    total_sum = []
    for k in constraint_list:
        list_keys = list(j_data[k].keys())
        list_values = list(j_data[k].values())
        list_values = [i[0] for i in list_values]
        total_sum.append(sum(list_values))
        dict_constraint[k] = dict(zip(list_keys, list_values))
    total = statistics.mode(total_sum)
    total_constraint = 0

    # Define mew_j (constraints to satisfy)
    mew_j = []
    for c in constraint_list:
        value_c = dict_constraint[c].values()
        total_constraint += len(value_c)
        mew_j.extend(value_c)
    total_constraint += 1

    mew_j.extend([total])
    mew_j_copy = copy.deepcopy(mew_j)
    mew_j = [i / total for i in mew_j]

    theta_0 = (np.random.rand(total_constraint))
    print("Initial theta:", theta_0)

    tau = 1 / total

    # Fit a maximum entropy model for both conditional probabilities and copula as prior
    final_county_population_combined, data_combined, group_by_combined, combined_intermediate = prior_calculations(
        prior_all, tau)

    # Rename prior probabilities column to avoid conflict
    group_by_combined = group_by_combined.rename({'p': 'p_combined'}, axis=1)

    # Find tae for the final populations
    tae_copula_conditional_final = tae_calculate(final_county_population_combined)

    # copula conditional
    conditional_combined = data_combined.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
    conditional_combined["cond_combined"] = conditional_combined["w"]

    # Read sample data from ACS
    actual_file = os.path.join(data_path, "actual_" + str(fips) + "_" + county_name + "_acs.csv")
    if os.path.exists(actual_file):
        print("File_exists")
        actual_data = pd.read_csv(actual_file)
    else:
        actual_data = get_pums(state_name, str(fips), county_pums)
        if len(actual_data) == 0:
            print("Length of file is 0")
            print(actual_data)
            actual_data = pd.DataFrame(columns=req_columns)
        actual_data.to_csv(actual_file, index=False)

    combined_file = os.path.join(data_path, "Combined_" + str(fips) + "_" + county_name + "_.csv")
    final_county_population_combined.to_csv(combined_file, index=False)

    # Check if the data for that county is available in PUMS. Proceed only if it is available
    if len(actual_data) != 0:
        group_by_actual = actual_data.groupby(req_columns, as_index=False).size().reset_index(drop=True)
        group_by_actual["Actual_prob"] = group_by_actual["size"] / sum(group_by_actual["size"])

        # Merge the distributions returned from the sample and max entropy optimization
        combine_all = pd.merge(group_by_combined, group_by_actual,
                               on=['age', 'gender', 'marital_status', 'emp_status', 'Insurance',
                                   'edu_attain', 'pov_status'], how='left')
        combine_all = pd.merge(combine_all, conditional_combined,
                               on=['age', 'gender', 'marital_status', 'emp_status', 'Insurance',
                                   'edu_attain', 'pov_status'], how='left')

        # Replace na values with 1/total (ensure that the profiles occur atleast once to avoid
        # infinite KL divergence values)
        combine_all = combine_all.fillna(1 / total)
        combine_all[["cond_combined", "Actual_prob"]] = combine_all[
            ["cond_combined", "Actual_prob"]].apply(lambda x: x / sum(x))

        kl_combined_final = kl(combine_all["Actual_prob"], combine_all["cond_combined"])
        association_combined_prior = association_plot(combined_intermediate[req_columns],
                                                      "Combined_cp-prior " + str(fips))
        association_actual = association_plot(
            actual_data[req_columns],
            "Actual " + str(fips))
        # Frobenius norm
        dist_combined_prior = matrix_distance(association_combined_prior, association_actual)
