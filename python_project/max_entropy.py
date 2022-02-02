import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import copy
from dython.nominal import associations
from sklearn.preprocessing import OneHotEncoder


county = "Baltimore city"
data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code\\data"

county_prior = pd.read_csv(os.path.join(data_path, "prior_"+county+".csv"))
# county_prior = pd.read_csv(os.path.join(data_path,"prior_baltimore_city.csv"))
constraint_list = ['age', 'gender', 'edu_attain',
                           'marital_status', 'pov_status', 'emp_status', 'Insurance', 'drug_overdose']


dict_ = {'age': {'under15': 148445, '15_17': 30309, '18_24': 74762, '25_29': 58490, '30_34': 56258,
                     '35_39': 52199, '40_44': 48153, '45_49': 52765, '50_54': 57324, '55_59': 58544, '60_64': 54047,
                     '65_69': 44609, '70_74': 30327, '75_79': 24135, '80_84': 16496, '85up': 20762},
             'gender': {'Male': 392421, 'Female': 435204},
             'edu_attain': {'lt_hs': 168206, 'some_hs': 68203, 'hs_grad': 174688, 'some_col': 140417,
                            'assoc_dec': 44576,
                            'ba_deg': 136818, 'grad_deg': 94717},
             'pov_status': {'below_pov_level': 98146, 'at_above_pov_level': 729479},
             'emp_status': {'not_in_labor_force': 374348, 'employed': 419870,
                            'unemployed': 33407},
             'geog_mobility': {'same house': 13452, 'same county': 140752, 'same state': 48644,
                               'diff state': 600033, 'moved from abroad': 24744},
             'nativity': {'born_other_state': 179101, 'born_out_us': 536007, 'born_state_residence': 9541,
                          'foreigner': 102976},
             'drug_overdose': {'Drug overdose': 386, 'No overdose': 827239},
             'race': {'black, afr amer': 48392, 'native amer': 230356, 'asian': 42659,
                      'pacific islander': 2219, 'two or more races': 10118, 'white alone': 22478,
                      'hispanic': 471403},
             'marital_status': {'never_mar': 381187, 'married': 294883, 'mar_apart': 34115, 'widowed': 46899,
                                'divorced': 70541},
             'ind_income': {'10k_lt15k': 129260,
                            '15k_lt25k': 62781, '25k_lt35k': 109939, '35k_lt50k': 100394, '50k_lt65k': 85843,
                            '65k_lt75k': 73651,
                            'gt75k no_income': 32839, '1_lt10k': 122914},
             'Insurance': {'Insured': 776177, 'Uninsured': 51448}}

if county == "Baltimore city":
    with open(os.path.join(data_path,"test_bc.json")) as json_data:
        j_data = json.load(json_data)

    dict_constraint = {}
    for k in constraint_list:
        list_keys = list(dict_[k].keys())
        list_values = list(j_data[k])
        dict_constraint[k] = dict(zip(list_keys, list_values))

    total = 614700
elif county == "Baltimore County":
    dict_constraint = dict_
    total = 827625

total_constraint = 0

mew_j = []

for c in constraint_list:
    value_c = dict_constraint[c].values()
    total_constraint += len(value_c)
    mew_j.extend(value_c)
total_constraint += 1

mew_j.extend([total])
mew_j_copy = copy.deepcopy(mew_j)
mew_j = [i/total for i in mew_j]


def weight_intermediate(theta, data, total_fc):
    theta_f_repeat = np.tile(theta, (len(data), 1))
    sum_j = np.sum(total_fc * theta_f_repeat, axis=1)
    exp_j = np.exp(sum_j - 1)
    return exp_j


def find_weights_entropy(theta, data, total_fc):
    return weight_intermediate(theta, data, total_fc)


def find_weights_entropy_conditional(theta,u_i, data, total_fc):
    exp_j = weight_intermediate(theta, data, total_fc)
    w_i = u_i * exp_j
    return w_i


def max_entropy(theta, u_i, data, total_fc, method):
    import math
    global t
    global f_values
    global weight_values
    print('Iteration', t)
    theta_repeat = np.tile(theta, (len(data), 1))
    sum_j = np.sum(total_fc * theta_repeat, axis=1)
    exp_j = np.exp(sum_j-1)
    if method == "entropy":
        second_term = sum(exp_j)
        first_term = sum(mew_j * theta)
        f = - first_term + second_term
        weights = find_weights_entropy(theta, data, total_fc)
        sum_weights = sum(weights)
    elif method == "entropy_conditional":
        second_term = sum(exp_j * u_i)
        first_term = sum(mew_j * theta)
        f = - first_term + second_term
        weights = find_weights_entropy_conditional(theta,u_i, data, total_fc)
        sum_weights = sum(weights)
    else:
        print("Error!!! Please specify the method you want to use to compute the weights.")
        sys.exit(1)
    print("Sum of weights:", sum_weights)
    weight_values.append(sum_weights)
    t += 1
    print("Value of f:", f)
    f_values.append(f)
    return f


def fit(u_i, data, total_fc, theta_0, method):
    global t
    global weight_values
    global f_values
    t = 1
    weight_values = []
    f_values = []
    res = minimize(max_entropy, x0 = theta_0, args = (u_i, data, total_fc, method), method='BFGS', options={'disp': True})
    theta_f = res.x
    print("Final theta", theta_f)
    f_values = np.array(f_values)
    f_values = np.nan_to_num(f_values)
    if method =="entropy":
        final_weights = find_weights_entropy(theta_f, data, total_fc)
    else:
        final_weights = find_weights_entropy_conditional(theta_f, u_i, data, total_fc)
    data["weights"] = np.array([round(i*total) for i in final_weights])
    final_population = data.reindex(data.index.repeat(data.weights))
    return final_population


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


def tae_calculate(data):
    predicted_values = []
    for con in constraint_list:
        for j in dict_constraint[con]:
            dat = data[data[con] == j]
            predicted_values.append(len(dat))
    predicted_values.append(len(data))
    print(predicted_values)
    print(mew_j_copy)
    difference = np.sum(np.absolute(np.array(mew_j_copy) - np.array(predicted_values)))
    return difference


def main():

    # Take data corresponding to top 10000 of the probabilities. Because higher the data, lower will be the efficiency
    # of the max entropy algorithm because of difficulty of convergence
    county_prior_data = county_prior.nlargest(10000, ['p']).reset_index(drop=True)
    # Get the immediate representation of the county to see what we get if we directly consider \
    # priors to be the population
    county_prior_cp = county_prior_data.copy()
    county_prior_cp["weights"] = np.array([round(i) for i in county_prior_cp['p'].to_numpy() * total])
    print(county_prior_cp)

    prior_intermediate = county_prior_cp.reindex(county_prior_cp.index.repeat(county_prior_cp.weights))
    tae_intermediate_prior = tae_calculate(prior_intermediate)
    print(tae_intermediate_prior)

    u_i_county = county_prior_data["p"].to_numpy()
    total_fc_county = constraints(county_prior_data)

    theta_0 = (np.random.rand(total_constraint))
    print("Initial theta:", theta_0)

    final_county_population_entropy = fit(u_i_county, county_prior_data, total_fc_county, theta_0, "entropy")
    final_county_population_entropy_conditional = fit(u_i_county, county_prior_data, total_fc_county, theta_0,
    "entropy_conditional")
    tae_county_entropy = tae_calculate(final_county_population_entropy)
    tae_county_entropy_conditional = tae_calculate(final_county_population_entropy_conditional)

    print("Total TAE with just the conditional probabilities method:", tae_intermediate_prior)
    print("Total TAE with the maximum entropy approach:", tae_county_entropy)
    print("Total TAE with BOTH the maximum entropy and conditional probabilities approach:",
    tae_county_entropy_conditional)

    def association_plot(df, name):
        fig, ax = plt.subplots(figsize=(10, 20))
        varLabels = ["age", "gender", "marital_sts", "pov_sts", "emp_sts",
                     "Insurance", "edu"]

        associations(df, nominal_columns='all', ax=ax, plot=False)
        ax.set_xticks(np.arange(0, 7))
        ax.set_yticks(np.arange(0, 7))
        ax.set_xticklabels(varLabels, rotation=45)
        ax.set_yticklabels(varLabels, rotation=45)
        fig.savefig('Categorical Correlation matrix - ' + name)


    association_plot(prior_intermediate[["age", "gender", "marital_status","pov_status", "emp_status",
                                                       "Insurance", "edu_attain"]],
                     "Conditional probabilities "+county)
    association_plot(final_county_population_entropy[["age", "gender", "marital_status","pov_status", "emp_status",
                                                       "Insurance", "edu_attain"]], "Maximum entropy "+county)
    association_plot(final_county_population_entropy_conditional[["age", "gender", "marital_status","pov_status", "emp_status",
                                                       "Insurance", "edu_attain"]], "Entropy + Conditional "+county)
    actual_baltimore = pd.read_csv(os.path.join(data_path, "categorical_baltimore_city.csv"))
    association_plot(actual_baltimore[["age", "gender", "marital_status","pov_status", "emp_status", "Insurance", "edu_attain"]],
                     "Actual "+county)


# main()
def association_plot(df, name):
    fig, ax = plt.subplots(figsize = (10,20))
    varLabels = ["age", "gender", "marital_sts","pov_sts", "emp_sts",
                                                       "Insurance", "edu"]

    associations(df, nominal_columns='all', ax=ax,plot=False)
    ax.set_xticks(np.arange(0, 7))
    ax.set_yticks(np.arange(0, 7))
    ax.set_xticklabels(varLabels, rotation = 45)
    ax.set_yticklabels(varLabels, rotation = 45)
    plt.show()
    # ax.set_xticklabels(ax.get_xticks(), rotation=90)


    # fig.savefig('Categorical Correlation matrix - ' + name)


actual_baltimore = pd.read_csv(os.path.join(data_path, "categorical_baltimore_County.csv"))
print(actual_baltimore)

dummies_df = pd.get_dummies(actual_baltimore)
print(dummies_df)
print(actual_baltimore["emp_status"].value_counts())
# associations(dummies_df[["pov_status_below_pov_level", "emp_status_unemployed","Insurance_Insured","edu_attain_lt_hs"]])

import seaborn as sns
# corr= dummies_df[["pov_status_below_pov_level", "emp_status_unemployed","Insurance_Uninsured","edu_attain_lt_hs"]].corr(method ='pearson')
# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values, annot=True)
# plt.show()

cov = dummies_df[["pov_status_below_pov_level", "emp_status_unemployed","Insurance_Insured","edu_attain_lt_hs"]].cov()
sns.heatmap(cov,
            xticklabels=cov.columns.values,
            yticklabels=cov.columns.values, annot=True)
plt.show()
print(cov)
# association_plot(actual_baltimore[["age", "gender", "marital_status", "pov_status", "emp_status", "Insurance",
# "edu_attain"]],   "Correlation matrix - Actual Baltimore county")
#
