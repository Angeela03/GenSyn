import pandas as pd
import os
import json
import statistics
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import copy
# from dython.nominal import associations
# from sklearn.preprocessing import OneHotEncoder
from get_puma import get_pums
from csv import DictWriter
from pypums import ACS
from dython.nominal import associations
import numpy as np

np.random.seed(7)

constraint_list = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status']
req_columns = ['age', 'gender', 'marital_status', 'emp_status', 'edu_attain', 'pov_status']

data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data"
# data_path = "/scratch/aachary/conditional_rpobabilties_generic/data"
read_top_50 = pd.read_csv(os.path.join(data_path, "top_50_od.csv"))
county_pums = pd.read_csv(os.path.join(data_path, "puma_county-DESKTOP-FIEPTEH.csv"))
read_copula_prior = pd.read_csv(os.path.join(data_path, "prior_copula.csv"))

print(county_pums)

# print(list(read_top_50["County"].values))


def weight_intermediate(theta, data, total_fc, cost):
    theta_f_repeat = np.tile(theta, (len(data), 1))
    cost_repeat = np.tile(cost, (len(data), 1))
    sum_j = np.sum(total_fc * theta_f_repeat * cost_repeat, axis=1)
    exp_j = np.exp(sum_j - 1)
    return exp_j


def find_weights_entropy(theta, data, total_fc, cost):
    return weight_intermediate(theta, data, total_fc, cost)


def find_weights_entropy_conditional(theta, u_i, data, total_fc, cost):
    exp_j = weight_intermediate(theta, data, total_fc, cost)
    w_i = u_i * exp_j
    return w_i


def max_entropy(theta, u_i, data, total_fc, method, cost):
    global t
    global f_values
    global weight_values
    print('Iteration', t, flush=True)
    theta_repeat = np.tile(theta, (len(data), 1))
    cost_repeat = np.tile(cost, (len(data), 1))
    sum_j = np.sum(total_fc * theta_repeat * cost_repeat, axis=1)
    exp_j = np.exp(sum_j - 1)
    if method == "entropy":
        second_term = sum(exp_j)
        first_term = sum(mew_j * theta * cost)
        f = - first_term + second_term
        weights = find_weights_entropy(theta, data, total_fc, cost)
        sum_weights = sum(weights)
    elif method == "entropy_conditional":
        second_term = sum(exp_j * u_i)
        first_term = sum(mew_j * theta * cost)
        f = - first_term + second_term
        weights = find_weights_entropy_conditional(theta, u_i, data, total_fc, cost)
        sum_weights = sum(weights)
    else:
        print("Error!!! Please specify the method you want to use to compute the weights.")
        sys.exit(1)
    print("Sum of weights:", sum_weights, flush=True)
    weight_values.append(sum_weights)
    t += 1
    print("Value of f:", f)
    f_values.append(f)
    return f


def jacobian_m(theta, u_i, data, total_fc, method, cost):
    j_jac = []
    for j in range(len(mew_j)):
        theta_repeat = np.tile(theta, (len(data), 1))
        sum_j = np.sum(total_fc * theta_repeat, axis=1)
        exp_j = np.exp(sum_j - 1)
        f_j_ = total_fc[:, j]

        if method == "entropy":
            second_term = sum(exp_j * f_j_)

        elif method == "entropy_conditional":
            second_term = sum(exp_j * u_i * f_j_)

        first_term = mew_j[j]
        f = -first_term + second_term
        j_jac.append(f)
    return np.array(j_jac)


def fit(u_i, dt, total_fc, theta_0, method, cost):
    data = dt.copy()
    global t
    global weight_values
    global f_values
    t = 1
    weight_values = []
    f_values = []
    res = minimize(max_entropy, jac=jacobian_m, x0=theta_0, args=(u_i, data, total_fc, method, cost), method='L-BFGS-B',
                   options={'disp': True})
    theta_f = res.x
    print("Final theta", theta_f, flush=True)
    f_values = np.array(f_values)
    f_values = np.nan_to_num(f_values)
    if method == "entropy":
        final_weights = find_weights_entropy(theta_f, data, total_fc, cost)
    elif method == "entropy_conditional":
        final_weights = find_weights_entropy_conditional(theta_f, u_i, data, total_fc, cost)
    data["w"] = final_weights
    data["weights"] = np.array([round(i * total) for i in final_weights])
    # data["weights"] = np.array([round(i) for i in final_weights])

    final_population = data.reindex(data.index.repeat(data.weights))
    return final_population, data


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
    # predicted_values.append(len(data))
    print(mew_j_copy)
    print(predicted_values)
    difference = np.sum(np.absolute(np.array(mew_j_copy)[:-1] - np.array(predicted_values)))
    return difference


def plot(list_dfs):
    len_dfs = len(list_dfs)
    fig, axs = plt.subplots(2, int(len_dfs / 2), figsize=(20, 20))
    plt.subplots_adjust(hspace=1)
    fig.suptitle('Actual vs Observed values for each attribute')
    list_i_j = [(i, j) for i in range(2) for j in range(int(len_dfs / 2))]
    for index in range(len(list_dfs)):
        i = list_i_j[index][0]
        j = list_i_j[index][1]
        list_dfs[index].plot(kind='bar', ax=axs[i][j], title=list_dfs[index][""].iloc[0], width=0.9)
    for ax in axs.reshape(-1):
        for y in ax.patches:
            ax.text(y.get_x() + y.get_width() / 4, y.get_height() * 1.05, f'{y.get_height():.1f}')
            ax.set_ylim(0, ax.get_ylim()[1] + 10)
    plt.show()
    plt.savefig("Results.png")


def plot_separate(list_dfs):
    for index in range(len(list_dfs)):
        ax = list_dfs[index].plot(kind='bar', title=list_dfs[index]["Attribute"].iloc[0], width=1, figsize=(10, 10))
        for y in ax.patches:
            ax.text(y.get_x() + y.get_width() / 4, y.get_height() * 1.05, f'{y.get_height():.1f}', fontsize=6)
            ax.set_ylim(0, ax.get_ylim()[1] + 10)
        plt.show()


def kl(p, q):
    vec = scipy.special.rel_entr(p, q)
    kl_div = np.sum(vec)
    return kl_div


def association_plot(df, name):
    fig, ax = plt.subplots(figsize=(10, 20))
    varLabels = ["age", "gender", "marital_sts", "pov_sts", "emp_sts", "edu"]
    associations(df, ax=ax, plot=False, cbar=False)
    ax.set_xticks(np.arange(0, 6))
    ax.set_yticks(np.arange(0, 6))
    ax.set_xticklabels(varLabels, rotation=45)
    ax.set_yticklabels(varLabels, rotation=45)
    # plt.show()
    # ax.set_xticklabels(ax.get_xticks(), rotation=90)
    fig.tight_layout()
    fig.savefig('Categorical Correlation matrix - ' + name)
    return 0


def get_max_tae(county_prior_data,prior_cp, simulated, total_constraint):
    # Define for syntropy optimization
    u_i_county = county_prior_data["p"].to_numpy()
    total_fc_county = constraints(county_prior_data)

    # Define for max entropy optimization
    u_i_county_ent = prior_cp["p"].to_numpy()
    total_fc_county_ent = constraints(prior_cp)

    theta_0 = (np.random.rand(total_constraint))
    print("Initial theta:", theta_0)

    # Optimization
    final_county_population_entropy, data_entropy = fit(u_i_county_ent, prior_cp, total_fc_county_ent, theta_0,
                                                        "entropy", uniform_freq)
    final_county_population_entropy_conditional, data_en_conditional = fit(u_i_county, county_prior_data,
                                                                           total_fc_county, theta_0,
                                                                           "entropy_conditional", uniform_freq)
    tae_county_entropy = tae_calculate(final_county_population_entropy)
    tae_county_entropy_conditional = tae_calculate(final_county_population_entropy_conditional)
    # tae_county_entropy_cost = tae_calculate(final_county_population_entropy_cost)
    tae_county_simulated = tae_calculate(simulated)

    # print("Total TAE with just the conditional probabilities method:", tae_intermediate_prior)
    print("Total TAE with the maximum entropy approach:", tae_county_entropy)
    print("Total TAE with BOTH the maximum entropy and conditional probabilities approach:",
          tae_county_entropy_conditional)
    print("Total TAE with simulated annealing:", tae_county_simulated)
    return final_county_population_entropy, data_entropy, final_county_population_entropy_conditional, data_en_conditional


row = read_top_50.iloc[4, :]
fips = row["County.Code"]
county_name = row["County"]
state_name = row["State_name"].strip()

# prior = pd.read_csv(os.path.join(data_path, "prior_" + str(fips) + "_" + county_name + ".csv"))
# prior = prior.groupby(constraint_list, as_index=False).sum().reset_index(drop=True)
# prior_cp = prior.copy()
# # Conditional
# group_by_conditional = prior_cp.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
#
# prior_copula = read_copula_prior.iloc[:, 1:]
# prior_copula_cp = prior_copula.copy()
# # Conditional
# group_by_copula = prior_copula_cp.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
prior = pd.read_csv(os.path.join(data_path, "prior_" + str(fips) + "_" + county_name + ".csv"))
prior = prior.groupby(constraint_list, as_index=False).sum().reset_index(drop=True)

prior_copula = read_copula_prior.iloc[:, 1:]
prior_all = pd.merge(prior, prior_copula, on=constraint_list, how="outer")
prior_all = prior_all.fillna(0)
prior_all["p"] = (prior_all["p_x"] +prior_all["p_y"])/2
prior_all["p"] = prior_all["p"] / sum(prior_all["p"])

prior_all =prior_all[['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status',"p"]]
prior_cp = prior_all.copy()
# Conditional
group_by_conditional = prior_cp.groupby(req_columns, as_index=False).sum().reset_index(drop=True)

# Results from simulated annealing
simulated = pd.read_csv(os.path.join(data_path, "simulated_" + str(fips) + "_" + county_name + ".csv"))
# Simulated data
simulated_req = simulated[req_columns]
group_by_simulated = simulated_req.groupby(req_columns, as_index=False).size().reset_index(drop=True)
group_by_simulated["Simulated_prob"] = group_by_simulated["size"] / sum(group_by_simulated["size"])

print(group_by_simulated)
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
mew_j = []

for c in constraint_list:
    value_c = dict_constraint[c].values()
    total_constraint += len(value_c)
    mew_j.extend(value_c)
total_constraint += 1

mew_j.extend([total])
mew_j_copy = copy.deepcopy(mew_j)
mew_j = [i / total for i in mew_j]

uniform_freq = [1 for i in mew_j]


def main():
    # Filter unwanted tuples
    county_prior_data = prior_all[prior_all["p"] > pow(10, -6)].reset_index(drop=True)
    county_prior_cp = county_prior_data.copy()
    county_prior_cp["weights"] = np.array([round(i) for i in county_prior_cp['p'].to_numpy() * total])
    prior_intermediate = county_prior_cp.reindex(county_prior_cp.index.repeat(county_prior_cp.weights))
    tae_intermediate_prior = tae_calculate(prior_intermediate)
    print(tae_intermediate_prior, flush=True)
    # copula_prior_data = prior_copula[prior_copula["p"] > pow(10, -7)].reset_index(drop=True)
    # copula_prior_cp = copula_prior_data.copy()
    # copula_prior_cp["weights"] = np.array([round(i) for i in copula_prior_cp['p'].to_numpy() * total])
    # prior_copula_intermediate = copula_prior_cp.reindex(copula_prior_cp.index.repeat(copula_prior_cp.weights))
    # tae_intermediate_copula_prior = tae_calculate(prior_copula_intermediate)
    # print(tae_intermediate_copula_prior)

    final_county_population_entropy, data_entropy, final_county_population_entropy_conditional, data_en_conditional = get_max_tae(county_prior_data,county_prior_cp,simulated_req, total_constraint)


    # print("Total TAE with BOTH the maximum entropy and conditional probabilities approach with cost:",
    #       tae_county_entropy_cost)

    # Validation
    # Actual data
    actual_data = get_pums(state_name, str(fips), county_pums)
    # actual_data.to_csv(os.path.join(data_path, "actual_data.csv"))
    # actual_data = pd.read_csv(os.path.join(data_path, "actual_data.csv"))
    # Get priors from conditional probabilities

    if len(actual_data) != 0:
        group_by_actual = actual_data.groupby(req_columns, as_index=False).size().reset_index(drop=True)
        group_by_actual["Actual_prob"] = group_by_actual["size"] / sum(group_by_actual["size"])

        # Entropy data
        entropy_req = data_entropy.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
        entropy_req["entropy_prob"] = entropy_req["w"]

        # Conditional entropy data
        conditional_en_req = data_en_conditional.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
        conditional_en_req["cond_en_prob"] = conditional_en_req["w"]

        # combine_all_ac_con = pd.merge(group_by_conditional, group_by_actual, on=req_columns, how='inner')
        # combine_all_ac_con[["p", "Actual_prob"]] = combine_all_ac_con[["p", "Actual_prob"]].apply(lambda x: x / sum(x))
        # print(sum(combine_all_ac_con["p"]), sum(combine_all_ac_con["Actual_prob"]))
        #
        # combine_all_ac_sm = pd.merge(group_by_actual, group_by_simulated, on=req_columns, how='inner')
        # combine_all_ac_sm[["Simulated_prob", "Actual_prob"]] = combine_all_ac_sm[["Simulated_prob", "Actual_prob"]].apply(
        #     lambda x: x / sum(x))
        # print(sum(combine_all_ac_sm["Simulated_prob"]), sum(combine_all_ac_sm["Actual_prob"]))
        #
        # combine_all_ac_en = pd.merge(group_by_actual, entropy_req,
        #                              on=req_columns, how='inner')
        # combine_all_ac_en[["entropy_prob", "Actual_prob"]] = combine_all_ac_en[["entropy_prob", "Actual_prob"]].apply(
        #     lambda x: x / sum(x))
        # print(sum(combine_all_ac_en["entropy_prob"]), sum(combine_all_ac_en["Actual_prob"]))
        #
        # combine_all_ac_syn = pd.merge(group_by_actual, conditional_en_req,
        #                               on=req_columns, how='inner')
        # combine_all_ac_syn[["cond_en_prob", "Actual_prob"]] = combine_all_ac_syn[["cond_en_prob", "Actual_prob"]].apply(
        #     lambda x: x / sum(x))
        #
        # print(sum(combine_all_ac_syn["cond_en_prob"]), sum(combine_all_ac_syn["Actual_prob"]))
        # combine_all = pd.merge(group_by_conditional, group_by_actual, on=['age', 'gender', 'marital_status', 'emp_status', 'Insurance',
        #    'edu_attain', 'pov_status'], how='left')
        # combine_all = pd.merge(combine_all, group_by_simulated, on=['age', 'gender', 'marital_status', 'emp_status', 'Insurance',
        #    'edu_attain', 'pov_status'], how='left')
        # combine_all = pd.merge(combine_all, entropy_req,
        #                        on=['age', 'gender', 'marital_status', 'emp_status', 'Insurance',
        #                            'edu_attain', 'pov_status'], how='left')
        # combine_all = pd.merge(combine_all, conditional_en_req,
        #                        on=['age', 'gender', 'marital_status', 'emp_status', 'Insurance',
        #                            'edu_attain', 'pov_status'], how='left')

        combine_all = pd.merge(group_by_conditional, group_by_actual, on=req_columns, how='left')
        combine_all = pd.merge(combine_all, group_by_simulated, on=req_columns, how='left')
        combine_all = pd.merge(combine_all, entropy_req,
                               on=req_columns, how='left')
        combine_all = pd.merge(combine_all, conditional_en_req,
                               on=req_columns, how='left')

        combine_all = combine_all.fillna(1/total)
        for i in ["p", "Simulated_prob", "Actual_prob", "entropy_prob", "cond_en_prob"]:
            print(sum(combine_all[i]))
        combine_all[["p", "Simulated_prob","entropy_prob", "Actual_prob","cond_en_prob"]] = combine_all[["p", "Simulated_prob",
                                                    "entropy_prob","Actual_prob","cond_en_prob"]].apply(lambda x: x/sum(x))

        for i in ["p", "Simulated_prob", "Actual_prob", "entropy_prob", "cond_en_prob"]:
            print(sum(combine_all[i]))
        # combine_all.to_csv(os.path.join(data_path, "combine_all.csv"))

        kl_simulated = kl(combine_all["Actual_prob"], combine_all["Simulated_prob"])
        kl_entropy = kl(combine_all["Actual_prob"], combine_all["entropy_prob"])
        kl_conditional = kl(combine_all["Actual_prob"], combine_all["p"])
        kl_entropy_conditional = kl(combine_all["Actual_prob"], combine_all["cond_en_prob"])
        print(kl_simulated, kl_conditional, kl_entropy, kl_entropy_conditional)

        # kl_simulated = kl(combine_all_ac_sm["Actual_prob"], combine_all_ac_sm["Simulated_prob"])
        # kl_entropy = kl(combine_all_ac_en["Actual_prob"], combine_all_ac_en["entropy_prob"])
        # kl_conditional = kl(combine_all_ac_con["Actual_prob"], combine_all_ac_con["p"])
        # kl_entropy_conditional = kl(combine_all_ac_syn["Actual_prob"], combine_all_ac_syn["cond_en_prob"])
        # print(kl_simulated, kl_conditional, kl_entropy, kl_entropy_conditional)

        association_plot(prior_intermediate[req_columns],
                         "Conditional probabilities " + str(fips))
        association_plot(final_county_population_entropy[req_columns], "Maximum entropy " + str(fips))
        association_plot(
            final_county_population_entropy_conditional[req_columns],
            "Entropy + Conditional " + str(fips))
        association_plot(
            actual_data[req_columns],
            "Actual " + str(fips))

        association_plot(
            simulated[req_columns],
            "Simulated annealing " + str(fips))
main()