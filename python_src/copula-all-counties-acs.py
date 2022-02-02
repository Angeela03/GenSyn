import pandas as pd
import os
import json
import statistics
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import copy
from get_puma_acs import get_pums
from csv import DictWriter
from pypums import ACS
from dython.nominal import associations
import numpy as np

np.random.seed(7)

constraint_list = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status', 'geog_mobility', 'nativity']
req_columns = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status', 'geog_mobility', 'nativity']

data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data_acs"
# data_path = "/scratch/aachary/conditional_rpobabilties_generic/data_acs"
read_top_50 = pd.read_csv(os.path.join(data_path, "top_50_acs.csv"))
county_pums = pd.read_csv(os.path.join(data_path, "puma_county.csv"))

print(list(read_top_50["County.Code"].values))


# print(read_top_50)
def weight_intermediate(theta, data, total_fc, cost):
    theta_f_repeat = np.tile(theta, (len(data), 1))
    cost_repeat = np.tile(cost, (len(data), 1))
    sum_j = np.sum(total_fc * theta_f_repeat *cost_repeat, axis=1)
    exp_j = np.exp(sum_j - 1)
    return exp_j


def find_weights_entropy(theta, data, total_fc, cost):
    return weight_intermediate(theta, data, total_fc, cost)


def find_weights_entropy_conditional(theta, u_i, data, total_fc,cost):
    exp_j = weight_intermediate(theta, data, total_fc,cost)
    w_i = u_i * exp_j
    return w_i


def max_entropy(theta, u_i, data, total_fc, method,cost):
    global t
    global f_values
    global weight_values
    print('Iteration', t, flush=True)
    theta_repeat = np.tile(theta, (len(data), 1))
    cost_repeat = np.tile(cost, (len(data), 1))
    sum_j = np.sum(total_fc * theta_repeat *cost_repeat, axis=1)
    exp_j = np.exp(sum_j - 1)
    if method == "entropy":
        second_term = sum(exp_j)
        first_term = sum(mew_j * theta)
        f = - first_term + second_term
        weights = find_weights_entropy(theta, data, total_fc,cost)
        sum_weights = sum(weights)
    elif method == "entropy_conditional":
        second_term = sum(exp_j * u_i)
        first_term = sum(mew_j * theta *cost)
        f = - first_term + second_term
        weights = find_weights_entropy_conditional(theta, u_i, data, total_fc,cost)
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


def jacobian_m(theta, u_i, data, total_fc, method,cost):
    j_jac = []
    for j in range(len(mew_j)):
        theta_repeat = np.tile(theta, (len(data), 1))
        sum_j = np.sum(total_fc * theta_repeat, axis=1)
        exp_j = np.exp(sum_j-1)
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
    res = minimize(max_entropy, jac = jacobian_m, x0=theta_0, args=(u_i, data, total_fc, method, cost), method='L-BFGS-B', options={'disp': True})
    theta_f = res.x
    print("Final theta", theta_f, flush=True)
    f_values = np.array(f_values)
    f_values = np.nan_to_num(f_values)
    if method == "entropy":
        final_weights = find_weights_entropy(theta_f, data, total_fc, cost)
    elif method == "entropy_conditional":
        final_weights = find_weights_entropy_conditional(theta_f, u_i, data, total_fc,cost)
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
    fig, axs = plt.subplots(2,  int(len_dfs/2), figsize=(20, 20))
    plt.subplots_adjust(hspace=1)
    fig.suptitle('Actual vs Observed values for each attribute')
    list_i_j = [(i, j) for i in range(2) for j in range(int(len_dfs/2))]
    for index in range(len(list_dfs)):

        i = list_i_j[index][0]
        j = list_i_j[index][1]
        list_dfs[index].plot(kind='bar', ax=axs[i][j], title=list_dfs[index][""].iloc[0],  width=0.9)
    for ax in axs.reshape(-1):
         for y in ax.patches:
             ax.text(y.get_x() + y.get_width() / 4, y.get_height() * 1.05, f'{y.get_height():.1f}')
             ax.set_ylim(0, ax.get_ylim()[1] + 10)
    plt.show()
    plt.savefig("Results.png")


def plot_separate(list_dfs):
    for index in range(len(list_dfs)):
        ax = list_dfs[index].plot(kind = 'bar', title=list_dfs[index]["Attribute"].iloc[0], width = 1, figsize=(10, 10))
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
    varLabels = ['age', 'gender', 'edu',
                   'marital_sts', 'pov_sts', 'emp_sts', 'geo_mobility', 'nativity']
    assoc = associations(df, ax=ax, plot=False,cbar=False)
    ax.set_xticks(np.arange(0, 8))
    ax.set_yticks(np.arange(0, 8))
    ax.set_xticklabels(varLabels, rotation=45)
    ax.set_yticklabels(varLabels, rotation=45)
    # plt.show()
    # ax.set_xticklabels(ax.get_xticks(), rotation=90)
    fig.tight_layout()
    fig.savefig(os.path.join(data_path, 'Categorical Correlation matrix acs - ' + name))
    corr = assoc["corr"]
    corr.to_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' + name +"_.csv"))
    flatten_assoc = corr.values.flatten()
    return flatten_assoc


def prior_calculations(prior, name, tau):
    prior_cp = prior.copy()
    # Conditional
    group_by_conditional = prior_cp.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
    if name == "entropy":
        county_prior_data = prior
        # county_prior_data = prior[prior["p"] > pow(10, -6)].reset_index(drop=True)
    else:
        county_prior_data = prior[prior["p"] >=tau].reset_index(drop=True)
    print(county_prior_data)
    county_prior_cp = county_prior_data.copy()
    county_prior_cp["weights"] = np.array([round(i) for i in county_prior_cp['p'].to_numpy() * total])
    prior_intermediate = county_prior_cp.reindex(county_prior_cp.index.repeat(county_prior_cp.weights))
    tae_intermediate_prior = tae_calculate(prior_intermediate)

    # Define for syntropy optimization
    u_i_county = county_prior_data["p"].to_numpy()
    total_fc_county = constraints(county_prior_data)
    final_county_population_entropy_conditional, data_en_conditional = fit(u_i_county, county_prior_data,
                                                                           total_fc_county, theta_0,
                                                                           name, uniform_freq)
    return final_county_population_entropy_conditional, data_en_conditional, group_by_conditional, tae_intermediate_prior, prior_intermediate


def matrix_distance(a, b):
    temp = a - b
    dist = np.sqrt(np.dot(temp.T, temp))
    return dist


for ind, row in read_top_50.iterrows():
    fips = row["County.Code"]
    county_name = row["County"].strip()
    state_name = row["State"].strip()
    print(fips)

    # Get priors from copula and conditional probabilities
    copula_prior = pd.read_csv(os.path.join(data_path, "prior_copula_acs_" + str(fips) + ".csv"))
    prior_copula = copula_prior.iloc[:, 1:]

    conditional_prior = pd.read_csv(os.path.join(data_path, "prior_" + str(fips) + "_" + county_name + "_acs.csv"))
    # conditional_prior_some = conditional_prior[conditional_prior["p"] > pow(10, -6)].reset_index(drop=True)
    prior_all = pd.merge(conditional_prior, prior_copula, on=constraint_list, how="outer")
    prior_all = prior_all.fillna(0)
    prior_all["p"] = (prior_all["p_x"] + prior_all["p_y"]) / 2
    prior_all["p"] = prior_all["p"] / sum(prior_all["p"])
    prior_all = prior_all[['age', 'gender', 'edu_attain',
                           'marital_status', 'pov_status', 'emp_status', 'geog_mobility', 'nativity', 'p']]
    # get constraints
    with open(os.path.join(data_path, "json_constraint_" + str(fips) + "_" + county_name + "_acs.json")) as json_data:
        j_data = json.load(json_data)
    dict_constraint = {}
    j_data = json.loads(j_data[0])


    print(j_data)
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
    # for k, v in dict_constraint.items():
    #     print(type(v))
    #     print(v)
    #     print(v.values())
    #
    #     s = sum(list(v.values()))
    #     print(s)
    for c in constraint_list:
        value_c = dict_constraint[c].values()
        total_constraint += len(value_c)
        mew_j.extend(value_c)
    total_constraint += 1

    mew_j.extend([total])
    mew_j_copy = copy.deepcopy(mew_j)
    mew_j = [i / total for i in mew_j]
    uniform_freq = [1 for i in mew_j]
    theta_0 = (np.random.rand(total_constraint))
    print("Initial theta:", theta_0)

    # county_prior_cp = conditional_prior_some.copy()
    # county_prior_cp["weights"] = np.array([round(i) for i in county_prior_cp['p'].to_numpy() * total])
    # prior_intermediate = county_prior_cp.reindex(county_prior_cp.index.repeat(county_prior_cp.weights))
    # tae_intermediate_prior = tae_calculate(prior_intermediate)
    # print(tae_intermediate_prior)
    # print(prior_intermediate)
    # association_plot(prior_intermediate[req_columns],
    #                  "Conditional probabilities " + str(fips))
    # prior_intermediate[req_columns].to_csv("test_prior_intermediate.csv", index=False)

    # Fit a maximum entropy model for copula as prior and both conditional probabiltiies and copula as prior
    tau = 1 / total

    # Fit a maximum entropy model for copula as prior and both conditional probabiltiies and copula as prior
    final_county_population_cp, data_cp, group_by_copula, tae_cp, cp_intermediate = prior_calculations(prior_copula,
                                                                                                       "entropy_conditional",
                                                                                                       tau)

    final_county_population_combined, data_combined, group_by_combined, tae_combined, combined_intermediate = prior_calculations(
        prior_all, "entropy_conditional", tau)

    # Fit a maximum entropy model for conditional as prior and without any prior
    final_county_population_entropy, data_entropy, group_by_entropy, tae_entropy, entropy_intermediate = \
        prior_calculations(conditional_prior, "entropy", tau)
    final_county_population_entropy_conditional, data_en_conditional, group_by_en_cond, tae_entropy_cond, entropy_cond_intermediate \
        = prior_calculations(conditional_prior, "entropy_conditional", tau)

    # Rename prior probabilitites column to avoid conflict
    group_by_copula = group_by_copula.rename({'p': 'p_cp'}, axis=1)
    group_by_combined = group_by_combined.rename({'p': 'p_combined'}, axis=1)
    group_by_cond = group_by_entropy.rename({'p': 'p_cond'}, axis=1)

    # Find tae for the final populations
    tae_copula_final = tae_calculate(final_county_population_cp)
    tae_copula_conditional_final = tae_calculate(final_county_population_combined)

    tae_county_entropy = tae_calculate(final_county_population_entropy)
    tae_county_entropy_conditional = tae_calculate(final_county_population_entropy_conditional)

    # Get final probabiltiies for copula
    conditional_cp = data_cp.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
    conditional_cp["cond_cp"] = conditional_cp["w"]

    # copula conditional
    conditional_combined = data_combined.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
    conditional_combined["cond_combined"] = conditional_combined["w"]

    # Entropy data
    entropy_req = data_entropy.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
    entropy_req["entropy_prob"] = entropy_req["w"]

    # Conditional entropy data
    conditional_en_req = data_en_conditional.groupby(req_columns, as_index=False).sum().reset_index(drop=True)
    conditional_en_req["cond_en_prob"] = conditional_en_req["w"]

    # Results from simulated annealing
    simulated = pd.read_csv(os.path.join(data_path, "simulated_" + str(fips) + "_" + county_name + "_acs.csv"))
    simulated_req = simulated[constraint_list]
    tae_county_simulated = tae_calculate(simulated_req)
    group_by_simulated = simulated_req.groupby(req_columns, as_index=False).size().reset_index(drop=True)
    group_by_simulated["Simulated_prob"] = group_by_simulated["size"] / sum(group_by_simulated["size"])

    field_names_tae = ["Method", "State", "County", "TAE"]
    with open(os.path.join(data_path, "tae_scores_copula_acs"+"_.csv"), "a") as f:
        dictwriter_object = DictWriter(f, fieldnames=field_names_tae)
        # Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow({"Method": "Copula-prior", "State": state_name, "County": county_name,
                                    "TAE": tae_cp})
        dictwriter_object.writerow(
            {"Method": "Copula+Conditional-prior", "State": state_name, "County": county_name, "TAE": tae_combined})
        dictwriter_object.writerow(
            {"Method": "Copula-final", "State": state_name, "County": county_name, "TAE": tae_copula_final})
        dictwriter_object.writerow(
            {"Method": "Copula+Conditional-final", "State": state_name, "County": county_name, "TAE": tae_copula_conditional_final})
        dictwriter_object.writerow({"Method": "Conditional probabilities", "State": state_name, "County": county_name,
                                    "TAE": tae_entropy})
        dictwriter_object.writerow(
            {"Method": "Max Entropy", "State": state_name, "County": county_name, "TAE": tae_county_entropy})
        dictwriter_object.writerow(
            {"Method": "SynthACS", "State": state_name, "County": county_name, "TAE": tae_county_simulated})
        dictwriter_object.writerow(
            {"Method": "SynTropy", "State": state_name, "County": county_name, "TAE": tae_county_entropy_conditional})

        # Close the file object
        f.close()

    print(tae_cp, tae_combined, tae_copula_final, tae_copula_conditional_final,tae_entropy,tae_county_entropy,tae_county_simulated,tae_county_entropy_conditional)

    actual_file = os.path.join(data_path, "actual_acs" + str(fips) + "_" + county_name + "_acs.csv")
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

    if len(actual_data) != 0:
        group_by_actual = actual_data.groupby(req_columns, as_index=False).size().reset_index(drop=True)
        group_by_actual["Actual_prob"] = group_by_actual["size"] / sum(group_by_actual["size"])

        combine_all = pd.merge(group_by_combined, group_by_actual,
                               on=req_columns, how='left')
        combine_all = pd.merge(combine_all, group_by_copula,
                               on=req_columns, how='left')
        combine_all = pd.merge(combine_all, conditional_cp,
                               on=req_columns, how='left')
        combine_all = pd.merge(combine_all, conditional_combined,
                               on=req_columns, how='left')
        combine_all = pd.merge(combine_all, group_by_cond, on=req_columns, how='left')
        combine_all = pd.merge(combine_all, group_by_simulated, on=req_columns, how='left')
        combine_all = pd.merge(combine_all, entropy_req,
                               on=req_columns, how='left')
        combine_all = pd.merge(combine_all, conditional_en_req,
                               on=req_columns, how='left')

        combine_all = combine_all.fillna(1 / total)
        # for i in ["p", "Simulated_prob", "Actual_prob", "entropy_prob", "cond_en_prob"]:
        #     print(sum(combine_all[i]))
        combine_all[["p_cp", "p_combined", "cond_cp", "cond_combined", "Actual_prob","p_cond","Simulated_prob", "entropy_prob", "cond_en_prob"]] = combine_all[
            ["p_cp", "p_combined", "cond_cp", "cond_combined", "Actual_prob","p_cond","Simulated_prob", "entropy_prob", "cond_en_prob"]].apply(lambda x: x / sum(x))
        for i in ["p_cp", "p_combined", "cond_cp", "cond_combined", "Actual_prob","p_cond","Simulated_prob", "entropy_prob", "cond_en_prob"]:
            print(sum(combine_all[i]))
        kl_cp = kl(combine_all["Actual_prob"], combine_all["p_cp"])
        kl_combined = kl(combine_all["Actual_prob"], combine_all["p_combined"])
        kl_cp_final = kl(combine_all["Actual_prob"], combine_all["cond_cp"])
        kl_combined_final = kl(combine_all["Actual_prob"], combine_all["cond_combined"])
        kl_simulated = kl(combine_all["Actual_prob"], combine_all["Simulated_prob"])
        kl_entropy = kl(combine_all["Actual_prob"], combine_all["entropy_prob"])
        kl_conditional = kl(combine_all["Actual_prob"], combine_all["p_cond"])
        kl_entropy_conditional = kl(combine_all["Actual_prob"], combine_all["cond_en_prob"])
        print(kl_cp, kl_combined, kl_cp_final, kl_combined_final,kl_simulated,kl_entropy,kl_conditional,kl_entropy_conditional)

        field_names_kl = ["Method", "State", "County", "KL_divergence"]
        with open(os.path.join(data_path, "kl_div_scores_acs"+"_copula.csv"), "a") as f:
            dictwriter_object = DictWriter(f, fieldnames=field_names_kl)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow({"Method": "Copula-prior", "State": state_name, "County": county_name,
                                        "KL_divergence": kl_cp})
            dictwriter_object.writerow(
                {"Method": "Copula+Conditional-prior", "State": state_name, "County": county_name, "KL_divergence": kl_combined})
            dictwriter_object.writerow(
                {"Method": "Copula-final", "State": state_name, "County": county_name, "KL_divergence": kl_cp_final})
            dictwriter_object.writerow(
                {"Method": "Copula+Conditional-final", "State": state_name, "County": county_name, "KL_divergence": kl_combined_final})
            dictwriter_object.writerow(
                {"Method": "Conditional probabilities", "State": state_name, "County": county_name,
                 "KL_divergence": kl_conditional})
            dictwriter_object.writerow(
                {"Method": "Max Entropy", "State": state_name, "County": county_name, "KL_divergence": kl_entropy})
            dictwriter_object.writerow(
                {"Method": "SynthACS", "State": state_name, "County": county_name, "KL_divergence": kl_simulated})
            dictwriter_object.writerow({"Method": "SynTropy", "State": state_name, "County": county_name,
                                        "KL_divergence": kl_entropy_conditional})
            # Close the file object
            f.close()

        association_cp_prior = association_plot(cp_intermediate[req_columns],
                                                "Copula-prior " + str(fips))
        association_combined_prior = association_plot(combined_intermediate[req_columns],
                                                      "Combined_cp-prior " + str(fips))
        association_cp_final = association_plot(final_county_population_cp[req_columns], "Copula-final " + str(fips))
        association_combined_final = association_plot(
            final_county_population_combined[req_columns],
            "Combined-final " + str(fips))
        association_conditional = association_plot(entropy_intermediate[req_columns],
                                                   "Conditional probabilities " + str(fips))
        association_entropy_final = association_plot(final_county_population_entropy[req_columns],
                                                     "Maximum entropy " + str(fips))
        association_en_cn_final = association_plot(
            final_county_population_entropy_conditional[req_columns],
            "Entropy + Conditional " + str(fips))
        association_actual = association_plot(
            actual_data[req_columns],
            "Actual " + str(fips))

        association_simulated = association_plot(
            simulated[req_columns],
            "Simulated annealing " + str(fips))

        dist_cp_prior = matrix_distance(association_cp_prior, association_actual)
        dist_combined_prior = matrix_distance(association_combined_prior, association_actual)
        dist_cp_final = matrix_distance(association_cp_final, association_actual)
        dist_combined_final = matrix_distance(association_combined_final, association_actual)
        dist_conditional = matrix_distance(association_conditional, association_actual)
        dist_entropy_final = matrix_distance(association_entropy_final, association_actual)
        dist_en_cn_final = matrix_distance(association_en_cn_final, association_actual)
        dist_simulated = matrix_distance(association_simulated, association_actual)

        field_names_kl = ["Method", "State", "County", "Norm"]
        with open(os.path.join(data_path, "Norm" + "_copula.csv"), "a") as f:
            dictwriter_object = DictWriter(f, fieldnames=field_names_kl)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow({"Method": "Copula-prior", "State": state_name, "County": county_name,
                                        "Norm": dist_cp_prior})
            dictwriter_object.writerow(
                {"Method": "Copula+Conditional-prior", "State": state_name, "County": county_name,
                 "Norm": dist_combined_prior})
            dictwriter_object.writerow(
                {"Method": "Copula-final", "State": state_name, "County": county_name, "Norm": dist_cp_final})
            dictwriter_object.writerow(
                {"Method": "Copula+Conditional-final", "State": state_name, "County": county_name,
                 "Norm": dist_combined_final})
            dictwriter_object.writerow(
                {"Method": "Conditional probabilities", "State": state_name, "County": county_name,
                 "Norm": dist_conditional})
            dictwriter_object.writerow(
                {"Method": "Max Entropy", "State": state_name, "County": county_name, "Norm": dist_entropy_final})
            dictwriter_object.writerow(
                {"Method": "SynthACS", "State": state_name, "County": county_name, "Norm": dist_simulated})
            dictwriter_object.writerow({"Method": "SynTropy", "State": state_name, "County": county_name,
                                        "Norm": dist_en_cn_final})
            # Close the file object
            f.close()
