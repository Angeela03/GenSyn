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
import seaborn as sns
# from dython.nominal import associations
# from sklearn.preprocessing import OneHotEncoder
from get_puma import get_pums
from csv import DictWriter
from pypums import ACS

np.random.seed(7)

constraint_list = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status', 'drug_overdose',  'Insurance', 'Veteran']

data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data"
# data_path = "/scratch/aachary/conditional_rpobabilties_generic/data"
read_top_50 = pd.read_csv(os.path.join(data_path, "top_50_od.csv"))
county_pums = pd.read_csv(os.path.join(data_path, "puma_county.csv"))


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
        first_term = sum(mew_j * theta * cost)
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

# sns.boxplot(y='TAE', x='Method',
#                  data=merge_tae,
#                  palette="colorblind")

# plt.gcf().subplots_adjust(bottom=0.5)
def plot_separate(list_dfs):
    for index in range(len(list_dfs)):
        ax = list_dfs[index].plot(kind = 'bar', title=list_dfs[index]["Attribute"].iloc[0], width = 0.75, figsize=(5, 5),colormap= "Paired")

        for y in ax.patches:
            ax.text(y.get_x() + y.get_width() / 4, y.get_height() * 1.05, f'{y.get_height():.1f}', fontsize=6, rotation=90)
            ax.set_ylim(0, ax.get_ylim()[1] + 10)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
        plt.tight_layout()
        plt.show()


def kl(p, q):
    vec = scipy.special.rel_entr(p, q)
    kl_div = np.sum(vec)
    return kl_div


for ind, row in read_top_50.iterrows():
    fips = row["County.Code"]
    county_name = row["County"]
    state_name = row["State_name"].strip()

    # Get priors from conditional probabilities
    prior = pd.read_csv(os.path.join(data_path, "prior_"+str(fips)+"_"+county_name+".csv"))

    # get constraints
    with open(os.path.join(data_path, "json_constraint_"+str(fips)+"_"+county_name+".json")) as json_data:
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
    #
    # # Filter unwanted tuples
    county_prior_data = prior[prior["p"] > pow(10, -7)].reset_index(drop=True)

    # Get the immediate representation of the county to see what we get if we directly consider \
    # priors to be the population
    county_prior_cp = county_prior_data.copy()
    county_prior_cp["weights"] = np.array([round(i) for i in county_prior_cp['p'].to_numpy() * total])
    prior_intermediate = county_prior_cp.reindex(county_prior_cp.index.repeat(county_prior_cp.weights))
    # for c in constraint_list:
    #     for d in [i for i in constraint_list if i != c]:
    #         groupby_predicted = prior_intermediate.groupby([c, d]).size().reset_index().transform(lambda x: x.ffill())
    #       groupby_actual =
    #     predicted_value = final_county_population_entropy_conditional[c].value_counts()
    # actual_value = dict_constraint[c]
    # list_actual = []
    # list_predicted_ = []
    # list_category = []
    # for i, value in predicted_value.items():
    #     list_actual.append(actual_value[i])
    #     list_predicted_.append(value)
    #     predicted_baltimore_value_i = predicted_value.loc[i]
    #     list_category.append(i)
    # attribute = [c] * len(actual_value)
    # df_i = pd.DataFrame({'Attribute': attribute,
    #                      'Category': list_category,
    #                      'Predicted': list_predicted_,
    #                      'Actual': list_actual},
    #                     columns=['Attribute', 'Category', 'Predicted', 'Actual'])
    #
    # df_i.set_index('Category', inplace=True)
    # list_dfs.append(df_i)
    #


    tae_intermediate_prior = tae_calculate(prior_intermediate)
    print(tae_intermediate_prior, flush=True)

    # define terms for max entropy optimization
    u_i_county = county_prior_data["p"].to_numpy()
    total_fc_county = constraints(county_prior_data)

    theta_0 = (np.random.rand(total_constraint))
    print("Initial theta:", theta_0)

    # Optimization
    final_county_population_entropy_conditional, data_en_conditional = fit(u_i_county, county_prior_data, total_fc_county, theta_0,
                                                      "entropy_conditional", uniform_freq)

    tae_county_entropy_conditional = tae_calculate(final_county_population_entropy_conditional)
    print("Total TAE with just the conditional probabilities method:", tae_intermediate_prior)
    print("Total TAE with BOTH the maximum entropy and conditional probabilities approach:",
          tae_county_entropy_conditional)
    list_dfs = []
    for c in constraint_list:
        predicted_value = final_county_population_entropy_conditional[c].value_counts()
        actual_value = dict_constraint[c]
        list_actual = []
        list_predicted_ = []
        list_category = []
        for i, value in predicted_value.items():
            list_actual.append(actual_value[i])
            list_predicted_.append(value)
            predicted_baltimore_value_i = predicted_value.loc[i]
            list_category.append(i)
        attribute = [c] * len(actual_value)
        df_i = pd.DataFrame({'Attribute': attribute,
                             'Category': list_category,
                             'Predicted': list_predicted_,
                             'Actual': list_actual},
                           columns=['Attribute', 'Category', 'Predicted', 'Actual'])

        df_i.set_index('Category', inplace=True)
        list_dfs.append(df_i)
    plot_separate(list_dfs)



