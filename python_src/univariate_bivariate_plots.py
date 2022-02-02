import pandas as pd
import os
import json
import statistics
import matplotlib.pyplot as plt
import copy
import sys
import numpy as np


# constraint_list = ['age', 'gender', 'edu_attain',
#                    'marital_status', 'pov_status', 'emp_status', 'drug_overdose',  'Insurance', 'Veteran']
# req_columns = ["age", "gender", "marital_status", "pov_status", "emp_status",
#                                              "Insurance", "edu_attain"]

constraint_list = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status', 'geog_mobility', 'nativity']
req_columns = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status', 'geog_mobility', 'nativity']

bi_variate_plots = [("martial_status", "age"), ("emp_status", "pov_status"), ("geog_mobility", "edu_attain")]

data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data_acs"
# data_path = "/scratch/aachary/conditional_rpobabilties_generic/data"
read_top_50 = pd.read_csv(os.path.join(data_path, "top_50_acs.csv"))
print(read_top_50)


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
        print(list_dfs[index])
        df_index = list_dfs[index]
        # df_index['Predicted'] = np.log2(df_index['Predicted'])
        # df_index['Actual'] = np.log2(df_index['Actual'])
        ax = list_dfs[index].plot(kind = 'bar', title=list_dfs[index]["Attribute"].iloc[0], width = 0.75, figsize=(5, 5),colormap= "Paired")

        for y in ax.patches:
            ax.text(y.get_x() + y.get_width() / 4, y.get_height() * 1.05, f'{y.get_height():.1f}', fontsize=13, rotation=90)
            ax.set_ylim(0, ax.get_ylim()[1] + 10)
        # plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
        right_side = ax.spines["right"]
        right_side.set_visible(False)

        top_side = ax.spines["top"]
        top_side.set_visible(False)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(list_dfs[index]["Attribute"].iloc[0], fontsize=16)
        handles = ax.legend_.legendHandles
        labels = [text.get_text() for text in ax.legend_.texts]
        plt.legend(handles, labels, fontsize=15)

        plt.tight_layout()
        plt.show()


def univariate():
    list_dfs = []
    for c in constraint_list:
        predicted_value = combined_data[c].value_counts()
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
    return 0


def bivariate():
    list_dfs = []
    for c in constraint_list:
        predicted_value = combined_data[c].value_counts()
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
    return 0


for ind, row in read_top_50.iterrows():
    fips = row["County.Code"]
    county_name = row["County"].strip()
    state_name = row["State"].strip()
    print(fips)

    combined_file = os.path.join(data_path, "Combined_" + str(fips) + "_" + county_name + "_.csv")
    combined_data = pd.read_csv(combined_file)

    # get constraints
    with open(os.path.join(data_path, "json_constraint_" + str(fips) + "_" + county_name + "_acs.json")) as json_data:
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
    univariate()
