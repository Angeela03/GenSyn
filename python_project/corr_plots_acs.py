

import numpy as np
import pandas as pd
import os
from csv import DictWriter
import seaborn as sns
import matplotlib.pyplot as plt

constraint_list = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status', 'geog_mobility', 'nativity']
req_columns = ['age', 'gender', 'edu_attain',
                   'marital_status', 'pov_status', 'emp_status', 'geog_mobility', 'nativity']

data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data_acs"
# data_path = "/scratch/aachary/conditional_rpobabilties_generic/data_acs"
read_top_50 = pd.read_csv(os.path.join(data_path, "top_50_acs.csv"))


conditional_pair = [("marital_status", "gender"), ("pov_status", "emp_status"), ("nativity", "age"), ("geog_mobility", "edu_attain")]
non_conditional_pair = [("pov_status", "edu_attain"), ("geog_mobility", "emp_status"), ("geog_mobility","nativity"), ("pov_status", "nativity")]


def find_absolute_diff(actual, predicted, pair):
    list_tae_conditional = []
    print(pair)
    for i,j in pair:
        actual_corr = actual.loc[i, j]
        print(actual_corr)
        predicted_corr = predicted.loc[i,j]
        print(predicted_corr)
        abs_diff = abs(actual_corr - predicted_corr)
        list_tae_conditional.append(abs_diff)
    return list_tae_conditional


def find_absolute_corr():

    for ind, row in read_top_50.iterrows():
        fips = row["County.Code"]
        county_name = row["County"].strip()
        state_name = row["State"].strip()
        print(fips)

        actual_file = os.path.join(data_path, "actual_acs" + str(fips) + "_" + county_name + "_acs.csv")
        if os.path.exists(actual_file):
            actual_data = pd.read_csv(actual_file)
            if len(actual_data) != 0:
                print("File_exists")


                association_cp_prior = pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' + "Copula-prior " + str(fips) +"_.csv"), index_col=0)

                association_combined_prior = pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' +
                                                              "Combined_cp-prior " + str(fips)+"_.csv"), index_col=0)
                association_cp_final = pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' + "Copula-final " + str(fips)+"_.csv"), index_col=0)
                association_combined_final = pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' +
                    "Combined-final " + str(fips)+"_.csv"), index_col=0)
                association_conditional = pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' +
                                                           "Conditional probabilities " + str(fips)+"_.csv"), index_col=0)
                association_entropy_final = pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' +
                                                             "Maximum entropy " + str(fips)+"_.csv"), index_col=0)
                association_en_cn_final = pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' +
                    "Entropy + Conditional " + str(fips)+"_.csv"), index_col=0)
                association_actual =  pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' +
                    "Actual " + str(fips)+"_.csv"), index_col=0)

                association_simulated =  pd.read_csv(os.path.join(data_path, 'Categorical Correlation matrix acs - ' +
                    "Simulated annealing " + str(fips)+"_.csv"), index_col=0)

                dist_cp_prior = find_absolute_diff(association_cp_prior, association_actual, conditional_pair)
                dist_combined_prior = find_absolute_diff(association_combined_prior, association_actual,conditional_pair)
                dist_cp_final = find_absolute_diff(association_cp_final, association_actual,conditional_pair)
                dist_combined_final = find_absolute_diff(association_combined_final, association_actual,conditional_pair)
                dist_conditional = find_absolute_diff(association_conditional, association_actual,conditional_pair)
                dist_entropy_final = find_absolute_diff(association_entropy_final, association_actual,conditional_pair)
                dist_en_cn_final = find_absolute_diff(association_en_cn_final, association_actual,conditional_pair)
                dist_simulated = find_absolute_diff(association_simulated, association_actual, conditional_pair)

                dist_cp_prior_non = find_absolute_diff(association_cp_prior, association_actual, non_conditional_pair)
                dist_combined_prior_non = find_absolute_diff(association_combined_prior, association_actual, non_conditional_pair)
                dist_cp_final_non = find_absolute_diff(association_cp_final, association_actual, non_conditional_pair)
                dist_combined_final_non = find_absolute_diff(association_combined_final, association_actual, non_conditional_pair)
                dist_conditional_non = find_absolute_diff(association_conditional, association_actual, non_conditional_pair)
                dist_entropy_final_non = find_absolute_diff(association_entropy_final, association_actual, non_conditional_pair)
                dist_en_cn_final_non = find_absolute_diff(association_en_cn_final, association_actual, non_conditional_pair)
                dist_simulated_non = find_absolute_diff(association_simulated, association_actual, non_conditional_pair)

                field_names_kl = ["Method", "State", "County", "Absolute error - Association", "Pair"]
                with open(os.path.join(data_path, "TAE-Association-pair.csv"), "a") as f:
                    dictwriter_object = DictWriter(f, fieldnames=field_names_kl)
                    # Pass the dictionary as an argument to the Writerow()
                    for p in range(len(conditional_pair)):

                        dictwriter_object.writerow({"Method": "Copula-prior", "State": state_name, "County": county_name,
                                                "Absolute error - Association": dist_cp_prior[p], 'Pair':conditional_pair[p]})
                    for p in range(len(conditional_pair)):
                        dictwriter_object.writerow(
                        {"Method": "Copula+Conditional-prior", "State": state_name, "County": county_name,
                            "Absolute error - Association": dist_combined_prior[p], 'Pair':conditional_pair[p]})
                    for p in range(len(conditional_pair)):
                        dictwriter_object.writerow(
                            {"Method": "Copula-final", "State": state_name, "County": county_name, "Absolute error - Association": dist_cp_final[p], 'Pair':conditional_pair[p]})
                    for p in range(len(conditional_pair)):

                        dictwriter_object.writerow(
                        {"Method": "Copula+Conditional-final", "State": state_name, "County": county_name,
                         "Absolute error - Association": dist_combined_final[p], 'Pair': conditional_pair[p]})
                    for p in range(len(conditional_pair)):

                        dictwriter_object.writerow(
                        {"Method": "Conditional probabilities", "State": state_name, "County": county_name,
                         "Absolute error - Association": dist_conditional[p], 'Pair': conditional_pair[p]})
                    for p in range(len(conditional_pair)):

                        dictwriter_object.writerow(
                        {"Method": "Max Entropy", "State": state_name, "County": county_name,  "Absolute error - Association": dist_entropy_final[p], 'Pair': conditional_pair[p]})
                    for p in range(len(conditional_pair)):

                      dictwriter_object.writerow(
                        {"Method": "SynthACS", "State": state_name, "County": county_name, "Absolute error - Association": dist_simulated[p], 'Pair': conditional_pair[p]})
                    for p in range(len(conditional_pair)):

                        dictwriter_object.writerow({"Method": "SynTropy", "State": state_name, "County": county_name,
                                                    "Absolute error - Association": dist_en_cn_final[p],
                                                    'Pair': conditional_pair[p]})
                    # Close the file object
                    f.close()

                with open(os.path.join(data_path, "TAE-Association-pair-non-conditional.csv"), "a") as f:
                    dictwriter_object = DictWriter(f, fieldnames=field_names_kl)
                    # Pass the dictionary as an argument to the Writerow()
                    for p in range(len(non_conditional_pair)):
                        dictwriter_object.writerow({"Method": "Copula-prior", "State": state_name, "County": county_name,
                                                    "Absolute error - Association": dist_cp_prior_non[p], 'Pair': non_conditional_pair[p]})
                    for p in range(len(non_conditional_pair)):
                        dictwriter_object.writerow(
                            {"Method": "Copula+Conditional-prior", "State": state_name, "County": county_name,
                             "Absolute error - Association": dist_combined_prior_non[p], 'Pair': non_conditional_pair[p]})
                    for p in range(len(non_conditional_pair)):
                        dictwriter_object.writerow(
                            {"Method": "Copula-final", "State": state_name, "County": county_name,
                             "Absolute error - Association": dist_cp_final_non[p], 'Pair': non_conditional_pair[p]})
                    for p in range(len(non_conditional_pair)):
                        dictwriter_object.writerow(
                            {"Method": "Copula+Conditional-final", "State": state_name, "County": county_name,
                             "Absolute error - Association": dist_combined_final_non[p], 'Pair': non_conditional_pair[p]})
                    for p in range(len(non_conditional_pair)):
                        dictwriter_object.writerow(
                            {"Method": "Conditional probabilities", "State": state_name, "County": county_name,
                             "Absolute error - Association": dist_conditional_non[p], 'Pair': non_conditional_pair[p]})
                    for p in range(len(non_conditional_pair)):
                        dictwriter_object.writerow(
                            {"Method": "Max Entropy", "State": state_name, "County": county_name,
                             "Absolute error - Association": dist_entropy_final_non[p], 'Pair': non_conditional_pair[p]})
                    for p in range(len(non_conditional_pair)):
                        dictwriter_object.writerow(
                            {"Method": "SynthACS", "State": state_name, "County": county_name,
                             "Absolute error - Association": dist_simulated_non[p], 'Pair': non_conditional_pair[p]})
                    for p in range(len(non_conditional_pair)):
                        dictwriter_object.writerow({"Method": "SynTropy", "State": state_name, "County": county_name,
                                                    "Absolute error - Association":dist_en_cn_final_non[p],
                                                    'Pair': non_conditional_pair[p]})
                    # Close the file object

                    f.close()


def plot_absolute(df):
    d_out = df.loc[~df["Method"].isin(["Copula+Conditional-prior", "Copula-final"]), :]
    # merge_ = d_out
    # merge_['Method'] = merge_['Method'].replace({'Max Entropy': 'Max Entropy - w/o priors'})
    d_out['Method'] = d_out['Method'].replace({'Copula-prior': 'SynC'})
    # merge_['Method'] = merge_['Method'].replace({'Copula-final': 'Copula-entropy'})
    d_out['Method'] = d_out['Method'].replace({'SynTropy': 'Syntropy'})
    d_out['Method'] = d_out['Method'].replace({'Copula+Conditional-final': 'Our approach'})
    d_out['Method'] = d_out['Method'].replace({'Conditional probabilities': 'Conditional'})

    d_out.drop_duplicates(keep="first", inplace=True)

    # df['TAE (log)'] = np.log2(df['TAE'])

    ax = sns.boxplot(y='Absolute error', x='Pair',
                     data=d_out, hue='Method',
                     palette="colorblind", showfliers=False)
    handles = ax.legend_.legendHandles
    labels = [text.get_text() for text in ax.legend_.texts]
    sns.swarmplot(y='Absolute error', x='Pair', hue='Method',
                  data=d_out, palette="colorblind", dodge=True, ax=ax)
    locs_tae, labels_tae = plt.xticks()
    plt.setp(labels_tae, rotation=15)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(17)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)

    plt.legend(handles, labels, fontsize=15)
    plt.tight_layout()
    # ax.set(ylim=(0, 10000))
    plt.show()


conditional = pd.read_csv(os.path.join(data_path, "TAE-Association-pair.csv"))
non_conditional = pd.read_csv(os.path.join(data_path, "TAE-Association-pair-non-conditional.csv"))
plot_absolute(conditional)
plot_absolute(non_conditional)