import pandas as pd
import seaborn as sns
sns.set(rc={"figure.figsize": (10, 5)})
import matplotlib.pyplot as plt
import os
import sys
import numpy as np


def process_df(df):
    df = df[df["Method"] == "Copula+Conditional-final"]
    group_by_ = df.groupby(['Method', 'State', 'County']).max().reset_index()
    val = group_by_["Count"].value_counts().index[0]

    group_ = df.groupby(['Method', 'State', 'County'])
    df_all = pd.DataFrame(columns=df.columns)
    for i, row in group_:
        if len(row) > val:
            row = row.iloc[:val, :]
        else:
            diff = val - len(row)
            last_row = row.iloc[-1, :]
            for j in range(1, diff + 1):
                new_row = last_row.copy()
                new_row["Count"] = len(row) + 1
                row = row.append(new_row)
        print(len(row))
        df_all = df_all.append(row)
    df_all = df_all.reset_index(drop=True)

    return df_all


def plot_kl(df, type):
    # sns.swarmplot(y='KL_divergence', x='Method',
    #               data=df, palette="colorblind", dodge=True)
    # locs, labels = plt.xticks()
    # plt.setp(labels, rotation=10)
    if type == "combined":
        our_approach = df[df["Count"]<=9]
        ax = sns.lineplot(y='KL_divergence', x='Count',
                data=our_approach, hue="County", legend=False)
        ax.set_xticks(range(0, 10))
        plt.title("Combined", fontsize = 16)

    else:
        our_approach = df[df["Count"]<=5]
        ax = sns.lineplot(y='KL_divergence', x='Count',
                data=our_approach, hue="County", legend=False)
        ax.set_xticks(range(0, 6))
        plt.title("ACS", fontsize = 16)

    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)

    plt.show()
    plt.close()


def plot_tae(df):
    if type == "combined":
        our_approach = df[df["Count"]<=9]
        ax = sns.lineplot(y='TAE', x='Count',
                data=our_approach, hue="County", legend=False)
        ax.set_xticks(range(0, 10))
        plt.title("Combined", fontsize = 16)

    else:
        our_approach = df[df["Count"]<=5]
        ax = sns.lineplot(y='TAE', x='Count',
                data=our_approach, hue="County", legend=False)
        ax.set_xticks(range(0, 6))
        plt.title("ACS", fontsize= 16)

    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)
    # df_plot = df[df["TAE"]<20000]
    # ax = sns.lineplot(y='TAE', x='Count',
    #             data=df, hue='County', legend=False)
    # ax.set_xticks(range(1, 11))

    plt.show()
    plt.close()


def plot_infrequent(df, type):
    actual = df[df["Data"] =="Actual"].reset_index(drop=True)
    our_approach =  df[df["Data"] =="Our approach"].reset_index(drop=True)

    per_diff = (np.abs(actual["TAE"] - our_approach["TAE"])/(actual["TAE"]))*100

    our_approach["% Difference: Observed|Expected"] = per_diff

    if type == "combined":
        our_approach = our_approach[our_approach["Count"]<=9]
        ax = sns.lineplot(y="% Difference: Observed|Expected", x='Count',
                          data=our_approach, hue='County', legend=False)
        ax.set_xticks(range(0, 10))
        plt.title("Combined", fontsize = 16)
    else:
        our_approach = our_approach[our_approach["Count"]<=5]
        ax = sns.lineplot(y="% Difference: Observed|Expected", x='Count',
                          data=our_approach, hue='County', legend=False)
        ax.set_xticks(range(0, 6))
        plt.title("ACS", fontsize= 16)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)



    plt.show()
    plt.close()


data_acs_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data_acs"
data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data"

is_data = pd.read_csv(os.path.join(data_path, "Insurance_by_gender_all_counties.csv"))
is_data['County'] = is_data['County'].str.strip()

groupby_pop = is_data.groupby(["County Code",  "County", "State"]).sum().reset_index()

d_kl = pd.read_csv(os.path.join(data_path, "kl_div_scores_tau_copula.csv"))
d_kl_acs = pd.read_csv(os.path.join(data_acs_path, "kl_div_scores_tau_acs_copula.csv"))

# d_kl_all = process_df(d_kl)
d_kl = d_kl[d_kl["Method"] == "Copula+Conditional-final"].reset_index()
mask = d_kl['KL_divergence'] != np.inf
d_kl.loc[~mask, 'KL_divergence'] = d_kl.loc[mask, 'KL_divergence'].max()


d_kl_acs = d_kl_acs[d_kl_acs["Method"] == "Copula+Conditional-final"].reset_index()
mask_acs = d_kl_acs['KL_divergence'] != np.inf
d_kl_acs.loc[~mask_acs, 'KL_divergence'] = d_kl_acs.loc[mask_acs, 'KL_divergence'].max()

# d_kl_all = d_kl_all[d_kl_all["KL_divergence"] < 0.86]

# d_kl_acs_all = process_df(d_kl_acs)

d_tae = pd.read_csv(os.path.join(data_path, "tae_scores_copula_tau_.csv"))
d_tae_acs = pd.read_csv(os.path.join(data_acs_path, "tae_scores_copula_tau_acs_.csv"))

d_tae = d_tae[d_tae["Method"] == "Copula+Conditional-final"].reset_index()
d_tae.loc[d_tae['TAE']>=10000, 'TAE'] = 10000


d_tae_acs = d_tae_acs[d_tae_acs["Method"] == "Copula+Conditional-final"].reset_index()
d_tae_acs.loc[d_tae_acs['TAE']>=10000, 'TAE'] = 10000


plot_tae(d_tae)
plot_tae(d_tae_acs)

# d_tae_all = process_df(d_tae)
# d_tae_all["TAE"] = d_tae_all["TAE"].astype(float)
# d_tae_acs_all = process_df(d_tae_acs)
# d_tae_acs_all["TAE"] = d_tae_acs_all["TAE"].astype(float)


data_infrequent = pd.read_csv(os.path.join(data_path, "Infrequent_copula_tau_.csv"))
data_infrequent_acs = pd.read_csv(os.path.join(data_acs_path, "Infrequent_copula_tau_acs_.csv"))

print(data_infrequent)
print(data_infrequent_acs)
plot_infrequent(data_infrequent, "combined")
plot_infrequent(data_infrequent_acs, "acs")

print(d_kl)
plot_kl(d_kl, "combined")
plot_kl(d_kl_acs, "acs")

# d_tae_all = d_tae_all[~((d_tae_all["Count"] == 1) & (d_tae_all["TAE"]>2000))]
# plot_tae(d_tae_all)
# plot_tae(d_tae_acs_all)
# #
#
# merge_kl = merge(d_kl)
# merge_kl["Data"] = ["ACS + Others"]*len(merge_kl)
# merge_kl_acs = merge(d_kl_acs)
# merge_kl_acs["Data"] = ["ACS"]*len(merge_kl_acs)
# merge_kl_all = merge_kl.append(merge_kl_acs)
#
#

#
# merge_tae = merge(d_tae)
# merge_tae["Data"] = ["ACS + Others"]*len(merge_tae)
# merge_tae_acs  = merge(d_tae_acs)
# merge_tae_acs["Data"] = ["ACS"]*len(merge_tae_acs)
# merge_tae_all = merge_tae.append(merge_tae_acs)
#
# # group_by_method = merge_tae.groupby(["Method"])
# # for i, row in group_by_method:
# #     plt.scatter(row["pct"], row["TAE"])
# #     # m,b = np.polyfit(row["pct"].values, row["TAE"].values, 1)
# #     # plt.plot(row["pct"], m * row["pct"] + b)
# # plt.show()
# # # merge_tae.plot.scatter(x= "pct", y="TAE", c="Method")
# sns.scatterplot(y='TAE', x='pct',
#                  data=merge_tae,
#                  palette="colorblind", hue="Method")
# plt.show()
#
# sns.scatterplot(y='TAE', x='pct',
#                  data=merge_tae_acs,
#                  palette="colorblind", hue="Method")
# plt.show()
#
# plot_kl_all(merge_kl_all)
# plot_tae_all(merge_tae_all)
# # plot_kl(merge_kl)
# # plot_kl(merge_kl_acs)
# #
# # plot_tae(merge_tae)
# # plot_tae(merge_tae_acs)
#
#
