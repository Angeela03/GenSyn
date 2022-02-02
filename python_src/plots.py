import pandas as pd
import seaborn as sns
sns.set(rc={"figure.figsize": (8, 5)})
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
# data = pd.read_excel("table_sensitivity_random_v3.xlsx")
# data = d[d["Method"] != "Intermediate"]
# data = d
# data["Method"].replace({"Baltimore only": "Single-level", "Heirachical": "Heirarchical"}, inplace=True)


def merge(d):
    d_out = d.loc[~d["Method"].isin(["Copula+Conditional-prior"]), :]
    # merge_ = d_out
    merge_ = pd.merge(d_out, groupby_pop, on=["County", "State"], how="left")

    # merge_['Method'] = merge_['Method'].replace({'Max Entropy': 'Max Entropy - w/o priors'})
    merge_['Method'] = merge_['Method'].replace({'Copula-prior': 'SynC'})
    # merge_['Method'] = merge_['Method'].replace({'Copula-final': 'Copula-entropy'})
    merge_['Method'] = merge_['Method'].replace({'SynTropy': 'Syntropy'})
    merge_['Method'] = merge_['Method'].replace({'Copula+Conditional-final': 'Our approach'})
    merge_['Method'] = merge_['Method'].replace({'Conditional probabilities': 'Conditional'})
    merge_.drop_duplicates(keep="first", inplace=True)
    # merge_ = merge_[["Max Entropy", "SynC", "SynthACS", "SynTropy","Our approach"]]
    return merge_


def plot_kl(df):
    sns.boxplot(y='KL_divergence', x='Method',
                data=df,
                palette="colorblind")
    sns.swarmplot(y='KL_divergence', x='Method',
                  data=df, palette="colorblind", dodge=True)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=10)
    plt.show()


def plot_kl_all(df):
    df = df.rename(lambda x: 'KL divergence' if x == 'KL_divergence' else x, axis=1)
    ax = sns.boxplot(y='KL divergence', x='Method',
                data=df, hue ='Data',
                palette="deep", order = ["Max Entropy", "Conditional", "SynC", "SynthACS","Syntropy","Our approach"])
    handles = ax.legend_.legendHandles
    labels = [text.get_text() for text in ax.legend_.texts]
    sns.swarmplot(y='KL divergence', x='Method',
                  data=df, hue= "Data", palette="deep", dodge=True, order = ["Max Entropy", "Conditional", "SynC", "SynthACS", "Syntropy","Our approach"], ax=ax)
    locs, labels_kl = plt.xticks()
    # plt.setp(labels_kl, rotation=15)
    # ax.xaxis.label.set_size(16)
    # ax.yaxis.label.set_size(17)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=14)
    # plt.legend(handles, labels, fontsize= 15)
    plt.setp(labels_kl, rotation=15)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(handles, labels, fontsize= 17)
    plt.tight_layout()
    plt.show()


def plot_tae(df):
    sns.boxplot(y='TAE', x='Method',
                data=df, hue='Data',
                palette="colorblind")

    sns.swarmplot(y='TAE', x='Method',
                  data=df, hue='Data', palette="colorblind", dodge=True)

    locs_tae, labels_tae = plt.xticks()
    # plt.setp(labels_tae, rotation=10)
    plt.show()


def plot_tae_all(df):
    # df['TAE (log)'] = np.log2(df['TAE'])

    ax = sns.boxplot(y='TAE', x='Method',
                data=df, hue= "Data",
                palette="deep", showfliers = False, order = ["Max Entropy",  "Conditional","SynC", "SynthACS", "Syntropy","Our approach"])

    handles = ax.legend_.legendHandles
    labels = [text.get_text() for text in ax.legend_.texts]
    sns.swarmplot(y='TAE', x='Method',
                  data=df,hue='Data', palette="deep", dodge=True, order = ["Max Entropy", "Conditional", "SynC", "SynthACS", "Syntropy","Our approach"],ax=ax)
    locs_tae, labels_tae = plt.xticks()
    plt.setp(labels_tae, rotation=15)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(handles, labels, fontsize= 17)
    plt.tight_layout()
    # ax.set(ylim=(0, 10000))
    plt.show()


def plot_norm(df):
    d_out = df.loc[~df["Method"].isin(["Copula+Conditional-prior", "Copula-final"]), :]
    # merge_ = d_out
    # merge_['Method'] = merge_['Method'].replace({'Max Entropy': 'Max Entropy - w/o priors'})
    d_out['Method'] = d_out['Method'].replace({'Copula-prior': 'SynC'})
    # merge_['Method'] = merge_['Method'].replace({'Copula-final': 'Copula-entropy'})
    d_out['Method'] = d_out['Method'].replace({'SynTropy': 'Syntropy'})
    d_out['Method'] = d_out['Method'].replace({'Copula+Conditional-final': 'Our approach'})
    d_out['Method'] = d_out['Method'].replace({'Conditional probabilities': 'Conditional'})
    print(d_out.columns)
    d_out.columns = ['Method', 'State', 'County', 'Frobenius Norm', 'Data']

    d_out.drop_duplicates(keep="first", inplace=True)
    ax = sns.boxplot(y='Frobenius Norm', x='Method',
                     data=d_out, hue='Data',
                     palette="deep", showfliers=False,order = ["Max Entropy", "Conditional", "SynC", "SynthACS", "Syntropy","Our approach"]
                    )
    handles = ax.legend_.legendHandles
    labels = [text.get_text() for text in ax.legend_.texts]
    sns.swarmplot(y='Frobenius Norm', x='Method',
                  data=d_out, hue='Data', palette="deep", dodge=True, ax=ax, order = ["Max Entropy", "Conditional", "SynC", "SynthACS", "Syntropy","Our approach"])
    locs_tae, labels_tae = plt.xticks()
    plt.setp(labels_tae, rotation=15)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(handles, labels, fontsize=17)

    # plt.setp(labels_tae, rotation=15)
    # ax.xaxis.label.set_size(16)
    # ax.yaxis.label.set_size(17)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=14)

    # plt.yticks(fontsize = 13)
    # plt.legend(fontsize="x-large")
    # plt.rc('legend', fontsize='medium')  # using a named size

    # plt.setp(labels_tae, rotation=15)
    # plt.legend(handles, labels, fontsize= 15)
    plt.tight_layout()
    # ax.set(ylim=(0, 10000))
    plt.show()


def plot_scatter_tae(df, type):
    df.columns = ['Method', 'State', 'County', 'TAE', 'County Code', 'Year', 'Population size',
     'Data']
    df = df.rename(columns={'pct': 'Population size'})

    ax = sns.scatterplot(y='TAE', x='Population size',
                    data=df, hue="Method", s=150)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(24)

    handles = ax.legend_.legendHandles
    labels = [text.get_text() for text in ax.legend_.texts]

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(handles, labels, fontsize= 20)
    plt.title(type, fontsize=22)
    plt.tight_layout()

    plt.show()


data_acs_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data_acs"
data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data"

is_data = pd.read_csv(os.path.join(data_path, "Insurance_by_gender_all_counties.csv"))
is_data['County'] = is_data['County'].str.strip()

groupby_pop = is_data.groupby(["County Code",  "County", "State"]).sum().reset_index()
groupby_pop["State"] = groupby_pop["State"].apply(lambda x: x.strip())

d_kl = pd.read_csv(os.path.join(data_path, "kl_div_scores__copula.csv"))
d_kl_acs = pd.read_csv(os.path.join(data_acs_path, "kl_div_scores_acs_copula.csv"))
merge_kl = merge(d_kl)
merge_kl["Data"] = ["Combined"]*len(merge_kl)
merge_kl_acs = merge(d_kl_acs)
merge_kl_acs["Data"] = ["ACS"]*len(merge_kl_acs)
merge_kl_all = merge_kl.append(merge_kl_acs)


d_tae = pd.read_csv(os.path.join(data_path, "tae_scores_copula__.csv"))
d_tae_acs = pd.read_csv(os.path.join(data_acs_path, "tae_scores_copula_acs_.csv"))
merge_tae = merge(d_tae)
merge_tae["Data"] = ["Combined"]*len(merge_tae)
merge_tae_acs  = merge(d_tae_acs)
merge_tae_acs["Data"] = ["ACS"]*len(merge_tae_acs)
merge_tae_all = merge_tae.append(merge_tae_acs)
merge_tae_all.loc[merge_tae_all.Method == "SynC", "TAE"] = 0
merge_tae_all = merge_tae_all[merge_tae_all["TAE"] <= 12500]


d_norm = pd.read_csv(os.path.join(data_path, "Norm_copula.csv"))
d_norm_acs = pd.read_csv(os.path.join(data_acs_path, "Norm_copula.csv"))
d_norm["Data"] = ["Combined"]*len(d_norm)
d_norm_acs["Data"] = ["ACS"]*len(d_norm_acs)
merge_norm_all = d_norm.append(d_norm_acs)
merge_norm_all.to_csv("merge_norm_all.csv", index=False)
plot_norm(merge_norm_all)
# plot_norm(merge_norm_all)

merge_tae_filtered = merge_tae[merge_tae["Method"].isin(["SynthACS", "Syntropy", "Our approach"])]
merge_tae_filtered_acs = merge_tae_acs[merge_tae_acs["Method"].isin([ "SynthACS", "Syntropy", "Our approach"])]

plot_scatter_tae(merge_tae_filtered,"Combined")
plot_scatter_tae(merge_tae_filtered_acs,"ACS")

plot_kl_all(merge_kl_all)
plot_tae_all(merge_tae_all)
# plot_kl(merge_kl)
# plot_kl(merge_kl_acs)

# plot_tae(merge_tae)
# plot_tae(merge_tae_acs)


