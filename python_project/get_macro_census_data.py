import pandas as pd
import censusdata
import os

sample = censusdata.search('acs5', 2018,'concept', 'age')


data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data"

censusdata.printtable(censusdata.censustable('acs5', 2015, 'B01001'))

range_49 = [str(i).zfill(3) for i in range(1,50)]
tables_B01001 = ["B01001_"+i+"E" for i in range_49]
data = censusdata.download('acs5', 2018,
                                   censusdata.censusgeo([('state', '*'),
                                                         ('county', '*')]),
                                   tables_B01001)
read_mapping_file = pd.read_excel(os.path.join(data_path,"mapping_age_sex_acs.xlsx"))
dict_mapping = read_mapping_file.set_index('Table').to_dict()['Variable_name']
dict_mapping = {str(k).strip():str(v).strip() for (k,v) in dict_mapping.items()}
print(dict_mapping)

data_col = data.columns
mod_col = [dict_mapping[i] for i in data_col]
data.columns = mod_col

data['Male under15'] = data['Male under5'] + data['Male 5_9'] + data['Male under15']
data['Male 18_24'] = data['Male 18_24'] + data['Male 20'] + data['Male 21'] +data["Male 22_24"]
data['Male 60_64'] = data['Male 60_61'] +data['Male 62_64']
data['Male 65_69'] = data['Male 65_66'] + data["Male 67_69"]

data['Female under15'] = data['Female under5'] + data['Female 5_9'] + data['Female under15']
data['Female 18_24'] = data['Female 18_24'] + data['Female 20'] + data['Female 21'] +data["Female 22_24"]
data['Female 60_64'] = data['Female 60_61'] +data['Female 62_64']
data['Female 65_69'] = data['Female 65_66'] + data["Female 67_69"]

data_final = data.drop(['Male under5','Male 5_9','Male 20','Male 21','Male 22_24', 'Male 60_61','Male 62_64','Male 65_66','Male 67_69'], axis=1)

data_final = data_final.drop(['Female under5','Female 5_9','Female 20','Female 21','Female 22_24', 'Female 60_61','Female 62_64','Female 65_66','Female 67_69'], axis=1)
data_final.reset_index(inplace=True)


data_final["index"] = data_final["index"].astype(str)
data_final["County"] = data_final["index"].str.split(',').str[0]

data_final["State"] = data_final["index"].str.split(',').str[1]
data_final["State"] = data_final["State"].str.split(':').str[0]

data_final["Summary level"] = data_final["index"].str.split(',').str[1]
data_final["Summary level"] = data_final["Summary level"].str.split(':').str[2]

data_final["State_code"] = data_final["index"].str.split(',').str[2]
data_final["State_code"] = data_final["State_code"].str.split(':').str[1]
data_final["State_code"] = data_final["State_code"].str.split('>').str[0]

data_final["County_code"] = data_final["index"].str.split(':').str[4]
data_final["County_code"] = data_final["County_code"].str.split(',').str[0]

gender_groups_acs = [["Male Male", "Female Female", "State", "County", "Summary level", "State_code", "County_code"]]
df_new = pd.DataFrame()


df_male = pd.DataFrame()
df_male["pct"] = data_final["Male Male"]
df_male["Gender"] = len(data_final) *["Male"]
df_male["State"] = data_final["State"]
df_male["County"] = data_final["County"]
df_male["State_code"] = data_final["State_code"]
df_male["County_code"] = data_final["County_code"]
df_female = pd.DataFrame()
df_female["pct"] = data_final["Female Female"]
df_female["Gender"] = len(data_final) * ["Female"]
df_female["State"] = data_final["State"]
df_female["County"] = data_final["County"]
df_female["State_code"] = data_final["State_code"]
df_female["County_code"] = data_final["County_code"]
df_new = df_male.append(df_female)


df_new.to_csv("acs_gender_total.csv", index=False)
print(data_final)

age_acs = ["under15","15_17", "18_24", "25_29",
"30_34",
"35_39",
"40_44",
"45_49",
"50_54",
"55_59",
"60_64",
"65_69",
"70_74",
"75_79","80_84",
"85up"]

df_age = data_final.copy()
for i in age_acs:
    df_age[i] = data_final[list(data_final.filter(regex=i))].sum(axis=1)

df_age_final = pd.DataFrame()
for i in age_acs:
    df_a = pd.DataFrame()
    df_a["pct"] = df_age[i]
    df_a["Age"] = len(df_age) * [i]
    df_a["State"] = df_age["State"]
    df_a["County"] = df_age["County"]
    df_a["State_code"] = df_age["State_code"]
    df_a["County_code"] = df_age["County_code"]
    df_age_final =df_age_final.append(df_a)
print(df_age_final)
df_age_final.to_csv("acs_age_total.csv", index=False)
