import pandas as pd
import numpy as np
import os
import sys


data_path = "C:\\Users\\achar\\OneDrive\\Documents\\Project_opiod\\project_final_code_synthetic\\data"
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
gender_acs = ["Male", "Female"]
overdose_data = pd.read_csv(os.path.join(data_path, "od_cdc_wonder.csv"))
insurance_data = pd.read_excel(os.path.join(data_path, "insurance_all.xlsx"))
veteran_data = pd.read_csv(os.path.join(data_path, "veteran_oeps.csv"))


def get_acs():
    overdose_data_cp = overdose_data.copy()
    overdose_data_cp["Five-Year Age Groups Code"] = overdose_data_cp["Five-Year Age Groups Code"].astype(str)
    overdose_data_cp["Five-Year Age Groups Code"] = overdose_data_cp["Five-Year Age Groups Code"].str.replace("15-19", "15-17")
    overdose_data_cp["Five-Year Age Groups Code"] = overdose_data_cp["Five-Year Age Groups Code"].str.replace("20-24", "18-24")
    overdose_data_cp["Five-Year Age Groups Code"] = overdose_data_cp["Five-Year Age Groups Code"].str.replace('-', '_')
    # print(overdose_data.columns)
    # print(np.unique(overdose_data["Five-Year Age Groups Code"].values))
    # print(overdose_data)
    group_by_state = overdose_data_cp.groupby("State")
    df_all = pd.DataFrame()
    for ind, row_state in group_by_state:
        group_by_county = row_state.groupby("County")
        for ind_county, row_county in group_by_county:
            for a in age_acs:
                df_a = row_county.loc[row_county["Five-Year Age Groups Code"] == a, :]
                if len(df_a) != 0:
                    dict_add_od = {"Year": "2018",
                                "County": ind_county,
                                "State": ind,
                                "County Code": df_a["County Code"].values[0],
                                "age": a,
                                "pct": df_a["Deaths"].values[0],
                                "level": "Overdose"
                                }
                    # print(dict_add_od)
                else:
                    dict_add_od = {"Year":"2018",
                    "County":ind_county,
                    "State": ind,
                    "County Code":row_county["County Code"].values[0],
                    "age": a,
                    "pct":0,
                    "level":"Overdose"}
                    # print(dict_add_od)
                df_all = df_all.append(dict_add_od, ignore_index=True)
                print(dict_add_od)

    print(df_all)
    df_all.to_csv(os.path.join(data_path, "Opioid_overdose_all_2018.csv"), index=False)


def format_overdose():
    read_od = pd.read_csv(os.path.join(data_path, "Opioid_overdose_all_2018.csv"))
    read_od["County Code"] = read_od["County Code"].astype(int).astype(str)
    # read_od["State_code"] = read_od["County Code"].str[:-3]
    # read_od["County_code"] = read_od["County Code"].str[-3:]
    # read_od["County_code"] = read_od["County_code"].str.lstrip('0')
    get_age_acs = pd.read_csv("acs_age_total.csv")
    get_age_acs["State_code"] = get_age_acs["State_code"].astype(str)
    get_age_acs["County_code"] = get_age_acs["County_code"].astype(str)
    get_age_acs['County_code'] = get_age_acs['County_code'].apply(lambda x: x.zfill(3))
    get_age_acs["County Code"] = get_age_acs["State_code"] + get_age_acs["County_code"]
    print(read_od.columns)
    print(get_age_acs.columns)
    joined_df  = pd.merge(read_od, get_age_acs[["County Code","pct","Age"]],  how="left", left_on= ["County Code","age"], right_on = ["County Code","Age"])

    print(joined_df)
    print(joined_df.columns)
    joined_df["pct_notoverdose"] = joined_df["pct_y"] - joined_df["pct_x"]
    req_od = joined_df[["County Code","County", "State", "age","Year","level" ]]
    req_od["pct"] =joined_df["pct_x"]
    req_no_od = joined_df[["County Code","County", "State","age","Year"]]
    req_no_od["level"] = len(req_no_od) *["No_overdose"]
    req_no_od["pct"] = joined_df["pct_notoverdose"]
    final_df = req_od.append(req_no_od)
    print(final_df)

    final_df.to_csv(os.path.join(data_path, "Overdose_by_age_all_counties.csv"), index=False)


def format_insurance():
    insurance_cp = insurance_data.copy()
    insurance_cp["sexcat"] = insurance_cp["sexcat"].replace(1, "Male")
    insurance_cp["sexcat"] = insurance_cp["sexcat"].replace(2,  "Female")
    insurance_cp["countyfips"] = insurance_cp["countyfips"].astype(str)
    insurance_cp["statefips"] = insurance_cp["statefips"].astype(str)
    insurance_cp["countyfips"] = insurance_cp["countyfips"].str.rjust(3, "0")
    insurance_cp["FIPS"] = insurance_cp["statefips"] +insurance_cp["countyfips"]
    print(insurance_cp)
    df_all = pd.DataFrame()
    group_by_state = insurance_cp.groupby("state_name")
    for ind, row_state in group_by_state:
        group_by_county = row_state.groupby("county_name")
        for ind_county, row_county in group_by_county:
            for g in gender_acs:
                df_a = row_county.loc[row_county["sexcat"] == g, :]
                if len(df_a) != 0:
                    dict_add_od = {"Year": "2018",
                                   "County": ind_county,
                                   "State": ind,
                                   "County Code": df_a["FIPS"].values[0],
                                   "gender": g,
                                   "pct": df_a["PCTIC"].values[0],
                                   "level": "Insured"
                                   }
                    # print(dict_add_od)
                else:
                    dict_add_od = {"Year": "2018",
                                   "County": ind_county,
                                   "State": ind,
                                   "County Code": row_county["FIPS"].values[0],
                                   "gender": g,
                                   "pct": 0,
                                   "level": "Insured"}
                    # print(dict_add_od)
                df_all = df_all.append(dict_add_od, ignore_index=True)
                print(dict_add_od)
        get_gender_acs = pd.read_csv("acs_gender_total.csv")
        get_gender_acs["State_code"] = get_gender_acs["State_code"].astype(str)
        get_gender_acs["County_code"] = get_gender_acs["County_code"].astype(str)
        get_gender_acs['County_code'] = get_gender_acs['County_code'].apply(lambda x: x.zfill(3))
        get_gender_acs["County Code"] = get_gender_acs["State_code"] + get_gender_acs["County_code"]
        joined_df = pd.merge(df_all, get_gender_acs[["County Code", "pct", "Gender"]], how="left",
                             left_on=["County Code", "gender"], right_on=["County Code", "Gender"])
        joined_df["pct_x"] = joined_df["pct_x"].replace('   . ', np.nan)
        joined_df.to_csv("test_df.csv")
        print(joined_df)
        print(joined_df.columns)
        joined_df["pct_x"] = joined_df["pct_x"].astype(float)
        joined_df["pct_x"] = joined_df["pct_x"] /100
        pct_insured  = round(joined_df["pct_x"] *joined_df["pct_y"])
        joined_df["pct_uninsured"] = joined_df["pct_y"] - pct_insured
        req_ins = joined_df[["County Code", "County", "State", "gender", "Year", "level"]]
        req_ins["pct"] = pct_insured
        req_no_ins = joined_df[["County Code", "County", "State", "gender", "Year"]]
        req_no_ins["level"] = len(req_no_ins) * ["Uninsured"]
        req_no_ins["pct"] = joined_df["pct_uninsured"]
        final_df = req_ins.append(req_no_ins)
        print(final_df)
        final_df.to_csv(os.path.join(data_path,"Insurance_by_gender_all_counties.csv"), index=False)


def format_veteran():
    insurance_cp = insurance_data.copy()
    insurance_cp["countyfips"] = insurance_cp["countyfips"].astype(str)
    insurance_cp["statefips"] = insurance_cp["statefips"].astype(str)
    insurance_cp["countyfips"] = insurance_cp["countyfips"].str.rjust(3, "0")
    insurance_cp["FIPS"] = insurance_cp["statefips"] + insurance_cp["countyfips"]
    veteran_cp = veteran_data.copy()
    veteran_cp["COUNTYFP"] = veteran_cp["COUNTYFP"].astype(str)
    # overdose_data["County Code"] = overdose_data["County Code"].astype(str)
    # print(veteran_cp[["Year", "State", "County", "County Code"]])

    insurance_req =  insurance_cp[["state_name", "county_name", "FIPS"]].drop_duplicates(keep='first')
    veteran_ = pd.merge(veteran_cp,insurance_req, how="left", left_on="COUNTYFP"
                          , right_on="FIPS")
    veteran_ = veteran_[["YEAR", "state_name", "county_name", "FIPS", "TotalPop", "TotalVetPop", "MalePop", "MaleVetPop",
                             "FemalePop", "FemaleVetPop"]]
    # veteran_cp["Male_not_vet"] = veteran_cp["MalePop"] - veteran_cp["MaleVetPop"]
    # veteran_cp["Female_not_vet"] = veteran_cp["FemalePop"] - veteran_cp["FemaleVetPop"]
    # veteran_cp.columns = ["Year",	"State", "County", "County Code", "TotalPop",	"TotalVetPop", "MalePop",
    #                       "Male_VetPop", "FemalePop", "Female_VetPop", "Male_not_vet", "Female_not_vet"]
    veteran_.columns = ["Year",	"State", "County", "County Code", "TotalPop",	"TotalVetPop", "MalePop",
                          "Male_VetPop", "FemalePop", "Female_VetPop"]
    common_columns = veteran_[["Year", "State", "County", "County Code"]]
    df_new = pd.DataFrame()
    for i in ["Male_VetPop", "Female_VetPop"]:
        df_vet = common_columns.copy()
        df_vet["pct"] = veteran_[i]
        df_vet["gender"] = [i.split('_', 1)[0]] * len(df_vet)
        df_vet["level"] = ["Veteran"]*len(df_vet)
        df_new = df_new.append(df_vet)

    get_gender_acs = pd.read_csv("acs_gender_total.csv")
    get_gender_acs["State_code"] = get_gender_acs["State_code"].astype(str)
    get_gender_acs["County_code"] = get_gender_acs["County_code"].astype(str)
    get_gender_acs['County_code'] = get_gender_acs['County_code'].apply(lambda x: x.zfill(3))
    get_gender_acs["County Code"] = get_gender_acs["State_code"] + get_gender_acs["County_code"]
    joined_df = pd.merge(df_new, get_gender_acs[["County Code", "pct", "Gender"]], how="left",
                         left_on=["County Code", "gender"], right_on=["County Code", "Gender"])

    print(joined_df)
    joined_df["pct_notvet"] = joined_df["pct_y"] - joined_df["pct_x"]
    req_vet = joined_df[["County Code", "County", "State", "gender", "Year", "level"]]
    req_vet["pct"] = joined_df["pct_x"]
    req_no_vet = joined_df[["County Code", "County", "State", "gender", "Year"]]
    req_no_vet["level"] = len(req_no_vet) * ["Not_veteran"]
    req_no_vet["pct"] = joined_df["pct_notvet"]
    final_df = req_vet.append(req_no_vet)
    final_df.to_csv(os.path.join(data_path,"Veteran_by_gender_all_counties.csv"), index=False)
    # joined_df["pct_x"] = joined_df["pct_x"].replace('   . ', np.nan)
    # print(joined_df)
    # joined_df.to_csv("test_df.csv", index=False)
    # for i in ["Male_not_vet", "Female_not_vet"]:
    #     df_no_vet = common_columns.copy()
    #     df_no_vet["pct"] = veteran_cp[i]
    #     df_no_vet["gender"] = [i.split('_', 1)[0]] * len(df_no_vet)
    #     df_no_vet["level"] = ["Not veteran"] * len(df_no_vet)
    #     df_new =df_new.append(df_no_vet)
    # df_new.to_csv(os.path.join(data_path,"Veteran_by_gender_all_counties.csv"), index=False)

    # male_vt["pct"] = veteran_cp["MaleVetPop"]
    #
    #
    # male_vt[""]
    # req_vet = veteran_cp[["County Code", "County", "State", "Gender", "Year", "level"]]
    # req_ins["pct"] = (joined_df["pct_x"] * joined_df["pct_y"])
    # req_no_ins = joined_df[["County Code", "County", "State", "Gender", "Year"]]
    # req_no_ins["level"] = len(req_no_ins) * ["Uninsured"]
    # req_no_ins["pct"] = joined_df["pct_uninsured"]
    # final_df = req_ins.append(req_no_ins)

    # final_df.to_csv("Insurance_by_gender_all_counties.csv", index=False)

get_acs()
format_overdose()
format_insurance()
format_veteran()