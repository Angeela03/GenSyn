# The following code retrieves sample data belonging to a county from PUMS and converts it into the required format

from pypums import ACS
import pandas as pd
import os

# Data documentation is available in https://data.census.gov/mdat/#/search?ds=ACSPUMS5Y2018
# https://www.census.gov/topics/income-poverty/poverty/guidance/poverty-measures.html
# ACS Documentation - https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/
# 2018_ACSSubjectDefinitions.pdf
data_path = os.path.join(".", "data")
cols_req = ['SERIALNO', 'DIVISION', 'SPORDER', 'PUMA', 'REGION', 'ST', 'AGEP', 'SEX', 'RAC1P', 'MAR', 'ESR', 'SCHL',
            'POVPIP', 'HICOV', 'NATIVITY', 'MIG']


# function to change numerical attributes to categorical
def to_cat(x):
    # age
    if x["AGEP"] < 15:
        x["age"] = 'under15'
    elif x["AGEP"] >= 15 and x["AGEP"] <= 17:
        x["age"] = '15_17'
    elif x["AGEP"] >= 18 and x["AGEP"] <= 24:
        x["age"] = '18_24'
    elif x["AGEP"] >= 25 and x["AGEP"] <= 29:
        x["age"] = '15_29'
    elif x["AGEP"] >= 30 and x["AGEP"] <= 34:
        x["age"] = '30_34'
    elif x["AGEP"] >= 35 and x["AGEP"] <= 39:
        x["age"] = '35_39'
    elif x["AGEP"] >= 40 and x["AGEP"] <= 44:
        x["age"] = '40_44'
    elif x["AGEP"] >= 45 and x["AGEP"] <= 49:
        x["age"] = '45_49'
    elif x["AGEP"] >= 50 and x["AGEP"] <= 54:
        x["age"] = '50_54'
    elif x["AGEP"] >= 55 and x["AGEP"] <= 59:
        x["age"] = '55_59'
    elif x["AGEP"] >= 60 and x["AGEP"] <= 64:
        x["age"] = '60_64'
    elif x["AGEP"] >= 65 and x["AGEP"] <= 69:
        x["age"] = '65_69'
    elif x["AGEP"] >= 70 and x["AGEP"] <= 74:
        x["age"] = '70_74'
    elif x["AGEP"] >= 75 and x["AGEP"] <= 79:
        x["age"] = '75_79'
    elif x["AGEP"] >= 80 and x["AGEP"] <= 84:
        x["age"] = '80_84'
    elif x["AGEP"] >= 85:
        x["age"] = "85up"

    # gender
    if x["SEX"] == 1:
        x["gender"] = "Male"
    elif x["SEX"] == 2:
        x["gender"] = "Female"

    # marital status

    if x["MAR"] == 1:
        x["marital_status"] = "married"
    elif x["MAR"] == 2:
        x["marital_status"] = "widowed"
    elif x["MAR"] == 3:
        x["marital_status"] = "divorced"
    elif x["MAR"] == 4:
        x["marital_status"] = "mar_apart"
    elif x["MAR"] == 5:
        x["marital_status"] = "never_mar"

    # SCHL

    if x["ESR"] == 0 or x["ESR"] == 3:
        x["emp_status"] = "unemployed"
    elif x["ESR"] == 1 or x["ESR"] == 2 or x["ESR"] == 4 or x["ESR"] == 5:
        x["emp_status"] = "employed"
    elif x["ESR"] == 6:
        x["emp_status"] = "not_in_labor_force"

    # # HICOV
    # if x["HICOV"] == 1:
    #     x["Insurance"] = "Insured"
    # elif x["HICOV"] == 2:
    #     x["Insurance"] = "Uninsured"

    # SCHL
    if x["SCHL"] == 24 or x["SCHL"] == 23 or x["SCHL"] == 22:
        x["edu_attain"] = "grad_deg"
    elif x["SCHL"] == 21:
        x["edu_attain"] = "ba_deg"
    elif x["SCHL"] == 20:
        x["edu_attain"] = "assoc_dec"
    elif x["SCHL"] == 19 or x["SCHL"] == 18:
        x["edu_attain"] = "some_col"
    elif x["SCHL"] == 16:
        x["edu_attain"] = "hs_grad"
    elif x["SCHL"] == 15 or x["SCHL"] == 17:
        x["edu_attain"] = "some_hs"
    elif x["SCHL"] < 15:
        x["edu_attain"] = "lt_hs"

    # Pov status
    if x["POVPIP"] < 100:
        x["pov_status"] = "below_pov_level"
    else:
        x["pov_status"] = "at_above_pov_level"

    # nativity
    if x["NATIVITY"] == 1:
        x["nativity"] = "native"
    elif x["NATIVITY"] == 2:
        x["nativity"] = "foreigner"

    # geog_mobility

    if x["MIG"] == 2:
        x["geog_mobility"] = "moved from abroad"
    elif x["MIG"] == 3:
        x["geog_mobility"] = "different house in us"
    else:
        x["geog_mobility"] = "same house"
    return x


# Get PUMS data
def get_pums(state_name, county_fips, get_county_pums):
    print(state_name)
    acs_data = ACS(year=2018, state=state_name, survey='5-Year')
    acs = acs_data
    df_micro = acs.as_dataframe()

    # Get the corresponding PUMA codes belonging to a county
    get_county_pums["County_code"] = get_county_pums["County_code"].astype(str)
    puma_code = get_county_pums[get_county_pums["County_code"] == county_fips]["PUMA"].values

    # Get data only for those PUMA codes
    data_req = df_micro[df_micro['PUMA'].isin(puma_code)]
    data_req = data_req[cols_req]
    data_final = data_req[['SERIALNO', 'AGEP', 'SEX', 'SCHL', 'MAR', 'POVPIP', 'ESR', 'MIG', 'NATIVITY']]
    dropped_df = data_final.dropna().reset_index(drop=True)

    # Change attributes to categorical
    combine_attr = dropped_df.iloc[:, 1:]
    combine_attr["POVPIP"] = combine_attr["POVPIP"].astype(int)
    to_categorical = combine_attr.apply(lambda x: to_cat(x), axis=1)
    to_categorical = to_categorical.iloc[:, 8:]
    return to_categorical


if __name__ == "__main__":
    county_pums = pd.read_csv(os.path.join(data_path, "puma_county.csv"))
    print(county_pums.columns)
    get_pums("Maryland", "24510", county_pums)
