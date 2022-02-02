memory.limit()
gc()
memory.size(max=F)
rm(list = ls())
.rs.restartR()
library(data.table)
library(synthACS)
library(dplyr)
library(acs)
library(purrr)

# Define generic variables
county = "Baltimore County"

year = 2018
yr_estimate = 5
# Define variables specific to geographic area
  if (county ==  "Baltimore city"){
    state_name = "MD"
    pct_uninsured = c(26560, 20635)
    pct_insured = c(262131, 305374)
    } else if (county == "Baltimore County"){
      state_name = "MD"
      pct_insured = c(365344, 410833)
      pct_uninsured = c(27077, 24371)}
setwd("C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code/r_codes/conditional_rpobabilties_generic")
# Fetch demographic data from a certain geographical area
geo_make <- geo.make(state=state_name, county=county)
data_SMSM_all <- pull_synth_data(endyear = year, span = yr_estimate, geography = geo_make)
# Derive synthetic datasets for that region
library('parallel')
data_synthetic <- derive_synth_datasets(data_SMSM_all, leave_cores = 0)

# Define overdose by age
overdose_file_name = paste0(county,"_overdose_age.csv")
overdose <- read.csv(overdose_file_name)
print(class(overdose))
rownames(overdose) <- NULL;
print(overdose)
print(overdose$age)
colnames(overdose) <- c('age','pct','level')

overdose$age <- as.factor(overdose$age)
overdose$level <- as.factor(overdose$level)
overdose = setDT(overdose)
overdose_list <-  list(do.call("rbind", list(overdose)))

# Define Insurance by gender
genders = c('Male', 'Female')
levels_i = rep("Insured", length = 2)
insured <- data.table(gender = genders, pct = pct_insured, level = levels_i)
rownames(insured) <- NULL;
levels_uninsured= rep("Uninsured", length = 2)
uninsured <- data.table(gender = genders, pct = pct_uninsured,
                                     level = levels_uninsured)
rownames(uninsured) <- NULL;
insured$gender <- as.factor(insured$gender)
insured$level <- as.factor(insured$level)
uninsured$gender <- as.factor(uninsured$gender)
uninsured$level <- as.factor(uninsured$level)
add_insurance <-  list(do.call("rbind", list(insured, uninsured)))
# Add overdose by age
merge_overdose_insurance <- all_geog_synthetic_new_attribute(data_synthetic,
                                                                            attr_name = 'drug_overdose',
                                                                            conditional_vars = c('age'),
                                                                             st_list = overdose_list)
# Add Insurance by age
merge_overdose_insurance <- all_geog_synthetic_new_attribute(merge_overdose_insurance,
                                                                             attr_name = 'Insurance',
                                                                             conditional_vars = c('gender'),
                                                                             st_list = add_insurance)
save_filename = paste0("merge_overdose_insurance_",county,".rds")
saveRDS(merge_overdose_insurance, file = save_filename)
synthetic_final <- marginalize_attr(merge_overdose_insurance, varlist = c("geog_mobility", "nativity", "ind_income", "race"), marginalize_out = TRUE)
write_name = paste0("prior_",county,".csv")
write.csv(synthetic_final[[1]][[2]],write_name, row.names = FALSE)