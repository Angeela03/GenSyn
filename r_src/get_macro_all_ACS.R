memory.limit(size=56000)
gc()
memory.size(max=F)
rm(list = ls())
# .rs.restartR()
library(renv)
renv::init()

library(data.table)
library(synthACS)
library(dplyr)
library(acs)
library(purrr)
library(stringr)

year = 2018
yr_estimate = 5


# geo_make <- geo.make(state ="MD", county="Baltimore County")
# an.error.occured <- FALSE
# tryCatch( { data_SMSM_all <- pull_synth_data(endyear = year, span = yr_estimate, geography = geo_make) }
#           , error = function(e) {an.error.occured <<- TRUE})
# print(an.error.occured)
# if (an.error.occured == TRUE){
#   next;
# }

setwd(".")
data_path <- file.path(".","data_acs")

# data_path <- "C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/data_acs"
# setwd("C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/r_codes/conditional_rpobabilties_generic")

source("pull_acs.R")
api.key.install("b9a6af90f75918cd6f2f005e7be48eec79f0cbf7")

# Read files
read_insurance_data = read.csv(file.path(data_path, "Insurance_by_gender_all_counties.csv"))

states = as.list(unique(read_insurance_data[c("State")]))
print(list(states[1]))

# df_all_counties = data.frame()
# df_norm_all_counties = data.frame()
# for (i in 30:length(states[[1]])){
#   state_name =states[[1]][[i]]
#   print(state_name)}
# length(states[[1]]
for (k in 27:27){
  state_name = states[[1]][[k]]
  print(state_name)
  geo_make <- geo.make(state = trimws(toString(state_name)), county="*")
  print(geo_make)
  an.error.occured <- FALSE
  tryCatch( { data_SMSM_all <- pull_synth_data(endyear = year, span = yr_estimate, geography = geo_make) }
            , error = function(e) {an.error.occured == TRUE})
  print(an.error.occured)
  if (an.error.occured == TRUE){
    next;
  }
  
  FIPS <- paste(data_SMSM_all$geography$state, data_SMSM_all$geography$county, sep = "")
  
  # Get data for that geography
  
  
  library('parallel')
  gc()
  an.error.occured <- FALSE
  
  tryCatch( { data_synthetic <- derive_synth_datasets(data_SMSM_all, leave_cores = 0) }
            , error = function(e) {an.error.occured <<- TRUE})
  print(an.error.occured)
  if (an.error.occured == TRUE){
    next;
  }
  
  # save(data_synthetic, file = "synthetic_all_states.rds")
  
  # data_synthetic<- readRDS("synthetic_maryland.rds")
  # print(class(data_synthetic))
  # 
  data_synthetic_counties <- names(data_synthetic)
  synthetic_final <- marginalize_attr(data_synthetic, varlist = c("ind_income", "race"), marginalize_out = TRUE)
  list_attr = list()
  list_norm_attr = list()
  df = data.frame()
  df_norm = data.frame()
  print(length(data_synthetic_counties))
  list_counties <-list()
  list_fips <-list()
  for (i in 1:length(data_synthetic_counties)){
    list_colnames = list()
    list_values = list()
    list_values_normalized = list()
    
  
    county = data_synthetic_counties[[i]]
    print(county)
    fips <- FIPS[[i]]
    synthetic_county = synthetic_final[[i]]
    prior_data <- synthetic_county[[2]]
    class(prior_data) = "data.frame"
    prior_data$nativity <- as.character(prior_data$nativity)
    prior_data[prior_data["nativity"] == "born_other_state", "nativity"] <- "native"
    prior_data[prior_data["nativity"] == "born_out_us","nativity"] <- "foreigner"
    prior_data[prior_data["nativity"] == "born_state_residence","nativity"] <- "native"
    prior_data[prior_data["nativity"] == "foreigner","nativity"] <- "foreigner"
    prior_data$nativity <- as.factor(prior_data$nativity)
    
    prior_data$geog_mobility <- as.character(prior_data$geog_mobility)
    prior_data["geog_mobility"][prior_data["geog_mobility"] == "same county"] <- "different house in us"
    prior_data["geog_mobility"][prior_data["geog_mobility"] == "same state"] <- "different house in us"
    prior_data["geog_mobility"][prior_data["geog_mobility"] == "diff state"] <- "different house in us"
    prior_data$geog_mobility <- as.factor(prior_data$geog_mobility)
    
    gc()
    prior_data <- prior_data %>%
      group_by( age,gender, marital_status, edu_attain, emp_status, pov_status, geog_mobility,nativity) %>%
      summarise(p = sum(p))
    
    
    class(prior_data) = "micro_synthetic"
    synthetic_county[[2]] = prior_data
    synthetic_county = list(synthetic_county)
    class(synthetic_county) = "synthACS"

    age <- all_geog_constraint_age(synthetic_county, method = "macro.table")
    gender <- all_geog_constraint_gender(synthetic_county, method = "macro.table")
    marital_status <- all_geog_constraint_marital_status(synthetic_county, method = "macro.table")
    edu_attain <- all_geog_constraint_edu(synthetic_county, method = "macro.table")
    emp_status <- all_geog_constraint_employment(synthetic_county, method = "synthetic")
    pov_status <- all_geog_constraint_poverty(synthetic_county, method = "synthetic")
    geog_mobility <- all_geog_constraint_geog_mob(synthetic_county, method = "synthetic")
    nativity <- all_geog_constraint_nativity(synthetic_county, method = "synthetic")
    
    # pov_status_cons <- rev(pov_status[[1]])
    # pov_status_cons = list(pov_status_cons)

    cll <- vector(mode = "list", length = 0)
    cll <- all_geogs_add_constraint(attr_name = "age", attr_total_list = age, macro_micro = synthetic_county)
    cll <- all_geogs_add_constraint(attr_name = "gender", attr_total_list = gender, macro_micro = synthetic_county , constraint_list_list = cll)
    cll <- all_geogs_add_constraint(attr_name = "marital_status", attr_total_list= marital_status, macro_micro = synthetic_county, constraint_list_list= cll)
    cll <- all_geogs_add_constraint(attr_name = "edu_attain", attr_total_list = edu_attain, macro_micro = synthetic_county, constraint_list_list = cll)
    cll <- all_geogs_add_constraint(attr_name = "emp_status", attr_total_list = emp_status, macro_micro = synthetic_county, constraint_list_list = cll)
    cll <- all_geogs_add_constraint(attr_name = "pov_status", attr_total_list = pov_status, macro_micro = synthetic_county, constraint_list_list = cll)
    cll <- all_geogs_add_constraint(attr_name = "geog_mobility", attr_total_list = geog_mobility, macro_micro = synthetic_county, constraint_list_list = cll)
    cll <- all_geogs_add_constraint(attr_name = "nativity", attr_total_list = nativity, macro_micro = synthetic_county, constraint_list_list = cll)
    age_county = cll[[1]]$age
    list_colnames = append(list_colnames,names(age_county))
    list_values =  append(list_values, unname(age_county))
    sum_list_values = Reduce('+', unname(age_county))
    list_values_norm = lapply(unname(age_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))

    gender_county = cll[[1]]$gender
    list_colnames = append(list_colnames,names(gender_county))
    list_values =  append(list_values, unname(gender_county))
    sum_list_values = Reduce('+', unname(gender_county))
    list_values_norm = lapply(unname(gender_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))


    marital_county = cll[[1]]$marital_status
    list_colnames = append(list_colnames,names(marital_county))
    list_values =  append(list_values, unname(marital_county))
    sum_list_values = Reduce('+', unname(marital_county))
    list_values_norm = lapply(unname(marital_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))

    edu_county = cll[[1]]$edu_attain
    list_colnames = append(list_colnames,names(edu_county))
    list_values =  append(list_values, unname(edu_county))
    sum_list_values = Reduce('+', unname(edu_county))
    list_values_norm = lapply(unname(edu_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))

    emp_county = cll[[1]]$emp_status
    list_colnames = append(list_colnames,names(emp_county))
    list_values =  append(list_values, unname(emp_county))
    sum_list_values = Reduce('+', unname(emp_county))
    list_values_norm = lapply(unname(emp_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))

    pov_county = cll[[1]]$pov_status
    list_colnames = append(list_colnames,names(pov_county))
    list_values =  append(list_values, unname(pov_county))
    sum_list_values = Reduce('+', unname(pov_county))
    list_values_norm = lapply(unname(pov_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))
    
    geog_county = cll[[1]]$geog_mobility
    names_geog_county = names(geog_county)
    if (("moved from abroad" %in% names_geog_county) == FALSE){
      geog_county[["moved from abroad"]] = 0
      geog_county<-geog_county[c("different house in us","moved from abroad","same house")]}

    

    list_colnames = append(list_colnames,names(geog_county))
    list_values =  append(list_values, unname(geog_county))
    sum_list_values = Reduce('+', unname(geog_county))
    list_values_norm = lapply(unname(geog_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))

    nat_county = cll[[1]]$nativity
    list_colnames = append(list_colnames,names(nat_county))
    list_values =  append(list_values, unname(nat_county))
    sum_list_values = Reduce('+', unname(nat_county))
    list_values_norm = lapply(unname(nat_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))

    an.error.occured <- FALSE
    tryCatch( {  df<-rbind(df,list_values)
    list_counties <- append(list_counties, county)
    list_fips <- append(list_fips, fips)
    }, error = function(e) {an.error.occured == TRUE})
    print(an.error.occured)
    if (an.error.occured == TRUE){
      next;}
    
    an.error.occured <- FALSE
    tryCatch( {  df_norm<-rbind(df_norm, list_values_normalized)} ,error = function(e) {an.error.occured == TRUE})
    print(an.error.occured)
    if (an.error.occured == TRUE){
      next;}
    names(df_norm) = list_colnames
    names(df) <- list_colnames
   
    gc()
  }
  print(length(list_counties))
  print(length(data_synthetic_counties))
  rownames(df) <- list_counties
  df$FIPS <- list_fips
  df$FIPS = as.numeric(df$FIPS)
  
  rownames(df_norm) <-list_counties
  df_norm$FIPS <- list_fips
  df_norm$FIPS = as.numeric(df_norm$FIPS)
  
  # df_all_counties <- rbind(df_all_counties, df)
  # df_norm_all_counties <- rbind(df_norm_all_counties, df_norm)
  # print(sapply(df_all_counties, class))
  # write.csv(df_all_counties,"All_counties_macro_acs.csv")
  # write.csv(df_norm_all_counties, "All_counties_macro_norm_acs.csv")
  # write.csv(df,"All_counties_macro_acs.csv", append=TRUE)
  # write.csv(df_norm, "All_counties_macro_norm_acs.csv", append=TRUE)
  # 
  write.table(df, "All_counties_macro_acs_mon.csv", sep = ",", col.names = !file.exists("All_counties_macro_acs.csv"), append = T)
  
  write.table(df_norm, "All_counties_macro_norm_acs_mon.csv", sep = ",", col.names = !file.exists("All_counties_macro_norm_acs.csv"), append = T)
  
  
  gc()
}


# print(length(data_synthetic_counties))
# print(cll)

