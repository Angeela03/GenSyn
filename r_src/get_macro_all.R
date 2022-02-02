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


# data_path <- "C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/data"
# setwd("C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/r_codes/conditional_rpobabilties_generic")

# geo_make <- geo.make(state ="MD", county="Baltimore County")
# an.error.occured <- FALSE
# tryCatch( { data_SMSM_all <- pull_synth_data(endyear = year, span = yr_estimate, geography = geo_make) }
#           , error = function(e) {an.error.occured <<- TRUE})
# print(an.error.occured)
# if (an.error.occured == TRUE){
#   next;
# }

setwd(".")
data_path <- file.path(".","data")

source("pull_acs.R")

api.key.install("b9a6af90f75918cd6f2f005e7be48eec79f0cbf7")

# Read files
read_od_data = read.csv(file.path(data_path, "Overdose_by_age_all_counties.csv"))
read_insurance_data = read.csv(file.path(data_path, "Insurance_by_gender_all_counties.csv"))
read_veteran_data = read.csv(file.path(data_path, "Veteran_by_gender_all_counties.csv"))


states = as.list(unique(read_od_data[c("State")]))

df_all_counties = data.frame()
df_norm_all_counties = data.frame()

# for (i in 30:length(states[[1]])){
#   state_name =states[[1]][[i]]
#   print(state_name)}

for (k in 1:length(states[[1]])){
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


  gc()
  
  age <- all_geog_constraint_age(data_synthetic, method = "macro.table")
  gender <- all_geog_constraint_gender(data_synthetic, method = "macro.table")
  marital_status <- all_geog_constraint_marital_status(data_synthetic, method = "macro.table")
  edu_attain <- all_geog_constraint_edu(data_synthetic, method = "macro.table")
  emp_status <- all_geog_constraint_employment(data_synthetic, method = "synthetic")
  pov_status <- all_geog_constraint_poverty(data_synthetic, method = "synthetic")
  cll <- vector(mode = "list", length = 0)
  cll <- all_geogs_add_constraint(attr_name = "age", attr_total_list = age, macro_micro = data_synthetic)
  cll <- all_geogs_add_constraint(attr_name = "gender", attr_total_list = gender, macro_micro = data_synthetic , constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "marital_status", attr_total_list= marital_status, macro_micro = data_synthetic, constraint_list_list= cll)
  cll <- all_geogs_add_constraint(attr_name = "edu_attain", attr_total_list = edu_attain, macro_micro = data_synthetic, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "emp_status", attr_total_list = emp_status, macro_micro = data_synthetic, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "pov_status", attr_total_list = pov_status, macro_micro = data_synthetic, constraint_list_list = cll)
  list_attr = list()
  list_norm_attr = list()
  df = data.frame()
  df_norm = data.frame()

    for (i in 1:length(data_synthetic_counties)){
    list_colnames = list()
    list_values = list()
    list_values_normalized = list()
  
    
    county = data_synthetic_counties[[i]]
    fips <- FIPS[[i]]
    
    age_county = cll[[i]]$age
    list_colnames = append(list_colnames,names(age_county))
    list_values =  append(list_values, unname(age_county))
    sum_list_values = Reduce('+', unname(age_county))
    list_values_norm = lapply(unname(age_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))
  
  
    gender_county = cll[[i]]$gender
    list_colnames = append(list_colnames,names(gender_county))
    list_values =  append(list_values, unname(gender_county))
    sum_list_values = Reduce('+', unname(gender_county))
    list_values_norm = lapply(unname(gender_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))
  
  
    marital_county = cll[[i]]$marital_status
    list_colnames = append(list_colnames,names(marital_county))
    list_values =  append(list_values, unname(marital_county))
    sum_list_values = Reduce('+', unname(marital_county))
    list_values_norm = lapply(unname(marital_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))
  
    edu_county = cll[[i]]$edu_attain
    list_colnames = append(list_colnames,names(edu_county))
    list_values =  append(list_values, unname(edu_county))
    sum_list_values = Reduce('+', unname(edu_county))
    list_values_norm = lapply(unname(edu_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))
  
    emp_county = cll[[i]]$emp_status
    list_colnames = append(list_colnames,names(emp_county))
    list_values =  append(list_values, unname(emp_county))
    sum_list_values = Reduce('+', unname(emp_county))
    list_values_norm = lapply(unname(emp_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))
  
    pov_county = cll[[i]]$pov_status
    list_colnames = append(list_colnames,names(pov_county))
    list_values =  append(list_values, unname(pov_county))
    sum_list_values = Reduce('+', unname(pov_county))
    # print(length(list_values_norm))
    list_values_norm = lapply(unname(pov_county), function(x){x/sum_list_values})
    list_values_normalized = append(list_values_normalized, unname(list_values_norm))

    od_fips <- read_od_data[read_od_data$County.Code==fips,]
    od_names <- list("Overdose", "No_overdose")
    if (nrow(od_fips) ==0){
      od_values <- list(NA,NA)
      od_values_norm <-list(NA, NA)
      
    }
    else{
      od =  od_fips %>%
        group_by(County.Code, level) %>%
        summarize(pct=sum(pct))
      od_label <- od[od["level"]=="Overdose","pct"][[1]]
      no_od_label <- od[od["level"]=="No_overdose","pct"][[1]]
      od_values <- list(od_label, no_od_label)
      sum_od_values <- Reduce(`+`, od_values)
      od_values_norm = lapply(od_values, function(x){x/sum_od_values})
    }
    list_colnames = append(list_colnames,od_names)
    list_values =  append(list_values, od_values)
    list_values_normalized = append(list_values_normalized, od_values_norm)
    
    is_fips <- read_insurance_data[read_insurance_data$County.Code==fips,]
    is_names <- list("Insured", "Uninsured")
    if (nrow(is_fips) ==0){
      is_values <- list(NA,NA)
      is_values_norm <-list(NA, NA)
      
    }
    else{
      is =  is_fips %>%
        group_by(County.Code, level) %>%
        summarize(pct=sum(pct))
      is_label <- is[is["level"]=="Insured","pct"][[1]]
      no_is_label <- is[is["level"]=="Uninsured","pct"][[1]]
      is_values <- list(is_label, no_is_label)
      sum_is_values <- Reduce(`+`, is_values)
      is_values_norm = lapply(is_values, function(x){x/sum_is_values})
    }
    list_colnames = append(list_colnames,is_names)
    list_values =  append(list_values, is_values)
    list_values_normalized = append(list_values_normalized, is_values_norm)
    
    vt_fips <- read_veteran_data[read_veteran_data$County.Code==fips,]
    vt_fips <- na.omit(vt_fips) 
    
    vt_names <- list("Veteran", "Not_veteran")
    if (nrow(vt_fips) ==0){
      vt_values <- list(NA,NA)
      vt_values_norm <-list(NA, NA)
      
    }
    else{
      vt =  vt_fips %>%
        group_by(County.Code, level) %>%
        summarize(pct=sum(pct))
      vt_label <- vt[vt["level"]=="Veteran","pct"][[1]]
      no_vt_label <- vt[vt["level"]=="Not_veteran","pct"][[1]]
      vt_values <- list(vt_label, no_vt_label)
      sum_vt_values <- Reduce(`+`, vt_values)
      vt_values_norm = lapply(vt_values, function(x){x/sum_vt_values})
    }
    
  
    list_colnames = append(list_colnames,vt_names)
    list_values =  append(list_values,vt_values)
    list_values_normalized = append(list_values_normalized, vt_values_norm)
    
    df_norm<-rbind(df_norm, list_values_normalized)
    names(df_norm) = list_colnames
    df<-rbind(df,list_values)
    names(df) <- list_colnames
  
  }
  
  rownames(df) <- data_synthetic_counties
  df$FIPS <- FIPS
  
  rownames(df_norm) <-data_synthetic_counties
  df_norm$FIPS <- FIPS
  df_all_counties <- rbind(df_all_counties, df)
  df_norm_all_counties <- rbind(df_norm_all_counties, df_norm)
  write.csv(df_all_counties,"All_counties_macro.csv")
  write.csv(df_norm_all_counties, "All_counties_macro_norm.csv")
  gc()
}


  # print(length(data_synthetic_counties))
  # print(cll)
  
