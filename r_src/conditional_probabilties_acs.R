# This code runs the conditional probabilities method for the ACS dataset and returns the joint probability distribution p1
# . This is built on top of SynthACS 

library(data.table)
library(synthACS)
library(dplyr)
library(acs)
library(purrr)
library(stringr)
library(hash)
library(jsonlite)

year = 2018
yr_estimate = 5

# Set the required data_path and directory
setwd(".")
data_dir <- file.path(".","data")

source("pull_acs.R")

# Read insurance data
read_insurance_data = read.csv(file.path(data_path, "Insurance_by_gender_all_counties.csv"))
# Read the 50 randomly sampled counties
sample_50 = read.csv(file.path(data_path, "top_50_acs.csv"))

save_constraints <- function(synthetic_data, county_code, county_name){

  # Get the marginals of these variables using the synthACS package
  age <- all_geog_constraint_age(synthetic_data, method = "macro.table")
  gender <- all_geog_constraint_gender(synthetic_data, method = "macro.table")
  marital_status <- all_geog_constraint_marital_status(synthetic_data, method = "macro.table")
  edu_attain <- all_geog_constraint_edu(synthetic_data, method = "macro.table")
  emp_status <- all_geog_constraint_employment(synthetic_data, method = "synthetic")
  pov_status <- all_geog_constraint_poverty(synthetic_data, method = "synthetic")
  geog_mobility <- all_geog_constraint_geog_mob(synthetic_data, method = "synthetic")
  nativity <- all_geog_constraint_nativity(synthetic_data, method = "synthetic")
  names_columns_ <- names(synthetic_data[[1]][[2]])[c(1:8)]
  
  # Write the marginals in a json file so that it can be retrieved in python
  h_ <- hash()
  
  for (i in names_columns_)
  {
    temp_h <- hash()
    temp_h[names(get(i)[[1]])] = get(i)[[1]]
    temp_h <- as.list(temp_h) 
    h_[[i]] <- temp_h
  }
  

  h_ <- as.list(h_) 
  json_h_ <- toJSON(h_)
  write_json(json_h_, file.path(data_path,paste("json_constraint_",county_code,"_",county_name,"_acs.json", sep="")))
  return(list(age, gender, marital_status, edu_attain,emp_status,pov_status, geog_mobility, nativity))
}

gc()

# Generate data for the 50 counties
for(i in 1:nrow(sample_50)) {
  row <- sample_50[i,]
  county_fips <- trimws(as.character(row["County.Code"][[1]][[1]])) # trims removes leading and trailing zeros
  county_code = str_sub(county_fips, -3,-1) # takes the sub string
  state <- trimws(as.character(row["State"][[1]][[1]]))
  county <- trimws(as.character(row["County"][[1]][[1]]))
  print(county)
  print(state)
  print(as.numeric(county_code))
  geo_make <- geo.make(state = state, county=as.numeric(county_code), check=T)
  
  # Get data for that geography
  an.error.occured <- FALSE
  tryCatch( { data_SMSM_all <- pull_synth_data(endyear = year, span = yr_estimate, geography = geo_make) }
            , error = function(e) {an.error.occured == TRUE})
  print(an.error.occured)
  if (an.error.occured == TRUE){
    next;
  }

  # Generating profiles using the conditional probability method for the required ACS variables
  library('parallel')
  data_synthetic <- derive_synth_datasets(data_SMSM_all, leave_cores = 0)
  synthetic_final <- marginalize_attr(data_synthetic, varlist = c("ind_income", "race"), marginalize_out = TRUE)
  prior_data <- data.frame(synthetic_final[[1]][[2]])
  
  # Change some of the attribute categories to make it in sync with the sample data
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

  prior_data <- prior_data %>%
    group_by( age,gender, marital_status, edu_attain, emp_status, pov_status, geog_mobility,nativity) %>%
    summarise(p = sum(p))

  # write down the results obtained from the conditional probabilities and also the constraints
  write_name = paste0("prior_",county_fips,"_",county,"_acs.csv")
  write.csv(prior_data,file.path(data_path,write_name), row.names = FALSE)
  
  class(prior_data) = "micro_synthetic"
  synthetic_final[[1]][[2]] = prior_data
  cons <- save_constraints(synthetic_final, county_fips, county)
}


