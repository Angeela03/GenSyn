memory.limit(size=56000)
gc()
# memory.size(max=F)
rm(list = ls())

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

# data_path <- "C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/data_acs"
# setwd("C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/r_codes/conditional_rpobabilties_generic")
# Read variables to be added other than ACS
#sink("aim1final.txt")
# List files and source each

setwd(".")
data_path <- file.path(".","data_acs")

source("pull_acs.R")


read_insurance_data = read.csv(file.path(data_path, "Insurance_by_gender_all_counties.csv"))
sample_50 = read.csv(file.path(data_path, "top_50_acs.csv"))

# Summary <- read_insurance_data %>%
#   group_by( State,County, County.Code) %>%
#   summarise(Net = sum(pct))
# 
# set.seed(7)
# sample_300 = Summary[sample(nrow(Summary), 300), ]
# summary_300_desc <-sample_300 %>% arrange(desc(Net)) %>% 
#   group_by(State,County, County.Code) 
# sample_50 <-  summary_300_desc[1:50,]
# write.csv(sample_50, file.path(data_path, "top_50_acs.csv"), row.names = FALSE)


save_constraints <- function(synthetic_data, county_code, county_name){

  # Get the marginals of these variables
  age <- all_geog_constraint_age(synthetic_data, method = "macro.table")
  gender <- all_geog_constraint_gender(synthetic_data, method = "macro.table")
  marital_status <- all_geog_constraint_marital_status(synthetic_data, method = "macro.table")
  edu_attain <- all_geog_constraint_edu(synthetic_data, method = "macro.table")
  emp_status <- all_geog_constraint_employment(synthetic_data, method = "synthetic")
  pov_status <- all_geog_constraint_poverty(synthetic_data, method = "synthetic")
  geog_mobility <- all_geog_constraint_geog_mob(synthetic_data, method = "synthetic")
  nativity <- all_geog_constraint_nativity(synthetic_data, method = "synthetic")
  names_columns_ <- names(synthetic_data[[1]][[2]])[c(1:8)]
  #Write the marginals in a json file so that it can be retrieved in python
  library(hash)
  h_ <- hash()
  
  for (i in names_columns_)
    # { str_i = paste(i, "_county_code", "_county_name", sep="")
  {
    temp_h <- hash()
    temp_h[names(get(i)[[1]])] = get(i)[[1]]
    temp_h <- as.list(temp_h) 
    h_[[i]] <- temp_h
  }
  
  library(jsonlite)
  
  h_ <- as.list(h_) 
  # print(h_)
  json_h_ <- toJSON(h_)
  print(json_h_)
  write_json(json_h_, file.path(data_path,paste("json_constraint_",county_code,"_",county_name,"_acs.json", sep="")))
  return(list(age, gender, marital_status, edu_attain,emp_status,pov_status, geog_mobility, nativity))
}
gc()
# nrow(sample_50)
# Generate data for the top 50 counties
for(i in 25:nrow(sample_50)) {
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
    # geography <-geo_make
  # endyear <- year
  # span <-yr_estimate
  # 
  
  library('parallel')
  data_synthetic <- derive_synth_datasets(data_SMSM_all, leave_cores = 0)
  # save_filename = paste0("synthetic_conditional_",county,".rds")
  # saveRDS(merge_overdose_insurance, file = save_filename)
  synthetic_final <- marginalize_attr(data_synthetic, varlist = c("ind_income", "race"), marginalize_out = TRUE)
  prior_data <- data.frame(synthetic_final[[1]][[2]])
  

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

  
  write_name = paste0("prior_",county_fips,"_",county,"_acs.csv")
  write.csv(prior_data,file.path(data_path,write_name), row.names = FALSE)
  
  class(prior_data) = "micro_synthetic"
  synthetic_final[[1]][[2]] = prior_data
  gc()
  cons <- save_constraints(synthetic_final, county_fips, county)
  gc()

  list_cll = list( "age", "gender", "marital_status", "edu_attain","emp_status","pov_status","geog_mobility","nativity")
  pov_status_cons <- unname(rev(cons[[6]][[1]]))
  names(pov_status_cons) <- c("below_pov_level","at_above_pov_level")     
  pov_status_cons = list(pov_status_cons)

  cll <- vector(mode = "list", length = 0)
  cll <- all_geogs_add_constraint(attr_name = "age", attr_total_list = cons[[1]], macro_micro = synthetic_final)
  cll <- all_geogs_add_constraint(attr_name = "gender", attr_total_list = cons[[2]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "marital_status", attr_total_list= cons[[3]], macro_micro = synthetic_final, constraint_list_list= cll)
  cll <- all_geogs_add_constraint(attr_name = "edu_attain", attr_total_list = cons[[4]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "emp_status", attr_total_list = cons[[5]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "pov_status", attr_total_list = pov_status_cons, macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "geog_mobility", attr_total_list = cons[[7]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "nativity", attr_total_list = cons[[8]], macro_micro = synthetic_final, constraint_list_list = cll)
  
  opt_simulated <- all_geog_optimize_microdata(synthetic_final, seed = 12L,
                                               constraint_list_list = cll, p_accept = 0.4, max_iter = 1000L,
                                               verbose = TRUE)
  # print(opt_simulated)
  simulated_fit <- get_best_fit(opt_simulated, geography = county)
  simulated_tae <- get_final_tae(opt_simulated, geography = county)
  print(simulated_tae)
  write_name = paste0("simulated_",county_fips,"_",county,"_acs.csv")
  write.csv(simulated_fit,file.path(data_path,write_name), row.names = FALSE)
  gc()
  
  
  
}


