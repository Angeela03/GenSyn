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
data_path <- file.path(".","data")

source("pull_acs.R")

# Read variables other than ACS
read_od_data = read.csv(file.path(data_path, "Overdose_by_age_all_counties.csv"))
read_insurance_data = read.csv(file.path(data_path, "Insurance_by_gender_all_counties.csv"))
read_veteran_data = read.csv(file.path(data_path, "Veteran_by_gender_all_counties.csv"))

# Merge OD and Insurance variales
merge_od_insurance <- merge(read_od_data, read_insurance_data, by = 'County.Code', all.x= TRUE)
read_od_data <- select(merge_od_insurance,County.Code, County.x, State.x,  Year.x,     level.x, pct.x, State.y)
names(read_od_data) <- c("County.Code","County", "State",  "Year",     "level", "pct", "State_name")
overdose_level <- read_od_data[read_od_data$level =="Overdose",]
no_od_level <- read_od_data[read_od_data$level =="No_overdose",]

# Get 50 counties in US having high OD count
Summary <- overdose_level %>%
  group_by( State,County, County.Code,State_name) %>%
  summarise(Net = sum(pct))

summary_top_50 <-Summary %>% arrange(desc(Net)) %>% 
  group_by(County, State,County.Code, State_name)
top_50 <- summary_top_50[1:50,]
write.csv(top_50, file.path(data_path, "top_50_od.csv"))

# Save the marginal totals in a file
save_constraints <- function(synthetic_data, od_pct, ins_pct, vet_pct, county_code, county_name){
  # Get the marginals of the ACS variables using the SynthACS package
  age <- all_geog_constraint_age(synthetic_data, method = "macro.table")
  gender <- all_geog_constraint_gender(synthetic_data, method = "macro.table")
  marital_status <- all_geog_constraint_marital_status(synthetic_data, method = "macro.table")
  edu_attain <- all_geog_constraint_edu(synthetic_data, method = "macro.table")
  emp_status <- all_geog_constraint_employment(synthetic_data, method = "synthetic")
  pov_status <- all_geog_constraint_poverty(synthetic_data, method = "synthetic")

  od <- od_pct[od_pct$level == "Overdose",]
  no_od = od_pct[od_pct$level == "No_overdose",]

  ins <- ins_pct[ins_pct$level =="Insured",]
  unins <-ins_pct[ins_pct$level == "Uninsured",]
  
  vet <- vet_pct[vet_pct$level =="Veteran",]
  not_vet <-vet_pct[vet_pct$level == "Not_veteran",]
  
  drug_overdose<-list(c('Overdose' =sum(od$pct), 'No_overdose' = sum(no_od$pct)))
  Insurance <-list(c('Insured' = sum(ins$pct), 'Uninsured' = sum(unins$pct)))
  Veteran <-list(c('Veteran' = sum(vet$pct), 'Not_veteran' = sum(not_vet$pct)))
  names_columns_ <- names(synthetic_data[[1]][[2]])[c(1:9)]
  
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
  print(json_h_)
  write_json(json_h_, file.path(data_path,paste("json_constraint_",county_code,"_",county_name,".json", sep="")))
  return(list(age, gender, marital_status, edu_attain,emp_status,pov_status,drug_overdose, Insurance, Veteran))
}

gc()

# Generate data for the top 50 counties
for(i in 1:nrow(top_50)) {
  row <- top_50[i,]
  county_fips <- trimws(as.character(row["County.Code"][[1]][[1]])) # trimws removes leading and trailing zeros
  county_code = str_sub(county_fips, -3,-1) # takes the substring
  state <- trimws(as.character(row["State"][[1]][[1]]))
  county <- trimws(as.character(row["County"][[1]][[1]]))
  print(county)
  print(state)
  print(as.numeric(county_code))
  
  geo_make <- geo.make(state = state, county=as.numeric(county_code), check=T)
  
  # Get data for that geography
  data_SMSM_all <- pull_synth_data(endyear = year, span = yr_estimate, geography = geo_make)

  # Generating profiles using the conditional probability method for the ACS variables
  library('parallel')
  data_synthetic <- derive_synth_datasets(data_SMSM_all, leave_cores = 0)

  # Prepare the other attributes to add together with the ACS variables
  overdose_county <- select(
    read_od_data[read_od_data$County.Code == as.character(row["County.Code"][[1]][[1]]), ] ,age, pct,level)
  insurance_county <- select(read_insurance_data[read_insurance_data$County.Code == as.character(row["County.Code"][[1]][[1]]), ] , gender,pct,level)
  veteran_county <- read_veteran_data[read_veteran_data$County.Code == as.character(row["County.Code"][[1]][[1]]), ] 
  veteran_county <- na.omit(veteran_county) 

  veteran_county<-select(veteran_county, gender, pct, level)
  rownames(overdose_county) <- NULL;
  rownames(insurance_county) <-NULL;
  rownames(veteran_county) <-NULL;
  
  overdose_list <-list(overdose_county)
  insurance_list <- list(insurance_county)
  veteran_list <- list(veteran_county)

  gc()
 
  # Merge other attributes using the conditional probabilities method based on their conditioning relationships
  merge_<- all_geog_synthetic_new_attribute(data_synthetic, attr_name = 'drug_overdose',
                                                               conditional_vars = c('age'),
                                                               st_list = overdose_list)
  # Add Insurance by age
  gc()
  merge_ <- all_geog_synthetic_new_attribute(merge_,
                                                               attr_name = 'Insurance',
                                                               conditional_vars = c('gender'),
                                                               st_list = insurance_list)
  gc()
  merge_ <- all_geog_synthetic_new_attribute(merge_,
                                                               attr_name = 'Veteran',
                                                               conditional_vars = c('gender'),
                                                               st_list = veteran_list)
  
  gc()
  
  # MArginalize other attributes that are not required
  synthetic_final <- marginalize_attr(merge_, varlist = c("geog_mobility", "nativity", "ind_income", "race"), marginalize_out = TRUE)
  
  # write down the results obtained from the conditional probabilities and also the constraints
  write_name = paste0("prior_",county_fips,"_",county,".csv")
  write.csv(synthetic_final[[1]][[2]],file.path(data_path,write_name), row.names = FALSE)
  cons <- save_constraints(synthetic_final, overdose_county,insurance_county,veteran_county, county_fips, county)
  
}


