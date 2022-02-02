memory.limit(size=56000)
gc()
memory.size(max=F)
rm(list = ls())
# .rs.restartR()


library(data.table)
library(synthACS)
library(dplyr)
library(acs)
library(purrr)
library(stringr)

year = 2018
yr_estimate = 5
# # Define variables specific to geographic area
# if (county ==  "Baltimore city"){
#   state_name = "MD"
#   pct_uninsured = c(26560, 20635)
#   pct_insured = c(262131, 305374)
# } else if (county == "Baltimore County"){
#   state_name = "MD"
#   pct_insured = c(365344, 410833)
#   pct_uninsured = c(27077, 24371)}
data_path <- "C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/data"
setwd("C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/r_codes/conditional_rpobabilties_generic")
# Read variables to be added other than ACS
#sink("aim1final.txt")
# List files and source each
source("pull_acs.R")

read_od_data = read.csv(file.path(data_path, "Overdose_by_age_all_counties.csv"))
read_insurance_data = read.csv(file.path(data_path, "Insurance_by_gender_all_counties.csv"))
read_veteran_data = read.csv(file.path(data_path, "Veteran_by_gender_all_counties.csv"))

merge_od_insurance <- merge(read_od_data, read_insurance_data, by = 'County.Code', all.x= TRUE)
read_od_data <- select(merge_od_insurance,County.Code, County.x, State.x,  Year.x,     level.x, pct.x, State.y)
names(read_od_data) <- c("County.Code","County", "State",  "Year",     "level", "pct", "State_name")
overdose_level <- read_od_data[read_od_data$level =="Overdose",]
no_od_level <- read_od_data[read_od_data$level =="No_overdose",]

print(overdose_level)
print(no_od_level)

# Get 50 counties in US having high OD count
Summary <- overdose_level %>%
  group_by( State,County, County.Code,State_name) %>%
  summarise(Net = sum(pct))

summary_top_50 <-Summary %>% arrange(desc(Net)) %>% 
  group_by(County, State,County.Code, State_name)
top_50 <- summary_top_50[1:50,]
print(top_50)
write.csv(top_50, file.path(data_path, "top_50_od.csv"))


save_constraints <- function(synthetic_data, od_pct, ins_pct, vet_pct, county_code, county_name){
  print(od_pct)
  print(ins_pct)
  print(vet_pct)
  # Get the marginals of these variables
  age <- all_geog_constraint_age(synthetic_data, method = "macro.table")
  gender <- all_geog_constraint_gender(synthetic_data, method = "macro.table")
  marital_status <- all_geog_constraint_marital_status(synthetic_data, method = "macro.table")
  edu_attain <- all_geog_constraint_edu(synthetic_data, method = "macro.table")
  emp_status <- all_geog_constraint_employment(synthetic_data, method = "synthetic")
  pov_status <- all_geog_constraint_poverty(synthetic_data, method = "synthetic")

  od <- od_pct[od_pct$level == "Overdose",]
  no_od = od_pct[od_pct$level == "No_overdose",]
  print(no_od)
  
  ins <- ins_pct[ins_pct$level =="Insured",]
  unins <-ins_pct[ins_pct$level == "Uninsured",]
  
  vet <- vet_pct[vet_pct$level =="Veteran",]
  not_vet <-vet_pct[vet_pct$level == "Not_veteran",]
  
  drug_overdose<-list(c('Overdose' =sum(od$pct), 'No_overdose' = sum(no_od$pct)))
  Insurance <-list(c('Insured' = sum(ins$pct), 'Uninsured' = sum(unins$pct)))
  Veteran <-list(c('Veteran' = sum(vet$pct), 'Not_veteran' = sum(not_vet$pct)))
  names_columns_ <- names(synthetic_data[[1]][[2]])[c(1:9)]
  
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
  write_json(json_h_, file.path(data_path,paste("json_constraint_",county_code,"_",county_name,".json", sep="")))
  return(list(age, gender, marital_status, edu_attain,emp_status,pov_status,drug_overdose, Insurance, Veteran))
  }
gc()
# Generate data for the top 50 counties
for(i in 1:nrow(top_50)) {
  row <- top_50[5,]
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
  # geography <-geo_make
  # endyear <- year
  # span <-yr_estimate
  # 

  library('parallel')
  data_synthetic <- derive_synth_datasets(data_SMSM_all, leave_cores = 0)

  # Extract overdose data for los angeles
  overdose_county <- select(
    read_od_data[read_od_data$County.Code == as.character(row["County.Code"][[1]][[1]]), ] ,age, pct,level)
  insurance_county <- select(read_insurance_data[read_insurance_data$County.Code == as.character(row["County.Code"][[1]][[1]]), ] , gender,pct,level)
  veteran_county <- read_veteran_data[read_veteran_data$County.Code == as.character(row["County.Code"][[1]][[1]]), ] 
  print(veteran_county)
  veteran_county <- na.omit(veteran_county) 
  print(veteran_county)
  # veteran_county<-veteran_county[!duplicated(veteran_county),] 
  veteran_county<-select(veteran_county, gender, pct, level)
  rownames(overdose_county) <- NULL;
  rownames(insurance_county) <-NULL;
  rownames(veteran_county) <-NULL;
  
  overdose_list <-list(overdose_county)
  insurance_list <- list(insurance_county)
  veteran_list <- list(veteran_county)
  print(overdose_list)
  print(veteran_list)
  gc()
  # print(sum(overdose_county$pct))
  # print(sum(insurance_county$pct))
  # print(sum(veteran_county$pct))
 
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
  
  
  # save_filename = paste0("synthetic_conditional_",county,".rds")
  # saveRDS(merge_overdose_insurance, file = save_filename)
  synthetic_final <- marginalize_attr(merge_, varlist = c("geog_mobility", "nativity", "ind_income", "race"), marginalize_out = TRUE)
  print(synthetic_final)
  write_name = paste0("prior_",county_fips,"_",county,".csv")
  write.csv(synthetic_final[[1]][[2]],file.path(data_path,write_name), row.names = FALSE)

  gc()
  cons <- save_constraints(synthetic_final, overdose_county,insurance_county,veteran_county, county_fips, county)
  print(cons)
  gc()
  print(synthetic_final)
  list_cll = list( "age", "gender", "marital_status", "edu_attain","emp_status","pov_status","drug_overdose", "Insurance", "Veteran")
  print(list_cll)

  # print(class(cons[[1]][[1]]))
  # cll <-vector(mode = "list", length = 0)
  # 
  # print(synthetic_final)
  # cll <- all_geogs_add_constraint(attr_name = "age", attr_total_list = cons[[1]], macro_micro = synthetic_final)
  # for (i in 2:length(list_cll)){
  #   names(cons[[i]]) <- "Baltimore city, Maryland"
  #   # print(cons[[i]])
  #   # 
  #   # print(class(cons[[i]]))
  #   # print(list_cll[[i]][[1]])
  #   # print(list(cons[[i]][[1]]))
  #   print(cons[[i]])
  #   print(class(cons[[i]][[1]]))
  #   names <- names(cons[[i]][[1]])
  #   print(names)
  #   # print(cons[[i]][[1]])
  #   cons[[i]][[1]]<-as.numeric(cons[[i]][[1]])
  #   names(cons[[i]][[1]]) <-names
  #   print(class(cons[[i]][[1]]))
  #   cll <- all_geogs_add_constraint(attr_name = list_cll[[i]][[1]], attr_total_list = cons[[i]], macro_micro = synthetic_final, constraint_list_list = cll)
  #   # 
    
  #}
  # print(cons[[6]])
  # print(cons[[7]])
  # names(cons[[7]]) <- "Baltimore city, Maryland"
  # names <- names(cons[[i]][[1]])
  # #   print(names)
  # #   # print(cons[[i]][[1]])
  # cons[[7]][[1]]<-as.numeric(cons[[i]][[1]])
  # names(cons[[7]][[1]]) <-names
  # print(synthetic_final[[1]][[2]])
  print(names(cons[[7]][[1]]))
  print(names(cons[[8]][[1]]))
  print(cons[[9]])
  cll <- vector(mode = "list", length = 0)
  print(synthetic_final)
  cll <- all_geogs_add_constraint(attr_name = "age", attr_total_list = cons[[1]], macro_micro = synthetic_final)
  cll <- all_geogs_add_constraint(attr_name = "gender", attr_total_list = cons[[2]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "marital_status", attr_total_list= cons[[3]], macro_micro = synthetic_final, constraint_list_list= cll)
  cll <- all_geogs_add_constraint(attr_name = "edu_attain", attr_total_list = cons[[4]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "emp_status", attr_total_list = cons[[5]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "pov_status", attr_total_list = cons[[6]], macro_micro = synthetic_final, constraint_list_list = cll)
  #cll <- all_geogs_add_constraint(attr_name = "nativity", attr_total_list = n, macro_micro = md_and_opiod, constraint_list_list = cll)
  #cll <- all_geogs_add_constraint(attr_name = "geog_mobility", attr_total_list = ge, macro_micro = md_and_opiod, constraint_list_list = cll)
  #cll <- all_geogs_add_constraint(attr_name = "ind_income", attr_total_list = i, macro_micro = md_and_opiod, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "drug_overdose", attr_total_list = cons[[7]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "Insurance", attr_total_list = cons[[8]], macro_micro = synthetic_final, constraint_list_list = cll)
  cll <- all_geogs_add_constraint(attr_name = "Veteran", attr_total_list = cons[[9]], macro_micro = synthetic_final, constraint_list_list = cll)
  
  opt_simulated <- all_geog_optimize_microdata(synthetic_final, seed = 12L,
                                        constraint_list_list = cll, p_accept = 0.4, max_iter = 1000L,
                                        verbose = TRUE)
  
  simulated_fit <- get_best_fit(opt_simulated, geography = county)
  simulated_tae <- get_final_tae(opt_simulated, geography = county)
  print(simulated_tae)
  
  write_name = paste0("simulated_",county_fips,"_",county,".csv")
  write.csv(simulated_fit,file.path(data_path,write_name), row.names = FALSE)
  gc()
  
  
  
}


