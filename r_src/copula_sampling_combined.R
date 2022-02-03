# Runs the Gaussian Copula sampling algorithm for the Combined dataset and returns the joint probability distribution p2

library(mvtnorm)
library(fitdistrplus)
library(dplyr)
library(readr)
library(renv)
renv::init()

setwd(".")
data_dir <- file.path(".","data")

# Get the macro data for multiple geographic locations and its normalized version
d_req <- read_csv(file.path(data_dir, "All_counties_macro_norm.csv"))
marginal_data <- read_csv(file.path(data_dir, "All_counties_macro.csv"))

# Get OD, Insurance, and Veteran data
read_od_data = read.csv(file.path(data_dir, "Overdose_by_age_all_counties.csv"))
read_insurance_data = read.csv(file.path(data_dir, "Insurance_by_gender_all_counties.csv"))
read_veteran_data = read.csv(file.path(data_dir, "Veteran_by_gender_all_counties.csv"))

top_50 <- read.csv(file.path(data_path, "top_50_od.csv"))

# Function for matching marginals (Algorithm 3)
match_marginal = function(output, marginals, varnames){
  matched = matrix(0, nrow = nrow(output), ncol = ncol(output))
  colnames(matched) = varnames
  for(i in 1:nrow(output)){
    prob = output[i,]/sum(output[i,])
    matched[i, which(output[i,] == max(output[i,]))] = 1
  }
  empirical_marginals = apply(matched, 2, sum)
  while(!all(empirical_marginals == marginals)){
    diff = empirical_marginals - marginals
    over_indexed = which(diff > 0)
    under_indexed = which(diff < 0)
    for(i in over_indexed){
      ind = which(matched[,i] == 1)
      ind = ind[order(output[ind, i], decreasing = FALSE)]
      matched[ind[1:diff[1,i]], ] = 0
    }
    
    for(j in 1:nrow(matched)){
      if(all(matched[j,] == 0)){
        prob = output[j, under_indexed]/sum(output[j, under_indexed])
        new_max_ind = which(output[j, under_indexed] == max(output[j, under_indexed]))
        matched[j, under_indexed[new_max_ind]] = 1
      }
    }
    empirical_marginals = apply(matched, 2, sum)
  }
  out = rep('', nrow(matched))
  for(i in 1:nrow(matched)){
    if(sum(matched[i, ]) == 1)
      out[i] = colnames(matched)[which(matched[i,] == 1)]
    else
      out[i] = ""
  }
  return(out)
}


d_req <- na.omit(d_req)
marginal_data <- na.omit(marginal_data)

# Define categories for the variables
age_grp <- c("under15", "15_17", "18_24", "25_29", "30_34", "35_39", "40_44", "45_49","50_54","55_59","60_64","65_69", 
             "70_74","75_79","80_84","85up")
gender_grp <- c("Male", "Female")
mar_grp <- c("never_mar", "married","mar_apart","widowed","divorced")
edu_grp <- c("lt_hs","some_hs","hs_grad","some_col","assoc_dec","ba_deg","grad_deg")
emp_grp <-c("not_in_labor_force", "employed", "unemployed")
pov_grp <- c("below_pov_level", "at_above_pov_level")
od_grp <- c("Overdose", "No_overdose")
is_grp <-c("Insured","Uninsured")
vt_grp <-c("Veteran", "Not_veteran")

d<- d_req[, !names(d_req) %in% c("FIPS")]
d <- d %>% select(2:ncol(d))

# Copula sampling for the 50 counties
for(j in 1:50) {
  row <- top_50[j,]
  county_fips <- trimws(as.character(row["County.Code"][[1]][[1]])) # trimws removes leading and trailing zeros
  
  # Get the marginal constraints for a specific county
  marginal_ <- marginal_data[marginal_data$FIPS ==county_fips,]
  marginal_constraints <- marginal_[,"Male"] + marginal_[,"Female"]
  marginal_total = marginal_constraints[1,"Male"]

  # Implement Algorithm 2 - Gaussian Copula Sampling
  n<- marginal_total
  m = ncol(d)
  
  it = 3
  df<-data.frame()
  for (k in 1:it){
    x = mvrnorm(n, mu = rep(0, m), Sigma = cor(d, use = 'complete.obs'), empirical = TRUE)
    u = pnorm(x)
    data = matrix(0, nrow = n, ncol = m)
    d[is.na(d)] = 0.00001
    for(i in 1:m){
      range_ = range(d[,i], na.rm=TRUE)
      if(range_[2] - range_[1] < 1){
        estimates = fitdist(pull(d, i), 'beta', method = 'mme')$estimate
        data[,i] = qbeta(u[,i], estimates[1], estimates[2])
      } else if (range_[2]  - range_[1] == 1){
        d[,i] = ifelse(d[,i] == 0, 0.00001, d[,i])
        d[,i] = ifelse(d[,i] == 1, 0.99999, d[,i])
        estimates = fitdist(pull(d, i), 'beta', method = 'mme', na.rm  =   T)$estimate
        data[,i] = qbeta(u[,i], estimates[1], estimates[2])
      }
    }
    
    # Get the corresponsing data for each variable
    colnames(data) <- colnames(d)
    age_data <- data[,age_grp]
    gender_data <- data[,gender_grp]
    mar_data <- data[,mar_grp]
    edu_data <- data[,edu_grp]
    emp_data <- data[,emp_grp]
    pov_data <- data[,pov_grp]
    od_data <- data[,od_grp]
    is_data <- data[,is_grp]
    vt_data <- data[,vt_grp]
    
    df_all <- cbind(age_data,gender_data,mar_data,edu_data,emp_data,pov_data, od_data,is_data, vt_data)
    
    age_marginal <- marginal_[,age_grp]
    gender_marginal <- marginal_[,gender_grp]
    mar_marginal <- marginal_[,mar_grp]
    edu_marginal <- marginal_[,edu_grp]
    emp_marginal <- marginal_[,emp_grp]
    pov_marginal <- marginal_[,pov_grp]
    od_marginal <- marginal_[,od_grp]
    is_marginal <- marginal_[,is_grp]
    vt_marginal <- marginal_[,vt_grp]
    
    age_marginals = match_marginal(age_data, age_marginal, colnames(age_data))
    gender_marginals = match_marginal(gender_data, gender_marginal, colnames(gender_data))
    mar_marginals = match_marginal(mar_data, mar_marginal, colnames(mar_data))
    edu_marginals = match_marginal(edu_data, edu_marginal, colnames(edu_data))
    emp_marginals = match_marginal(emp_data, emp_marginal, colnames(emp_data))
    pov_marginals = match_marginal(pov_data, pov_marginal, colnames(pov_data))
    od_marginals = match_marginal(od_data, od_marginal, colnames(od_data))
    is_marginals = match_marginal(is_data, is_marginal, colnames(is_data))
    vt_marginals = match_marginal(vt_data, vt_marginal, colnames(vt_data))
    
    # Combine results from all variables together
    final_df <- data.frame (age  = c(age_marginals),
                      gender = c(gender_marginals),
                      marital_status = c(mar_marginals),
                      edu_attain = c(edu_marginals),
                      emp_status = c(emp_marginals),
                      pov_status = c(pov_marginals),
                      drug_overdose = c(od_marginals),
                      Insurance = c(is_marginals),
                      Veteran = c(vt_marginals))
    
    group_by_df <-final_df %>% group_by_all() %>% summarise(COUNT = n())
    group_by_df[, "w"] <- lapply(group_by_df[, "COUNT"], function(x) x/sum(x))
    group_by_df <- group_by_df[, -which(names(group_by_df) == "COUNT")] 
    if (nrow(df)==0){df<-group_by_df} else
    {df<-merge(x=df,y=group_by_df,by=c("age","gender","marital_status","edu_attain","emp_status","pov_status","drug_overdose","Insurance","Veteran"),all=TRUE)}
    gc()
  }

  df<- df %>% replace(is.na(.), 0)

  # Take the average of the probability distributions returned by the above steps
  last_k_col = df[,(ncol(df)-it+1):ncol(df)]
  df[,"p"] = rowMeans(last_k_col)
  df_final <- df[,c("age","gender","marital_status","edu_attain","emp_status","pov_status","drug_overdose","Insurance","Veteran","p")]
  
  # Normalize the p values
  df_final[,"p"] <- df_final[,"p"]/sum(df_final[,"p"])
  
  # Write the file
  write_name <- paste0("prior_copula_",county_fips,".csv", sep="")
  write.csv(df_final, file.path(data_dir, write_name))
}
