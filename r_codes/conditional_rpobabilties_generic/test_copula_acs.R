#renv::install("mvtnorm")
# setwd("C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/r_codes/conditional_rpobabilties_generic")
# data_dir = "C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/data_acs/"
library(renv)
renv::init()
library(mvtnorm)
#renv::install("fitdistrplus")
library(fitdistrplus)
library(dplyr)
setwd(".")
data_dir <- file.path(".","data_acs")

# renv::install("hms")
# renv::install("readr")

#renv::install("vcd")
# library(vcd)
# renv::install("corrplot")
# library(corrplot)
# .rs.restartR()


# d_req<- read_csv(paste(file_dir, "merged_county_data.csv", sep = ''))
d_req <- read.csv(file.path(data_dir, "All_counties_macro_norm_acs.csv"))
marginal_data <- read.csv(file.path(data_dir, "All_counties_macro_acs.csv"))

names(d_req) <- sub("^X", "", names(d_req))
names(marginal_data) <- sub("^X", "", names(marginal_data))

top_50 = read.csv(file.path(data_dir, "top_50_acs.csv"))

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

age_grp <- c("under15", "15_17", "18_24", "25_29", "30_34", "35_39", "40_44", "45_49","50_54","55_59","60_64","65_69", 
             "70_74","75_79","80_84","85up")
gender_grp <- c("Male", "Female")
mar_grp <- c("never_mar", "married","mar_apart","widowed","divorced")
edu_grp <- c("lt_hs","some_hs","hs_grad","some_col","assoc_dec","ba_deg","grad_deg")
emp_grp <-c("not_in_labor_force", "employed", "unemployed")
pov_grp <- c("below_pov_level", "at_above_pov_level")
geog_grp <- c("same.house", "different.house.in.us", "moved.from.abroad")
nat_grp <-c("native", "foreigner")

d<- d_req[, !names(d_req) %in% c("FIPS")]
names(d) <- trimws(names(d))
d <- d[,2:ncol(d)]
for(j in 5:10) {
  row <- top_50[j,]
  print(row)
  county_fips <- trimws(as.character(row["County.Code"][[1]][[1]])) # trimws removes leading and trailing zeros
  marginal_ <- marginal_data[marginal_data$FIPS ==county_fips,]
  print(marginal_)
  marginal_constraints <- marginal_[,"Male"] + marginal_[,"Female"]
  # marginal_total = marginal_constraints[1,"Male"]
  # print(marginal_total)
  n<- marginal_constraints
  m = ncol(d)
  print(n)

  
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
        #dist = 'beta'
        estimates = fitdist(pull(d, i), 'beta', method = 'mme')$estimate
        #data[,i] = scaleRange(qbeta(u[,i], estimates[1], estimates[2]))
        data[,i] = qbeta(u[,i], estimates[1], estimates[2])
      } else if (range_[2]  - range_[1] == 1){
        d[,i] = ifelse(d[,i] == 0, 0.00001, d[,i])
        d[,i] = ifelse(d[,i] == 1, 0.99999, d[,i])
        estimates = fitdist(pull(d, i), 'beta', method = 'mme', na.rm  =   T)$estimate
        data[,i] = qbeta(u[,i], estimates[1], estimates[2])
      } else{
        dist = 'lnorm'
        estimates = fitdist(d[,i], 'lnorm', method = 'mme')$estimate
        data[,i] = qlnorm(u[,i], estimates[1], estimates[2])}
    }
    # print(data)
    # cor(data[,c("below_pov_level", "unemployed", "lt_hs")])
    # print(mean(data[,1]))
    print('Data generated for iteration')
    print(k)
    # match_marginal = function(output, marginals, varnames){
    colnames(data) <- colnames(d)
    age_data <- data[,age_grp]
    gender_data <- data[,gender_grp]
    mar_data <- data[,mar_grp]
    edu_data <- data[,edu_grp]
    emp_data <- data[,emp_grp]
    pov_data <- data[,pov_grp]
    geog_data <- data[,geog_grp]
    nat_data <- data[,nat_grp]
    
    df_all <- cbind(age_data,gender_data,mar_data,edu_data,emp_data,pov_data, geog_data, nat_data)
    # print(df_all)
    # write.csv(df_all, "df_all_copla.csv")
    
    age_marginal <- marginal_[,age_grp]
    gender_marginal <- marginal_[,gender_grp]
    mar_marginal <- marginal_[,mar_grp]
    edu_marginal <- marginal_[,edu_grp]
    emp_marginal <- marginal_[,emp_grp]
    pov_marginal <- marginal_[,pov_grp]
    geog_marginal <- marginal_[,geog_grp]
    nat_marginal <- marginal_[,nat_grp]

    age_marginals = match_marginal(age_data, age_marginal, colnames(age_data))
    gender_marginals = match_marginal(gender_data, gender_marginal, colnames(gender_data))
    mar_marginals = match_marginal(mar_data, mar_marginal, colnames(mar_data))
    edu_marginals = match_marginal(edu_data, edu_marginal, colnames(edu_data))
    emp_marginals = match_marginal(emp_data, emp_marginal, colnames(emp_data))
    pov_marginals = match_marginal(pov_data, pov_marginal, colnames(pov_data))
    geog_marginals = match_marginal(geog_data, geog_marginal, colnames(geog_data))
    nat_marginals = match_marginal(nat_data, nat_marginal, colnames(nat_data))
    
    final_df <- data.frame (age  = c(age_marginals),
                            gender = c(gender_marginals),
                            marital_status = c(mar_marginals),
                            edu_attain = c(edu_marginals),
                            emp_status = c(emp_marginals),
                            pov_status = c(pov_marginals),
                            geog_mobility = c(geog_marginals),
                            nativity = c(nat_marginals))
    
    group_by_df <-final_df %>% group_by_all() %>% summarise(COUNT = n())
    group_by_df[, "w"] <- lapply(group_by_df[, "COUNT"], function(x) x/sum(x))
    group_by_df <- group_by_df[, -which(names(group_by_df) == "COUNT")] 
    if (nrow(df)==0){df<-group_by_df} else
    {df<-merge(x=df,y=group_by_df,by=c("age","gender","marital_status","edu_attain","emp_status","pov_status","geog_mobility","nativity"),all=TRUE)}
    gc()
  }
  print("data for all iterations generated")
  df<- df %>% replace(is.na(.), 0)
  
  last_k_col = df[,(ncol(df)-it+1):ncol(df)]
  df[,"p"] = rowMeans(last_k_col)
  df_final <- df[,c("age","gender","marital_status","edu_attain","emp_status","pov_status","geog_mobility","nativity","p")]
  # df_final[, "p"] <- lapply(df_final[, "p"], function(x) x/sum(x))
  df_final[,"p"] <- df_final[,"p"]/sum(df_final[,"p"])
  df_final$geog_mobility <- as.character(df_final$geog_mobility)
  print(sapply(df_final, class))
  df_final[df_final["geog_mobility"] == "different.house.in.us", "geog_mobility"] <- "different house in us"
  df_final[df_final["geog_mobility"] == "same.house", "geog_mobility"] <- "same house"
  df_final[df_final["geog_mobility"] == "moved.from.abroad", "geog_mobility"] <- "moved from abroad"
  write_name <- paste0("prior_copula_acs_",county_fips,".csv", sep="")
  write.csv(df_final, file.path(data_dir, write_name))
  
}


