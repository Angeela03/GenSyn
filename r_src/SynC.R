library(renv)
renv::init()

library(mvtnorm)
library(fitdistrplus)
library(dplyr)
library(vcd)
# install.packages("cli")
# devtools::install_github("rstudio/tensorflow")
# devtools::install_github("rstudio/keras")
# library(tensorflow)
# install_tensorflow(method = "auto")
# # install.packages("keras")
# .rs.restartR()
library(keras)

# install.packages("keras")
reticulate::conda_install(packages = "keras")
# keras::install_keras()
# #devtools::install_github("rstudio/tensorflow")
# install.packages("tensorflow")
# use_condaenv("crime-env")
# Sys.setenv(KERAS_BACKEND="theano") 
# install_keras()
# library(tensorflow)
# tensorflow::install_tensorflow()
# tensorflow::tf_config()



# setwd("C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/r_codes/conditional_rpobabilties_generic")
# data_dir = "C:/Users/achar/OneDrive/Documents/Project_opiod/project_final_code_synthetic/data/"

setwd(".")
data_path <- file.path(".","data_acs")

d_req <- read.csv(file.path(data_dir, "All_counties_macro_norm.csv"))
marginal_data <- read.csv(file.path(data_dir, "All_counties_macro.csv"))

names(d_req) <- sub("^X", "", names(d_req))
names(marginal_data) <- sub("^X", "", names(marginal_data))

read_od_data = read.csv(file.path(data_dir, "Overdose_by_age_all_counties.csv"))
read_insurance_data = read.csv(file.path(data_dir, "Insurance_by_gender_all_counties.csv"))
read_veteran_data = read.csv(file.path(data_dir, "Veteran_by_gender_all_counties.csv"))

merge_od_insurance <- merge(read_od_data, read_insurance_data, by = 'County.Code', all.x= TRUE)
read_od_data <- dplyr::select(merge_od_insurance,County.Code, County.x, State.x,  Year.x,     level.x, pct.x, State.y)
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

sample_data <- function(n,d){
n<- marginal_total
m = ncol(d)  

x = mvrnorm(n, mu = rep(0, m), Sigma = cor(d, use = 'complete.obs'), empirical = TRUE)
u = pnorm(x)
data = matrix(0, nrow = n, ncol = m)
d[is.na(d)] = 0.00001
for(i in 1:m){
range_ = range(d[,i], na.rm=TRUE)
if(range_[2] - range_[1] < 1){
  # dist = 'beta'
  estimates = fitdist(pull(d, i), 'beta', method = 'mme')$estimate
  # data[,i] = scaleRange(qbeta(u[,i], estimates[1], estimates[2]))
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
return(data)
}

powerTwo = function(num){
  power = 1
  while(2^power <= num){
    power = power + 1
  }
  return(2^(power-1))
}

model_to_pop = function(core, target, input){
  ind = sample(1:nrow(core), round(0.3*nrow(core)))
  print(ind)
  train_x = core[-ind, ]
  train_y = target[-ind, ]
  val_x = core[ind, ]
  val_y = target[ind, ]
  test_y = input
  model <- keras_model_sequential()
  if(all(train_y <= 1)){
    model %>%
      layer_dense(units = max(powerTwo(ncol(train_x))/2, 4), 
                  input_shape = ncol(train_x), activation = 'relu') %>%
      layer_dense(units = max(powerTwo(ncol(train_x))/4, 2), activation = 'relu') %>%
      layer_dense(units = ncol(train_y), activation = 'softmax') %>%
      compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = list('mae')
      )
  } else{
    model %>%
      layer_dense(units = max(powerTwo(ncol(train_x))/2, 4), 
                  input_shape = ncol(train_x), activation = 'relu') %>%
      layer_dense(units = max(powerTwo(ncol(train_x))/4, 2), activation = 'relu') %>%
      layer_dense(units = ind) %>%
      compile(
        optimizer = 'adam',
        loss = "mse",
        metrics = list("mae")
      )
  }
  
  model %>% fit(
    train_x,
    train_y,
    epochs = 5,
    batch_size = 512,
    validation_data = list(val_x, val_y)
  )
  
  test_y = model %>% predict(input, batch_size = 32)
  return(test_y)
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
od_grp <- c("Overdose", "No_overdose")
is_grp <-c("Insured","Uninsured")
vt_grp <-c("Veteran", "Not_veteran")

batches = list(gender_grp, mar_grp, edu_grp, emp_grp, pov_grp, od_grp, is_grp, vt_grp) 
print(batches)


d<- d_req[, !names(d_req) %in% c("FIPS")]
d <- d[,2:ncol(d)]

core = subset(d, select=age_grp)

for(j in 1:nrow(top_50)) {
  row <- top_50[50,]
  county_fips <- trimws(as.character(row["County.Code"][[1]][[1]])) # trimws removes leading and trailing zeros
  marginal_ <- marginal_data[marginal_data$FIPS ==county_fips,]
  print(marginal_)
  marginal_constraints <- marginal_[,"Male"] + marginal_[,"Female"]
  print(marginal_constraints)
  marginal_total = marginal_constraints
  
  y_dash = sample_data(marginal_total, core)
  colnames(y_dash) = age_grp
  print(y_dash)
  
  data =  data.frame(core)
  for (b in 1:length(batches)){
    core_b = append(age_grp, batches[[1]])
    d_b = d[,core_b]
    print(d_b)
    k_b = sample_data(marginal_total, d_b)
    colnames(k_b) = core_b
    print(k_b)
    
    core_kb = k_b[,age_grp]
    target_kb = k_b[,batches[[1]]]
    print(core_kb)
    print(target_kb)
    input = core
    core =core_kb
    target= target_kb
    ind = sample(1:nrow(core), round(0.3*nrow(core)))
    print(ind)
    train_x = core[-ind, ]
    train_y = target[-ind, ]
    val_x = core[ind, ]
    val_y = target[ind, ]
    test_x = input
    model <- keras_model_sequential()
    if(all(train_y <= 1)){
      model %>%
        layer_dense(units = max(powerTwo(ncol(train_x))/2, 4), 
                    input_shape = ncol(train_x), activation = 'relu') %>%
        layer_dense(units = max(powerTwo(ncol(train_x))/4, 2), activation = 'relu') %>%
        layer_dense(units = ncol(train_y), activation = 'softmax') %>%
        compile(
          optimizer = 'adam',
          loss = 'categorical_crossentropy',
          metrics = list('mae')
        )
    } else{
      model %>%
        layer_dense(units = max(powerTwo(ncol(train_x))/2, 4), 
                    input_shape = ncol(train_x), activation = 'relu') %>%
        layer_dense(units = max(powerTwo(ncol(train_x))/4, 2), activation = 'relu') %>%
        layer_dense(units = ind) %>%
        compile(
          optimizer = 'adam',
          loss = "mse",
          metrics = list("mae")
        )
    }
    
    model %>% fit(
      train_x,
      train_y,
      epochs = 5,
      batch_size = 512,
      validation_data = list(val_x, val_y)
    )
    
    test_y = model %>% predict(input, batch_size = 32)

    predicted_b = model_to_pop(core_kb, target_kb, y_dash)
    print(predicted_b)
    colnames(predicted_b) = batches[[1]]
    data =cbind(data, predicted_b)
  }
    age_data <- data[,age_grp]
    gender_data <- data[,gender_grp]
    mar_data <- data[,mar_grp]
    edu_data <- data[,edu_grp]
    emp_data <- data[,emp_grp]
    pov_data <- data[,pov_grp]
    od_data <- data[,od_grp]
    is_data <- data[,is_grp]
    vt_data <- data[,vt_grp]
    
    age_marginal <- marginal_[,age_grp]
    gender_marginal <- marginal_[,gender_grp]
    mar_marginal <- marginal_[,mar_grp]
    edu_marginal <- marginal_[,edu_grp]
    emp_marginal <- marginal_[,emp_grp]
    pov_marginal <- marginal_[,pov_grp]
    od_marginal <- marginal_[,od_grp]
    is_marginal <- marginal_[,is_grp]
    vt_marginal <- marginal_[,vt_grp]
    
    age_marginals = match_marginal(age_data, age_marginal, age_grp)
    gender_marginals = match_marginal(gender_data, gender_marginal, gender_grp)
    mar_marginals = match_marginal(mar_data, mar_marginal, mar_grp)
    edu_marginals = match_marginal(edu_data, edu_marginal, edu_grp)
    emp_marginals = match_marginal(emp_data, emp_marginal, emp_grp)
    pov_marginals = match_marginal(pov_data, pov_marginal, pov_grp)
    od_marginals = match_marginal(od_data, od_marginal, od_grp)
    is_marginals = match_marginal(is_data, is_marginal,is_grp)
    vt_marginals = match_marginal(vt_data, vt_marginal, vt_grp)
    
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
    group_by_df[, "p"] <- lapply(group_by_df[, "COUNT"], function(x) x/sum(x))
    group_by_df <- group_by_df[, -which(names(group_by_df) == "COUNT")] 
    
    df_final <- group_by_df[,c("age","gender","marital_status","edu_attain","emp_status","pov_status","drug_overdose","Insurance","Veteran","p")]
    write_name <- paste0("sync_",county_fips,".csv", sep="")
    write.csv(df_final, file.path(data_dir, write_name))
  }




