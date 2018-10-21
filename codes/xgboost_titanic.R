library(xgboost)
library(Ckmeans.1d.dp)
getwd()

# 读取原始数据文件

# 自变量及其列名
train_xgb <- xgb.DMatrix("./train_xgb")
train_xgb_name <- readRDS("./train_xgb_name.R")
colnames(train_xgb) <- make.names(train_xgb_name)
head(train_xgb_name)
print(train_xgb,verbose = FALSE)

# 因变量
train_xgb_label <- getinfo(train_xgb, 'label')

# nrounds
set.seed(100)
xgb_1_nrounds <- xgb.cv(
  data = train_xgb, 
  max.depth = 5, 
  eta = 0.1, nthread = 2, 
  nrounds = 1000, 
  objective = "binary:logistic", 
  max_delta_step=4, 
  subsample = 0.8, 
  verbose = 0 ,
  nfold = 5,  
  metrics = list("rmse","auc","error") ,
  gamma = 0,
  colsample_bytree=0.8, 
  scale_pos_weight = 1, 
  min_child_weight = 1)
eva_log <- xgb_1_nrounds$evaluation_log
eva_log[eva_log$test_auc_mean==max(eva_log$test_auc_mean),]

# 查看各变量重要性
set.seed(100)
xgb_1_nrounds <- xgb.train(
  data = train_xgb, 
  max.depth = 5, 
  eta = 0.1, nthread = 2, 
  nrounds = 1000, 
  objective = "binary:logistic", 
  max_delta_step=4, 
  subsample = 0.8, 
  verbose = 0 ,
  nfold = 5,  
  metrics = list("rmse","auc","error") ,
  gamma = 0,
  colsample_bytree=0.8, 
  scale_pos_weight = 1, 
  min_child_weight = 1)
importance_nrounds <- xgb.importance(train_xgb_name, model = xgb_1_nrounds)
gg <- xgb.ggplot.importance(importance_nrounds, measure = "Frequency", rel_to_first = TRUE)
gg + ggplot2::ylab("Frequency")

# 编写查找最好的max depth and min child weight 组合函数
xgb_depth_child_weight_perfect <- function(
  depth, 
  child_weight, 
  input_data, 
  input_eta, 
  input_nrounds, 
  input_objective, 
  input_max_delta_step, 
  input_subsample, 
  input_verbose, 
  input_nfold, 
  input_metrics,
  input_gamma, 
  input_colsample_bytree, 
  input_scale_pos_weight
){
  last_auc <- 0
  for (i in 1:length(depth)) {
    for (j in 1:length(child_weight)) {
      
      set.seed(100)
      depth_child_model <- xgb.cv(
        max_depth = depth[i], 
        eta = input_eta , 
        metrics =input_metrics, 
        min_child_weight = child_weight[j], 
        data = input_data,  
        nrounds =input_nrounds,
        max_delta_step = input_max_delta_step,
        subsample = input_subsample, 
        verbose = input_verbose, 
        nfold = input_nfold, 
        gamma = input_gamma,
        colsample_bytree = input_colsample_bytree,
        scale_pos_weight = input_scale_pos_weight,
        objective = input_objective,
      )
      depth_child_model_eva_log <- depth_child_model$evaluation_log
      now_auc <- max(depth_child_model_eva_log$test_auc_mean)
      tmp <- now_auc
      if(now_auc > last_auc){
        last_auc <- tmp
        perfect_depth <- depth[i]
        perfect_child_weight <- child_weight[j]
      }
    }
  }
  print("perfect_depth")
  print(perfect_depth)
  print("perfect_child_weight")
  print( perfect_child_weight)
}
# 查找最好的max depth and min child weight 组合
xgb_depth_child_weight_perfect(
  input_data = train_xgb ,
  depth = 3:6,
  input_eta = 0.1,
  input_nrounds = 77,
  input_objective = "binary:logistic",
  input_max_delta_step=4,
  input_subsample = 0.8,
  input_verbose = 0,
  input_nfold = 5,
  input_metrics = list("rmse","auc","error"),
  input_gamma = 0,
  input_colsample_bytree=0.8,
  input_scale_pos_weight = 1,
  child_weight = seq(0,1, by=0.2)
)
# 查看该组合下max test auc mean
set.seed(100)
xg_2_depth_chlid_weight <- xgb.cv(
  data = train_xgb, 
  max_depth= 6, 
  eta = 0.1, 
  nrounds = 77, 
  bjective = "binary:logistic", 
  max_delta_step = 4, 
  subsample = 0.8,
  verbose = 0, 
  gamma = 0, 
  colsample_bytree = 0.8, 
  scale_pos_weight = 1, 
  min_child_weight = 0,
  nfold = 5, 
  metrics = list("rmse","auc","error"),
)
eva_log <- xg_2_depth_chlid_weight$evaluation_log
eva_log[eva_log$test_auc_mean == max(eva_log$test_auc_mean),]
# 查看各变量重要性
set.seed(100)
xg_2_depth_chlid_weight <- xgb.train(
  data = train_xgb, 
  max_depth = 6, 
  eta = 0.1, 
  nrounds = 77, 
  bjective = "binary:logistic", 
  max_delta_step = 4, 
  subsample = 0.8,
  verbose = 0, 
  gamma = 0, 
  colsample_bytree = 0.8, 
  scale_pos_weight = 1, 
  min_child_weight = 0,
  nfold = 5, 
  metrics = list("rmse","auc","error")
)
importance_depth_child_weight <- xgb.importance(train_xgb_name, model = xg_2_depth_chlid_weight)
gg <- xgb.ggplot.importance(importance_depth_child_weight[1:15,], measure = "Frequency", rel_to_first = TRUE)
gg + ggplot2::ylab("Frequency")

# 编写查找最好的gamma函数
xgb_gamma_perfect <- function(
  depth, 
  child_weight, 
  input_data, 
  input_eta, 
  input_nrounds, 
  input_objective, 
  input_max_delta_step, 
  input_subsample, 
  input_verbose, 
  input_nfold, 
  input_metrics, 
  input_gamma, 
  input_colsample_bytree, 
  input_scale_pos_weight){
  last_auc <- 0
  for (i in 1:length(input_gamma)) {
    set.seed(100)
    depth_child_model <- xgb.cv(
      gamma = input_gamma[i], 
      eta = input_eta ,
      metrics = input_metrics, 
      min_child_weight = child_weight, 
      data = input_data,
      nrounds = input_nrounds,
      max_delta_step = input_max_delta_step,
      subsample = input_subsample, 
      verbose = input_verbose, 
      nfold = input_nfold,
      max_depth = depth,
      colsample_bytree = input_colsample_bytree,
      scale_pos_weight = input_scale_pos_weight,
      objective = input_objective
    )
    depth_child_model_eva_log <- depth_child_model$evaluation_log
    now_auc <- max(depth_child_model_eva_log$test_auc_mean)
    tmp <- now_auc
    if(now_auc>last_auc){
      last_auc <- tmp
      perfect_gamma <- input_gamma[i]
    }
  }
  print("perfect_gamma")
  print(perfect_gamma)
}
xgb_gamma_perfect(
  input_data = train_xgb, 
  depth = 6, 
  input_eta = 0.1, 
  input_nrounds = 77, 
  input_objective = "binary:logistic",  
  input_max_delta_step=4,  
  input_subsample = 0.8, 
  input_verbose = 0 ,
  input_nfold = 5,  
  input_metrics = list("rmse","auc","error") ,
  input_gamma = seq(0, 0.5, by = 0.1),
  input_colsample_bytree=0.8, 
  input_scale_pos_weight = 1, 
  child_weight = 0)
# 编写查找最大subsample、coltree_perfect最大组合的函数
xgb_subsample_coltree_perfect <- function(
  depth, 
  child_weight, 
  input_data, 
  input_eta, 
  input_nrounds, 
  input_objective, 
  input_max_delta_step, 
  para_subsample, 
  input_verbose, 
  input_nfold, 
  input_metrics, 
  input_gamma,
  para_colsample_bytree, 
  input_scale_pos_weight
){
  last_auc <- 0
  for (i in 1:length(para_subsample)) {
    for (j in 1:length(para_colsample_bytree)) {
      set.seed(100)
      depth_child_model <- xgb.cv(
        max_depth = depth, 
        eta = input_eta ,
        metrics = input_metrics, 
        min_child_weight = child_weight, 
        data = input_data,
        nrounds = input_nrounds,
        max_delta_step=input_max_delta_step,
        subsample=para_subsample[i], 
        verbose=input_verbose,
        nfold=input_nfold, 
        gamma=input_gamma,
        colsample_bytree=para_colsample_bytree[j],
        scale_pos_weight=input_scale_pos_weight,
        objective=input_objective
      )
      depth_child_model_eva_log <-
        depth_child_model$evaluation_log
      now_auc <- max(depth_child_model_eva_log$test_auc_mean)
      tmp <- now_auc
      if(now_auc>last_auc){
        last_auc <- tmp
        perfect_subsample <- para_subsample[i]
        perfect_colsample_bytree <- para_colsample_bytree[j]
      }
    }
  }
  print("perfect_subsample")
  print(perfect_subsample)
  print("perfect_colsample_bytree")
  print(perfect_colsample_bytree)
}
# 查找最大subsample、coltree_perfect最大组合
xgb_subsample_coltree_perfect(
  input_data = train_xgb, 
  depth = 6, 
  input_eta = 0.1, 
  input_nrounds = 77, 
  input_objective = "binary:logistic",  
  input_max_delta_step=4,  
  para_subsample = seq(0.6, 1, by=0.1), 
  input_verbose = 0 ,
  input_nfold = 5,  
  input_metrics = list("rmse","auc","error") ,
  input_gamma = 0,
  para_colsample_bytree = seq(0.6, 1, by=0.1),
  input_scale_pos_weight = 1, 
  child_weight = 0
)
# 查看最大test auc mean
set.seed(100)
xg_4_subsample_coltree <- xgb.cv(
  data = train_xgb, 
  max_depth = 6, 
  eta = 0.1, 
  nrounds = 77, 
  bjective = "binary:logistic", 
  max_delta_step = 4, 
  subsample = 0.9,
  verbose = 0, 
  gamma = 0, 
  colsample_bytree = 1, 
  scale_pos_weight = 1, 
  min_child_weight = 0,
  nfold = 5, 
  metrics = list("rmse","auc","error")
)
eva_log <- xg_4_subsample_coltree$evaluation_log
eva_log[eva_log$test_auc_mean == max(eva_log$test_auc_mean),]

# 查看各变量重要性
importance_subsample_coltree <- xgb.importance(train_xgb_name, model = xg_4_subsample_coltree)
gg <- xgb.ggplot.importance(importance_subsample_coltree[1:15,], measure = "Frequency", rel_to_first = TRUE)
gg + ggplot2::ylab("Frequency")
# 编写查找最大正则化参数函数
xgb_reg_alpha_perfect <- function(
  depth, 
  child_weight, 
  input_data, 
  input_eta, 
  input_nrounds, 
  input_objective,
  input_max_delta_step, 
  input_subsample,
  input_verbose, 
  input_nfold, 
  input_metrics, 
  input_gamma, 
  input_colsample_bytree,
  input_scale_pos_weight,
  input_reg_alpha
){
  last_auc <- 0
  for (i in 1:length(input_reg_alpha)) {
    set.seed(100)
    depth_child_model <- xgb.cv(
      max_depth = depth, 
      reg_alpha = input_reg_alpha[i], 
      eta = input_eta ,
      metrics = input_metrics,
      min_child_weight = child_weight, 
      data = input_data,  
      nrounds = input_nrounds,
      max_delta_step = input_max_delta_step,
      subsample = input_subsample, 
      verbose = input_verbose, 
      nfold = input_nfold, 
      max_depth = depth, 
      colsample_bytree = input_colsample_bytree,
      scale_pos_weight = input_scale_pos_weight,
      objective = input_objective)
    depth_child_model_eva_log <- depth_child_model$evaluation_log
    now_auc <- max(depth_child_model_eva_log$test_auc_mean)
    tmp <- now_auc
    if(now_auc>last_auc){
      last_auc <- tmp
      perfect_reg_alpha <- input_reg_alpha[i]
    }
  }
  print("perfect_reg_alpha")
  print(perfect_reg_alpha)
}
# 查找最大正则化参数
xgb_reg_alpha_perfect(
  input_data = train_xgb, 
  depth = 6, 
  input_eta = 0.1, 
  input_nrounds = 77, 
  input_objective = "binary:logistic", 
  input_max_delta_step = 4,  
  input_subsample = 0.9, 
  input_verbose = 0 ,
  input_nfold = 5,  
  input_metrics = list("rmse","auc","error") ,
  input_gamma = 0,
  input_colsample_bytree = 1,
  input_scale_pos_weight = 1, 
  child_weight = 0, 
  input_reg_alpha = c(0.00001, 0.01, 0.1, 1, 100))
set.seed(100)
xg_5_reg_alpha <- xgb.cv(
  data = train_xgb, 
  max_depth = 6, 
  eta = 0.1, 
  nrounds = 77, 
  bjective = "binary:logistic", 
  max_delta_step = 4, 
  subsample = 0.9,
  verbose = 0, 
  gamma = 0, 
  colsample_bytree = 1, 
  scale_pos_weight = 1, 
  min_child_weight = 0,
  nfold = 5, 
  metrics = list("rmse","auc","error"),
  reg_alpha = 0.00001
)
# 查看此时test auc mean
eva_log <- xg_5_reg_alpha$evaluation_log
eva_log[eva_log$test_auc_mean == max(eva_log$test_auc_mean),]
# 查看各变量重要性
set.seed(100)
xg_5_reg_alpha <- xgb.train(
  data = train_xgb, 
  max_depth = 6, 
  eta = 0.1, 
  nrounds = 69, 
  bjective = "binary:logistic", 
  max_delta_step = 4, 
  subsample = 0.9,
  verbose = 0, 
  gamma = 0, 
  colsample_bytree = 1, 
  scale_pos_weight = 1, 
  min_child_weight = 0,
  nfold = 5, 
  metrics = list("rmse","auc","error"),
  reg_alpha = 0.00001
)
importance_reg_alpha <- xgb.importance(train_xgb_name, model = xg_5_reg_alpha)
gg <- xgb.ggplot.importance(importance_subsample_coltree[1:15,], measure = "Frequency", rel_to_first = TRUE)
gg + ggplot2::ylab("Frequency")

# 降低学习率
set.seed(100)
xg_6_eta <- xgb.cv(
  data = train_xgb, 
  max_depth= 6, 
  eta = 0.01, 
  nrounds = 5000, 
  objective = "binary:logistic", 
  max_delta_step = 4, 
  subsample = 0.9,
  verbose = 0,
  gamma = 0, 
  colsample_bytree = 1, 
  scale_pos_weight = 1, 
  min_child_weight = 0,
  nfold = 5, 
  metrics = list("rmse","auc","error"),
  reg_alpha = 0.00001
)
eva_log <- xg_6_eta$evaluation_log
eva_log[eva_log$test_auc_mean == max(eva_log$test_auc_mean),]
