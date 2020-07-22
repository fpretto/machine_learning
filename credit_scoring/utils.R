##------------------------------------------------------------------------------------##
## Helpers - Data Visualization                                                       ##
##------------------------------------------------------------------------------------##

## Boxplot

plotBoxplot <- function(df, varX, varY){
  
  ggplot(data = df) + 
    geom_boxplot(mapping = aes(x = df[,varX], y = df[,varY])) +
    ggtitle(paste('Distribution of', varX, 'vs', varY)) +
    xlab(varX) + 
    ylab(varY)
}

## Histogram

plotHistogram <- function(df, varX, varY){
  
  ggplot(data = df) + 
    geom_histogram(mapping = aes(x = df[,varX], fill = df[,varY]), alpha = 0.5) +
    ggtitle(paste('Distribution of', varX, 'vs', varY)) +
    xlab(varX) + 
    ylab(varY)
}

## Histogram in log scale

plotLogHistogram <- function(df, varX, varY){
  
  ggplot(data = df) + 
    geom_histogram(mapping = aes(x = log(df[,varX]), fill = df[,varY]), alpha = 0.5) +
    ggtitle(paste('Distribution of', varX, 'vs', varY)) +
    xlab(varX) + 
    ylab(varY)
}

## Barplot

plotBarplot <- function(df, varX, varY, horizontal = FALSE){
  
  if (horizontal == TRUE){
    ggplot(data = df) + 
      geom_bar(stat='identity', mapping = aes(x = df[,varX], y = df[,varY]), position = 'dodge') +
      ggtitle(paste(varY,'vs',varX)) +
      coord_flip()+
      xlab(varX) + 
      ylab(varY)
  }
  else { 
    ggplot(data = df) + 
      geom_bar(stat='identity', mapping = aes(x = df[,varX], y = df[,varY]), position = 'dodge') +
      ggtitle(paste(varY,'vs',varX)) +
      xlab(varX) + 
      ylab(varY)
  } 
}

plotBarplotLine <- function(df, varX, varY, varLine){
  
  ggplot(data = df) + 
      geom_bar(stat='identity', mapping = aes(x = df[,varX], y = df[,varY]), position = 'dodge') +
      geom_line(mapping = aes(x = df[,varX], y = df[,varLine])) +
      ggtitle(paste(varY,'por',varX)) +
      xlab(varX) + 
      ylab(varY)

}

# Mode

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Outliers capping

topeoOutliers <- function(df, var, quantile, deviations){

  IQR <- quantile(df[,var], quantile) - quantile(df[,var], 1-quantile)
  
  ubound <- quantile(df[,var], quantile) + IQR * deviations
  
  df[, var] <- ifelse(df[,var] > ubound, ubound, df[,var])
  
  return(df)
}


##------------------------------------------------------------------------------------##
## Helpers - metricas                                                                 ##
##------------------------------------------------------------------------------------##

# Score TP
scoreTP <- function(probs, threshold, labels){
  # Confusion Matrix
  matriz.confusion <- table(Predicted=probs>threshold, Real = labels)

  ta  <- sum(diag(matriz.confusion))/sum(matriz.confusion)
  tfp <- matriz.confusion[2,1]/sum(matriz.confusion[2,])
  tfn <- matriz.confusion[1,2]/sum(matriz.confusion[1,])

  scoreTP <- 0.5 * ta + 0.5 * (1 - (tfp + tfn))
  print(paste('Score TP:', round(scoreTP,2)))
}

# Performance metrics
metricas <- function(probs, threshold, labels){ 
  
  # Confusion Matrix
  if(length(table(Predicted=probs>threshold, Real = labels)) == 4){
    matriz.confusion <- table(Predicted=probs>threshold, Real = labels)
    print(matriz.confusion)
    
    # Accuracy
    accuracy <- sum(diag(matriz.confusion))/sum(matriz.confusion)
    print(paste('Accuracy:', round(accuracy,2)))
    
    # Precision and Recall
    precision <- matriz.confusion[2,2]/sum(matriz.confusion[2,])
    print(paste('Precision:', round(precision,2)))
    
    recall <- matriz.confusion[2,2]/sum(matriz.confusion[,2])
    print(paste('Recall:', round(recall,2)))
    
    # AUC
    library(ROCR) 
    
    pred_auc<-prediction(probs, labels)
    auc<- performance(pred_auc,"auc")
    print(paste('AUC:', round(auc@y.values[[1]],2)))
    
    perf_lr <- performance(prediction(probs, labels), 'tpr', 'fpr')
    plot(perf_lr)
    
    # Score TP
    scoreTP(probs, threshold, labels)
  }
}

##------------------------------------------------------------------------------------##
## Helpers - Random Search                                                            ##
##------------------------------------------------------------------------------------##

# Random Grid Search for Random Forest

rnf_random_grid <- function(size,
                            min_subset, max_subset,
                            min_mtry, max_mtry,
                            min_ntree, max_ntree,
                            min_maxnodes, max_maxnodes) {
  
  rgrid <- data.frame(ntree = sample(c(min_ntree:max_ntree), size = size, replace = TRUE),
                      mtry = sample(c(min_mtry:max_mtry), size = size, replace = TRUE),
                      maxnodes = sample(c(min_maxnodes:max_maxnodes), size = size, replace = TRUE),
                      subset = round(runif(size, min_subset, max_subset), 5))
  return(rgrid)    
}

train_rnf <- function(x_train, y_train, rgrid, x_test, y_test, threshold) {
  
  predicted_models <- list()
  
  for (i in seq_len(nrow(rgrid))) {
    print(i)
    print(rgrid[i,])
    trained_model <- randomForest(x = x_train,
                                  y = y_train,
                                  subset = rgrid[i,4],
                                  mtry = rgrid[i,2], 
                                  ntree = rgrid[i,1], 
                                  maxnodes = rgrid[i,3],
                                  replace = TRUE,
                                  importance = TRUE)
    
    y_train_hat = predict(trained_model, x_train, type = 'prob')[,2]
    y_test_hat = predict(trained_model, x_test, type = 'prob')[,2]
    
    perf_tr <- metricas(y_train_hat, threshold, y_train)
    perf_vd <- metricas(y_test_hat, threshold, y_test)
    #print(c(perf_tr, perf_vd))
    
    predicted_models[[i]] <- list(results = data.frame(rgrid[i,],
                                                       perf_tr = perf_tr,
                                                       perf_vd = perf_vd),
                                  model = trained_model)
    rm(trained_model)
    gc()
  }
  
  return(predicted_models)
}

# Random Grid Search for Gradient Boosting Machines

gbm_random_grid <- function(size,
                            min_ntree, max_ntree,
                            min_depth, max_depth,
                            min_shrinkage, max_shrinkage) {
  
  rgrid <- data.frame(n.trees = sample(c(min_ntree:max_ntree), size = size, replace = TRUE),
                      interaction.depth = sample(c(min_depth:max_depth), size = size, replace = TRUE),
                      shrinkage = round(runif(size, min_shrinkage, max_shrinkage), 5))
  return(rgrid)    
}


train_gbm <- function(train, rgrid, x_test, y_test, threshold) {
  
  predicted_models <- list()
  
  for (i in seq_len(nrow(rgrid))) {
    print(i)
    print(rgrid[i,])
    trained_model <- gbm(RESULT ~ .,
                         data = train,
                         distribution="bernoulli",
                         n.trees=rgrid[i,1],
                         shrinkage=rgrid[i,3],
                         cv.folds=3,
                         interaction.depth=rgrid[i,2])
    
    x_train <- train
    x_train$RESULT <- NULL
    y_train_hat = predict(trained_model, x_train, type = 'response')
    y_test_hat = predict(trained_model, x_valid, type = 'response')
  
    perf_tr <- metricas(y_train_hat, threshold, train$RESULT)
    perf_vd <- metricas(y_test_hat, threshold, y_test)
    #print(c(perf_tr, perf_vd))
    
    predicted_models[[i]] <- list(results = data.frame(rgrid[i,],
                                                       perf_tr = perf_tr,
                                                       perf_vd = perf_vd))
    rm(trained_model)
    gc()
  }
  
  return(predicted_models)
}


# Results analysis
result_table <- function(pred_models) {
  res_table <- data.frame()
  i <- 1
  for (m in pred_models) {
    res_table <- rbind(res_table, data.frame(i = i, m$results))
    i <- i + 1
  }
  res_table <- res_table[order(-res_table$perf_vd),]
  return(res_table)
}
