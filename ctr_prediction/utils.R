##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Definicion de funciones                                                            ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

#### Carga de 1 dataset sampleado
load_csv_data <- function(csv_file, sample_ratio = 1, drop_cols = NULL, sel_cols = NULL) { 
  dt <- fread(csv_file, header = TRUE, sep = ",", stringsAsFactors = TRUE, na.strings = "", 
              drop = drop_cols, select = sel_cols, showProgress = TRUE) 
  rows <- nrow(dt)
  if (sample_ratio < 1) { 
    sample_size <- as.integer(sample_ratio * rows) 
    dt <- dt[sample(.N, sample_size)] 
  }
  return(dt)
}

#### Carga de N datasets sampleados
load_train_data <- function(data_dir, sample_ratio=1, drop_cols=NULL, 
                            sel_cols=NULL) { 
  train_days <- seq(15, 21, by=1) 
  data_set <- list() 
  
  for (i in train_days){
    csv_file = paste(data_dir, 'ctr_', as.character(i), '.csv', sep="")
    df <- load_csv_data(csv_file, sample_ratio = sample_ratio, drop_cols = drop_cols, sel_cols = sel_cols)
    data_set <- rbind(data_set, df)
  }
  
  csv_file = paste(data_dir, 'ctr_test.csv', sep="")
  test <- load_csv_data(csv_file, sample_ratio = 1, drop_cols = drop_cols, sel_cols = sel_cols)
  data_set <- rbind(data_set, test, fill= TRUE)
  return(data_set) 
}

#### Eliminacion de variables con una proporcion de nulos mayor al umbral definido
drop_null_cols <- function(df, umbral = 0.5){
  # Proporcion de NAs por columna
  na_columnas <- data.frame('pct_nulos'= colMeans(is.na(df)))
  na_columnas$variable <- rownames(na_columnas)
  
  # Eliminacion de columnas con mas del 50% de nulos
  rm_columnas <- na_columnas[na_columnas$pct_nulos > umbral,]
  
  print(paste('Cantidad de columnas eliminadas:', length(rm_columnas$variable)))
  print(paste('Columnas eliminadas:', rm_columnas$variable))
  
  df <- select(df, -c(rm_columnas$variable))
  
  return(df)
}


#### One-Hot Encoding

one_hot_sparse <- function(data_set) {
  
  created <- FALSE
  
  if (sum(sapply(data_set, is.numeric)) > 0) {  # Si hay, Pasamos los numéricos a una matriz esparsa (sería raro que no estuviese, porque "Label"  es numérica y tiene que estar sí o sí)
    out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.numeric), with = FALSE]), "dgCMatrix")
    created <- TRUE
  }
  
  if (sum(sapply(data_set, is.logical)) > 0) {  # Si hay, pasamos los lógicos a esparsa y lo unimos con la matriz anterior
    if (created) {
      out_put_data <- cbind2(out_put_data,
                             as(as.matrix(data_set[,sapply(data_set, is.logical),
                                                   with = FALSE]), "dgCMatrix"))
    } else {
      out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.logical), with = FALSE]), "dgCMatrix")
      created <- TRUE
    }
  }
  
  # Identificamos las columnas que son factor (OJO: el data.frame no debería tener character)
  fact_variables <- names(which(sapply(data_set, is.factor)))
  
  # Para cada columna factor hago one hot encoding
  i <- 0
  
  for (f_var in fact_variables) {
    
    f_col_names <- levels(data_set[[f_var]])
    f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
    j_values <- as.numeric(data_set[[f_var]])  # Se pone como valor de j, el valor del nivel del factor
    
    if (sum(is.na(j_values)) > 0) {  # En categóricas, trato a NA como una categoría más
      j_values[is.na(j_values)] <- length(f_col_names) + 1
      f_col_names <- c(f_col_names, paste(f_var, "NA", sep = "_"))
    }
    
    if (i == 0) {
      fact_data <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                x = rep(1, nrow(data_set)),
                                dims = c(nrow(data_set), length(f_col_names)))
      fact_data@Dimnames[[2]] <- f_col_names
    } else {
      fact_data_tmp <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                    x = rep(1, nrow(data_set)),
                                    dims = c(nrow(data_set), length(f_col_names)))
      fact_data_tmp@Dimnames[[2]] <- f_col_names
      fact_data <- cbind(fact_data, fact_data_tmp)
    }
    
    i <- i + 1
  }
  
  if (length(fact_variables) > 0) {
    if (created) {
      out_put_data <- cbind(out_put_data, fact_data)
    } else {
      out_put_data <- fact_data
      created <- TRUE
    }
  }
  return(out_put_data)
}

#### Generacion de Random Grid
random_grid <- function(size,
                        min_nrounds, max_nrounds,
                        min_max_depth, max_max_depth,
                        min_eta, max_eta,
                        min_gamma, max_gamma,
                        min_colsample_bytree, max_colsample_bytree,
                        min_min_child_weight, max_min_child_weight,
                        min_subsample, max_subsample) {
  
  rgrid <- data.frame(nrounds = sample(c(min_nrounds:max_nrounds),
                                       size = size, replace = TRUE),
                      max_depth = sample(c(min_max_depth:max_max_depth),
                                         size = size, replace = TRUE),
                      eta = round(runif(size, min_eta, max_eta), 5),
                      gamma = round(runif(size, min_gamma, max_gamma), 5),
                      colsample_bytree = round(runif(size, min_colsample_bytree,
                                                     max_colsample_bytree), 5),
                      min_child_weight = round(runif(size, min_min_child_weight,
                                                     max_min_child_weight), 5),
                      subsample = round(runif(size, min_subsample, max_subsample), 5))
  return(rgrid)    
}

#### Entrenamiento con Random Search

train_xgboost <- function(data_train, data_val, rgrid) {
  
  watchlist <- list(train = data_train, valid = data_val)
  
  predicted_models <- list()
  
  for (i in seq_len(nrow(rgrid))) {
    print(i)
    print(rgrid[i,])
    trained_model <- xgb.train(data = data_train,
                               params=as.list(rgrid[i, c("max_depth",
                                                         "eta",
                                                         "gamma",
                                                         "colsample_bytree",
                                                         "subsample",
                                                         "min_child_weight")]),
                               nrounds = rgrid[i, "nrounds"],
                               watchlist = watchlist,
                               objective = "binary:logistic",
                               eval.metric = "auc",
                               print_every_n = 10)
    
    perf_tr <- tail(trained_model$evaluation_log, 1)$train_auc
    perf_vd <- tail(trained_model$evaluation_log, 1)$valid_auc
    print(c(perf_tr, perf_vd))
    
    predicted_models[[i]] <- list(results = data.frame(rgrid[i,],
                                                       perf_tr = perf_tr,
                                                       perf_vd = perf_vd),
                                  model = trained_model)
    rm(trained_model)
    gc()
  }
  
  return(predicted_models)
}


#### Tabla de resultados 
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

#### Rebalanceo con 35% de la muestra
balanceo_dataframe <- function(data_dir){
  train <- data_dir[!is.na(Label),]
  test <- data_dir[is.na(Label),]
  data_set <- list() 
  filtro <- list()
  sample_parcial <- list()
  
  
  combinatoria = c("lunes Manana","lunes Tarde","lunes Noche 1","lunes Noche 2",
                   "martes Manana","martes Tarde","martes Noche 1","martes Noche 2",
                   "miércoles Manana","miércoles Tarde","miércoles Noche 1","miércoles Noche 2",
                   "jueves Manana","jueves Tarde","jueves Noche 1","jueves Noche 2",
                   "viernes Manana","viernes Tarde","viernes Noche 1","viernes Noche 2",
                   "sábado Manana","sábado Tarde","sábado Noche 1","sábado Noche 2",
                   "domingo Manana","domingo Tarde","domingo Noche 1","domingo Noche 2")

  necesario = c(103487,	196583,	84529,	101953,
                143183,	138234,	84561,	129333,
                138285,	96176,	67920,	127025,
                85708,	103732,	60136,	98533,
                76402,	119042,	70163,	115079,
                134717,	133065,	76859,	133320,
                103678,	148119,	37529,	118256)
  
  balanceo_df <- data.frame (cbind(combinatoria, necesario))
  #balanceo_df$necesario <- as.integer(as.character(balanceo_df$necesario))
  #str(balanceo_df)
  
  for (i in 1:nrow(balanceo_df) ) {
    print(balanceo_df[i,1])
    filtro <- train[train$weekday_turno == balanceo_df[i,1],]
    print("A filtrar:")
    print(nrow(filtro))
    sample_parcial <- filtro[sample(.N, as.integer(as.character(balanceo_df$necesario[i])), replace = TRUE)]
    #sample_parcial <- filtro[sample(.N, 115079, replace = TRUE)]
    print("Filtrado:")
    print(nrow(sample_parcial))
    data_set <- rbind(data_set, sample_parcial)
    print("Longitud data_set")
    print(nrow(data_set))
  }
  data_set <- rbind(data_set, test, fill= TRUE)
  return(data_set) 
}