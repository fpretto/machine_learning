rm(list=ls())
gc()
options(scipen = 999)
set.seed(1990)

##------------------------------------------------------------------------------------##
## Carga de librerias y funciones                                                     ##
##------------------------------------------------------------------------------------##

library(dplyr)
library(ggplot2)
library(gtools)
library(glmnet)
library(randomForest)
library(gbm)
library(DMwR)

setwd('C:/Repo/Github/machine_learning/credit_scoring/')
source('utils.R')


##------------------------------------------------------------------------------------##
## Carga de datasets                                                                  ##
##------------------------------------------------------------------------------------##

trainFile <- 'TRAIN_data.txt'
dfTrain <- read.csv(trainFile, header = TRUE)

testFile <- 'TEST_data.txt'
dfTest <- read.csv(testFile, header = TRUE)

# Separacion en Train/Test
vd_index <- sample(rownames(dfTrain), 0.3 * length(rownames(dfTrain)))
train <- dfTrain[setdiff(rownames(dfTrain), vd_index),]
valid <- dfTrain[vd_index,]

##------------------------------------------------------------------------------------##
## Data Cleaning                                                                      ##
##------------------------------------------------------------------------------------##


## Proporcion de NAs por columna
na_columnas <- data.frame('pct_nulos'= colMeans(is.na(train)))
# No hay nulos en el dataframe

dim(train)
colnames(train)
str(train)
summary(train)

# Transformacion de la variable "INCOME.TYPE" a factor ya que se trata de categorias
train$INCOME.TYPE <- as.factor(train$INCOME.TYPE)

##------------------------------------------------------------------------------------##
## Exploratory Data Analysis / Feature Engineering                                    ##
##------------------------------------------------------------------------------------##

#                           Medidas descriptivas del target                            #
########################################################################################

prop.table(table(train$RESULT))*100


# Transformacion del target para simplificar el EDA
train$fl_RESULT <- ifelse(train$RESULT == "FUNDED", 1, 0)

#                                Analisis Univariado                                   #
########################################################################################

####### Property
## Distribucion de Property Value por Label
plotBoxplot(train, 'RESULT', 'PROPERTY.VALUE')
plotHistogram(train, 'PROPERTY.VALUE', 'RESULT')
plotLogHistogram(train, 'PROPERTY.VALUE', 'RESULT')

train <- topeoOutliers(train, 'PROPERTY.VALUE', 0.95, 5)

train <- mutate(train, decilPropertyValue = ntile(train$PROPERTY.VALUE,10))
train$decilPropertyValue <- as.factor(train$decilPropertyValue)
decilPropertyValue <- as.data.frame(train %>% group_by(decilPropertyValue) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(decilPropertyValue, 'decilPropertyValue', 'TASA.FUNDED')

## Tasa de Funded por Property Type
Property <- as.data.frame(train %>% group_by(PROPERTY.TYPE) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(Property, 'PROPERTY.TYPE', 'COUNT', horizontal = TRUE)
plotBarplot(Property, 'PROPERTY.TYPE', 'TASA.FUNDED', horizontal = TRUE)


# Conclusion: Se realizo una transformacion logaritmica a PROPERTY.VALUE para minimizar el efecto
#   de los outliers. Igualmente, parece haber algunos valores muy atipicos, por lo que dichos valores se topearon segun una cota superior
#   La distribucion para los casos a los que se otorgaron el prestamo parece muy similar
#   a la distribucion de los que no se otorgaron, no aportando mucha informacion para separar las clases.
#   Sin embargo, discretizando la variable y analizando por deciles, se puede observar que a mayor valor
#   de la propiedad, menor es la tasa de prestamos otorgados.
#   Por otro lado, el tipo de propiedad tambien parece tener incidencia en la tasa de prestamos otorgados.
#################

####### Mortgage

### Distribucion de Mortgage Payment por Label
plotBoxplot(train, 'RESULT', 'MORTGAGE.PAYMENT')
plotHistogram(train, 'MORTGAGE.PAYMENT', 'RESULT')

train <- topeoOutliers(train, 'MORTGAGE.PAYMENT', 0.95, 4)

train <- mutate(train, decilMortgagePayment = ntile(train$MORTGAGE.PAYMENT,10))
train$decilMortgagePayment <- as.factor(train$decilMortgagePayment)
decilMortgagePayment <- as.data.frame(train %>% group_by(decilMortgagePayment) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(decilMortgagePayment, 'decilMortgagePayment', 'TASA.FUNDED')

### Distribucion de Mortgage Amount por Label
plotBoxplot(train, 'RESULT', 'MORTGAGE.AMOUNT')
plotHistogram(train, 'MORTGAGE.AMOUNT', 'RESULT')

train <- topeoOutliers(train, 'MORTGAGE.AMOUNT', 0.95, 3)

## Tasa de Funded por Mortgage Amount
train <- mutate(train, decilMortgageAmount = ntile(train$MORTGAGE.AMOUNT,10))
train$decilMortgageAmount <- as.factor(train$decilMortgageAmount)
decilMortgageAmount <- as.data.frame(train %>% group_by(decilMortgageAmount) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(decilMortgageAmount, 'decilMortgageAmount', 'TASA.FUNDED')

### Distribucion de Mortgage Term por Label
train$TERM <- as.factor(train$TERM)
MortgageTerm <- as.data.frame(train %>% group_by(TERM) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(MortgageTerm, 'TERM', 'TASA.FUNDED')

### Tasa de Funded por Mortgage Purpose
Mortgage <- as.data.frame(train %>% group_by(MORTGAGE.PURPOSE) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(Mortgage, 'MORTGAGE.PURPOSE', 'TASA.FUNDED')


# Conclusion: En primer lugar, se topearon outliers para que su presencia no afecte a la distribucion de la variable.
#   Discretizando la variable MORTGAGE.PAYMENT parece evidenciarse diferencias en la tasa de prestamos
#   otorgados, siendo esta menor al aumentar la cuota de la hipoteca. Para MORTGAGE.AMOUNT parece haber 
#   una relacion similar, aunque bastante menos evidente.
#   Por otro lado, la duracion solicitada (TERM) parece tener un comportamiento similar y entre los motivos 
#   de la solicitud (Purchase/Refinance) tambien parece haber una leve diferencia en la tasa a favor de
#   la categoria "Purchase". 
#################

# GDS

summary(train$GDS)
train <- filter(train, GDS <= 100 & GDS >= 0) # Eliminamos valores atipicos

### Distribucion de GDS por Label
plotBoxplot(train, 'RESULT', 'GDS')
plotHistogram(train, 'GDS', 'RESULT')

## Tasa de Funded por GDS
train <- mutate(train, decilGDS = ntile(train$GDS,10))
train$decilGDS <- as.factor(train$decilGDS)
decilGDS <- as.data.frame(train %>% group_by(decilGDS) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(decilGDS, 'decilGDS', 'TASA.FUNDED')

# Conclusion: La variable GDS no parece tener demasiada influencia en la tasa de prestamos otorgados.
#################

# TDS

summary(train$TDS)
train <- filter(train, TDS <= 100 & TDS >= 0) # Eliminamos valores atipicos

### Distribucion de TDS por Label
plotBoxplot(train, 'RESULT', 'TDS')
plotHistogram(train, 'TDS', 'RESULT')

## Tasa de Funded por TDS
train <- mutate(train, decilTDS = ntile(train$TDS,10))
train$decilTDS <- as.factor(train$decilTDS)
decilTDS <- as.data.frame(train %>% group_by(decilTDS) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(decilTDS, 'decilTDS', 'TASA.FUNDED')

# Conclusion: La variable TDS parece separar un poco mejor las clases que GDS, aunque la diferencia es
#   bastante pequena.
#################

# LTV

summary(train$LTV)

### Distribucion de LTV por Label
plotBoxplot(train, 'RESULT', 'LTV')
plotHistogram(train, 'LTV', 'RESULT')

## Tasa de Funded por LTV
train <- mutate(train, decilLTV = ntile(train$LTV,10))
train$decilLTV <- as.factor(train$decilLTV)
decilLTV <- as.data.frame(train %>% group_by(decilLTV) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(decilLTV, 'decilLTV', 'TASA.FUNDED')

# Conclusion: Parecerìa que haber alguna pequeña separacion de las clases, aunque no demasiado pronunciada.
#################

# Interest Rate
train$factorRATE <- as.factor(train$RATE)
## Tasa de Funded por Interest Rate
RATE <- as.data.frame(train %>% group_by(factorRATE) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(RATE, 'factorRATE', 'TASA.FUNDED')

train$rateRango <- as.factor(ifelse(train$RATE <= 5, 'bajoRATE',
                                    ifelse(train$RATE < 7, 'medioRATE', 'altoRATE')))

rangosRATE <- as.data.frame(train %>% group_by(rateRango) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(rangosRATE, 'rateRango', 'TASA.FUNDED')

# Conclusion: Pareceria que a medida que aumenta la tasa de interes, mayor cantidad de prestamos se otorgan.
#     Sin embargo, al analizar la cantidad de casos existentes por cada tasa de interes, se ve que para algunas
#     la frecuencia es muy baja. Es por esto que se decidiò categorizar las tasas de interes en Alta, Media y Baja
#     para tratar de diferenciar el comportamiento.
######################################################################################

# Amortization

### Tasa de Funded por Meses de Amortizacion Solicitados
Amortization <- as.data.frame(train %>% group_by(AMORTIZATION) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(Amortization, 'AMORTIZATION', 'TASA.FUNDED')

# Conclusion: 
######################################################################################

# Payment Frequency

### Tasa de Funded por Payment Frequency
PaymentFreq <- as.data.frame(train %>% group_by(PAYMENT.FREQUENCY) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(PaymentFreq, 'PAYMENT.FREQUENCY', 'TASA.FUNDED', horizontal = TRUE)

# Conclusion: 
######################################################################################

# FSA

### Tasa de Funded por FSA
FSA <- as.data.frame(train %>% group_by(FSA) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(FSA, 'FSA', 'TASA.FUNDED', horizontal = TRUE)

# Variables para caracterizar los codigos FSA en clusters
varsClusters <- c("FSA","PROPERTY.VALUE", "MORTGAGE.PAYMENT", "GDS", "LTV", "TDS",
                  "MORTGAGE.AMOUNT", "INCOME", "CREDIT.SCORE", "RATE", "fl_RESULT")


FSAClusters <- train[, varsClusters]
FSAClusters <- aggregate(FSAClusters[, 2:11], list(FSAClusters$FSA), mean)

# Elbow Method para determinar al cantidad de clusters
evol_variabilidad <- data.frame()
for (i in c(1:20)) {
  clusters <- kmeans(FSAClusters[,2:11], centers=i, iter.max=30,  nstart=20)
  evol_variabilidad <- rbind(evol_variabilidad,
                             data.frame(k=i,
                                        var=clusters$tot.withinss))
}
plot(c(1:20), evol_variabilidad$var, type="o", xlab="# Clusters", ylab="tot.withinss", main = "Elbow method")

# En base al elbow method, utilizamos 4 clusters
clusters <- kmeans(FSAClusters[,2:10], centers=6, iter.max=30,  nstart=20)

clusters$size

FSAClusters <- cbind(FSAClusters,'FSACluster' = clusters$cluster)

train <- merge(train, FSAClusters[, c('Group.1', 'FSACluster')], by.x= 'FSA', by.y = 'Group.1', all.x = TRUE)
train$FSACluster <- as.factor(train$FSACluster)
FSAClust <- as.data.frame(train %>% group_by(FSACluster) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(FSAClust, 'FSACluster', 'TASA.FUNDED', horizontal = TRUE)


# Conclusion: Algunos clusters parecen discretizar la tasa de otorgacion de prestamos, auqnue la diferencia no es
#       muy marcada.
######################################################################################

# Age.Range

### Tasa de Funded por Rango de Edad
rangosEdad <- as.data.frame(train %>% group_by(AGE.RANGE) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(rangosEdad, 'AGE.RANGE', 'TASA.FUNDED')

# Conclusion: No parece haber grandes diferencias entre los distintos rangos de edad individualmente.
######################################################################################

# Gender

### Tasa de Funded por Genero
Gender <- as.data.frame(train %>% group_by(GENDER) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(Gender, 'GENDER', 'TASA.FUNDED')

# Conclusion: No parece haber grandes diferencias entre los distintos generos individualmente.
######################################################################################

# Income

### Distribucion de Label por Income
plotBoxplot(train, 'RESULT', 'INCOME')
plotHistogram(train, 'INCOME', 'RESULT')
plotLogHistogram(train, 'INCOME', 'RESULT')

train <- mutate(train, decilINCOME = ntile(log(train$INCOME),10))
train$decilINCOME <- as.factor(train$decilINCOME)
decilINCOME <- as.data.frame(train %>% group_by(decilINCOME) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(decilINCOME, 'decilINCOME', 'TASA.FUNDED')

### Tasa de Funded por Income Type
IncomeType <- as.data.frame(train %>% group_by(INCOME.TYPE) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(IncomeType, 'INCOME.TYPE', 'COUNT')
plotBarplot(IncomeType, 'INCOME.TYPE', 'TASA.FUNDED')

freqIncomeType <- table(train$INCOME.TYPE)
listExclude <- rownames(freqIncomeType[freqIncomeType < 30] )
train$INCOME.TYPE <- ifelse(train$INCOME.TYPE %in% listExclude, "-2", train$INCOME.TYPE)
train$INCOME.TYPE <- as.factor(train$INCOME.TYPE)


# Conclusion: Se realizo una transformacion logaritmica para mitigar el efecto de los outliers. Mas alla de esto, no
#     se observa que haya tasas de prestamos significativamente diferentes entre los distintos deciles.
#     Para la variable categorica Income Type, si pareceria haber algunas categorias que discretizan el target mejor que otras.
#     Sin embargo, dado que algunas categorias tienen muy pocos casos, estas tasas podrian no ser significativas, por lo que se
#     agruparon en una nueva categoria "Otros", notada como -2.
######################################################################################

# Naics Code
### Tasa de Funded por Naics Code
naicsCode <- as.data.frame(train %>% group_by(NAICS.CODE) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2), COUNT = n()))
plotBarplot(naicsCode, 'NAICS.CODE', 'TASA.FUNDED', horizontal = TRUE)

# Conclusion: Algunos codigos NAICS parecen tener tasas de rpestamos significativamente menor al resto.
######################################################################################

# Credit Score

### Distribucion de Label por Credit Score
plotBoxplot(train, 'RESULT', 'CREDIT.SCORE')
plotHistogram(train, 'CREDIT.SCORE', 'RESULT')

train <- mutate(train, decilCreditScore = ntile(train$CREDIT.SCORE,10))
train$decilCreditScore <- as.factor(train$decilCreditScore)
decilCreditScore <- as.data.frame(train %>% group_by(decilCreditScore) %>% summarise(TASA.FUNDED = round(mean(fl_RESULT)*100,2)))
plotBarplot(decilCreditScore, 'decilCreditScore', 'TASA.FUNDED')

# Conclusion: 
######################################################################################

#                               Analisis Multivariado                                  #
########################################################################################

# Dado que las variables GDS y TDS son practicamente iguales (se agrega un elemento de costo 'Otros'),
# es muy probable que sean colineales y por lo tanto esten altamente correlacionadas.

corrs <- cor(train)
corrplot(corrs, method ='circle')

# Conclusion: Dada la alta correlacion entre ambas variables, es conveniente desestimar una de ellas,
#             ya que aportan la misma informacion, e incluir ambas puede afectar al modelo.
######################################################################################



##------------------------------------------------------------------------------------##
## Feature Engineering para Validation y Test Sets                                    ##
##------------------------------------------------------------------------------------##

### Validation Set

valid <- merge(valid, FSAClusters[, c('Group.1', 'FSACluster')], by.x= 'FSA', by.y = 'Group.1', all.x = TRUE)
valid <- mutate(valid, decilPropertyValue = ntile(valid$PROPERTY.VALUE,10))
valid <- mutate(valid, decilCreditScore = ntile(valid$CREDIT.SCORE,10))
valid <- mutate(valid, decilTDS = ntile(valid$TDS,10))
valid <- mutate(valid, decilLTV = ntile(valid$LTV,10))
valid <- mutate(valid, decilMortgagePayment = ntile(valid$MORTGAGE.PAYMENT,10))
valid <- mutate(valid, decilMortgageAmount = ntile(valid$MORTGAGE.AMOUNT,10))
valid$rateRango <- as.factor(ifelse(valid$RATE <= 5, 'bajoRATE',
                                    ifelse(valid$RATE < 7, 'medioRATE', 'altoRATE')))
valid$FSACluster <- as.factor(valid$FSACluster)
valid$FSACluster[is.na(valid$FSACluster)] <- "4"
valid$TERM <- as.factor(valid$TERM)

valid$decilCreditScore <- as.factor(valid$decilCreditScore)
valid$decilPropertyValue <- as.factor(valid$decilPropertyValue)
valid$decilTDS <- as.factor(valid$decilTDS)
valid$decilLTV <- as.factor(valid$decilLTV)
valid$decilMortgagePayment <- as.factor(valid$decilMortgagePayment)
valid$decilMortgageAmount <- as.factor(valid$decilMortgageAmount)

valid$INCOME.TYPE <- as.factor(valid$INCOME.TYPE)
listExclude <- append(listExclude, "13")
valid$INCOME.TYPE <- ifelse(valid$INCOME.TYPE %in% listExclude, "-2", valid$INCOME.TYPE)
valid$INCOME.TYPE <- as.factor(valid$INCOME.TYPE)

### Test Set

dfTest <- merge(dfTest, FSAClusters[, c('Group.1', 'FSACluster')], by.x= 'FSA', by.y = 'Group.1', all.x = TRUE)
dfTest <- mutate(dfTest, decilPropertyValue = ntile(dfTest$PROPERTY.VALUE,10))
dfTest <- mutate(dfTest, decilCreditScore = ntile(dfTest$CREDIT.SCORE,10))
dfTest <- mutate(dfTest, decilTDS = ntile(dfTest$TDS,10))
dfTest <- mutate(dfTest, decilLTV = ntile(dfTest$LTV,10))
dfTest <- mutate(dfTest, decilMortgagePayment = ntile(dfTest$MORTGAGE.PAYMENT,10))
dfTest <- mutate(dfTest, decilMortgageAmount = ntile(dfTest$MORTGAGE.AMOUNT,10))
dfTest$rateRango <- as.factor(ifelse(dfTest$RATE <= 5, 'bajoRATE',
                                     ifelse(dfTest$RATE < 7, 'medioRATE', 'altoRATE')))
dfTest$FSACluster <- as.factor(dfTest$FSACluster)
dfTest$FSACluster[is.na(dfTest$FSACluster)] <- "4"
dfTest$TERM <- as.factor(dfTest$TERM)

dfTest$decilCreditScore <- as.factor(dfTest$decilCreditScore)
dfTest$decilPropertyValue <- as.factor(dfTest$decilPropertyValue)
dfTest$decilTDS <- as.factor(dfTest$decilTDS)
dfTest$decilLTV <- as.factor(dfTest$decilLTV)
dfTest$decilMortgagePayment <- as.factor(dfTest$decilMortgagePayment)
dfTest$decilMortgageAmount <- as.factor(dfTest$decilMortgageAmount)

dfTest$INCOME.TYPE <- as.factor(dfTest$INCOME.TYPE)
listExclude <- append(listExclude, "13")
dfTest$INCOME.TYPE <- ifelse(dfTest$INCOME.TYPE %in% listExclude, "-2", dfTest$INCOME.TYPE)
dfTest$INCOME.TYPE <- as.factor(dfTest$INCOME.TYPE)


##------------------------------------------------------------------------------------##
## Seleccion de variables                                                             ##
##------------------------------------------------------------------------------------##

## Seleccion de variables a partir del analisis univariado

varsSelected <- c('decilPropertyValue', 'decilMortgagePayment', 'TERM', 'MORTGAGE.PURPOSE',
                  'decilTDS', 'decilLTV', 'AMORTIZATION', 'NAICS.CODE', 'decilCreditScore', 
                  'PROPERTY.TYPE', 'decilMortgageAmount', 'rateRango', 'FSACluster','RESULT')

modelTrain <- train[,varsSelected]
modelValid <- valid[,varsSelected]

x_train = modelTrain[, 1:length(modelTrain)-1]
y_train = modelTrain[,'RESULT']

x_valid = modelValid[, 1:length(modelValid)-1]
y_valid = modelValid[,'RESULT']

y_train <- as.factor(ifelse(y_train == 'FUNDED', 1, 0))
y_valid <- as.factor(ifelse(y_valid == 'FUNDED', 1, 0))

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Modelado                                                                           ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

##------------------------------------------------------------------------------------##
## Regresion Logistica                                                                ##
##------------------------------------------------------------------------------------##

### Iteracion 1 - Modelo base
reg.log = glm(RESULT ~ .,  data = modelTrain,
              family = binomial(link = "logit"))

summary(reg.log)

# Prediccion
y_hat_train = predict(reg.log, newdata = modelTrain, type = 'response')
y_hat = predict(reg.log, newdata = modelValid, type = 'response')
# Metricas de performance
metricas(y_hat_train, 0.7, modelTrain$RESULT)
metricas(y_hat, 0.7, modelValid$RESULT)

### Iteracion 2 - Modelo base + Regularizacion LASSO

x_train <- modelTrain ; x_train$RESULT <- NULL ; x_train$fl_RESULT <- NULL
x_train <- model.matrix( ~ .-1, x_train)

y_train <- modelTrain[,'RESULT'] ; y_train <- as.matrix(y_train)

grid.l =exp(seq(10 , -10 , length = 100))
reg.log.lasso = glmnet(x = x_train , y = y_train,
                       family = 'binomial', alpha = 1 , lambda = grid.l)

plot(reg.log.lasso, xvar = "lambda", label = TRUE, main = "Evolución Coeficientes y # Variables por Lambda")

# CV para LASSO (alpha = 1)
cv.out = cv.glmnet(x_train, y_train, family = 'binomial', alpha = 1, nfolds = 10)
plot (cv.out)
bestlam = cv.out$lambda.min
bestlam

reg.log.lasso = glmnet(x = x_train , y = y_train,
                       family = 'binomial', alpha = 1 , lambda = bestlam)


x_valid <- modelValid ; x_valid$RESULT <- NULL ; x_valid$fl_RESULT <- NULL
x_valid <- model.matrix( ~ .-1, x_valid)

# Prediccion
y_hat_train = predict(reg.log.lasso, newx = x_train, type = 'response', s = bestlam)
y_hat = predict(reg.log.lasso, newx = x_valid, type = 'response', s = bestlam)
# Metricas de performance
metricas(y_hat_train, 0.7, modelTrain$RESULT)
metricas(y_hat, 0.7, modelValid$RESULT)

### Iteracion 3 - SMOTE + Modelo base + Regularizacion LASSO
smotedTrain <- SMOTE(RESULT ~ ., data = modelTrain, perc.over = 200)
prop.table(table(smotedTrain$RESULT))
prop.table(table(modelTrain$RESULT))

x_train <- smotedTrain ; x_train$RESULT <- NULL ; x_train$fl_RESULT <- NULL
x_train <- model.matrix( ~ .-1, x_train)
y_train <- smotedTrain[,'RESULT'] ; y_train <- as.matrix(y_train)

# CV para LASSO (alpha = 1)
cv.out = cv.glmnet(x_train, y_train, family = 'binomial', alpha = 1, nfolds = 10)
plot (cv.out)
bestlam = cv.out$lambda.min
bestlam

reg.log.lasso = glmnet(x = x_train , y = y_train,
                       family = 'binomial', alpha = 1 , lambda = bestlam)

# Prediccion
y_hat_train = predict(reg.log.lasso, newx = x_train, type = 'response', s = bestlam)
y_hat = predict(reg.log.lasso, newx = x_valid, type = 'response', s = bestlam)
# Metricas de performance
metricas(y_hat_train, 0.65, smotedTrain$RESULT)
metricas(y_hat, 0.65, modelValid$RESULT)

##------------------------------------------------------------------------------------##
## Random Forest                                                                      ##
##------------------------------------------------------------------------------------##

### Iteracion 1 - Modelo base
r.forest = randomForest(x = x_train,
                        y = y_train,
                        subset = 1,
                        mtry = 8,
                        ntree = 300, 
                        maxnodes = 20,
                        replace = TRUE,
                        importance = TRUE)

# Variable Importance
varImpPlot(r.forest)

barplot(sort(r.forest$importance[,4], decreasing = TRUE),
        main = "Importancia Relativa Var en % de Decreased Gini", col = "lightblue",
        horiz = TRUE, las = 1, cex.names = .6, xlim = c(-1, 150))

# Prediccion
y_hat_train = predict(r.forest, x_train, type = 'prob')[,2]
y_hat = predict(r.forest, x_valid, type = 'prob')[,2]

# Metricas de performance
metricas(y_hat_train, 0.9, y_train)
metricas(y_hat, 0.9, y_valid)

### Iteracion 2 - Modelo base + Grid Search

## Random Grid Search
rnf_grid <- rnf_random_grid(size = 30,
                            min_subset = 0.7, max_subset = 1,
                            min_mtry = 6, max_mtry = 12,
                            min_ntree = 300, max_ntree = 500,
                            min_maxnodes = 15, max_maxnodes = 30)

rnf_models <- train_rnf(x_train, y_train, rnf_grid, x_valid, y_valid, 0.9)

# Analisis de las salidas
res_table <- result_table(rnf_models)
print(res_table)

# Modelo optimo
r.forest.optimal = randomForest(x = x_train,
                        y = y_train,
                        subset = 0.94743,
                        mtry = 6,
                        ntree = 321, 
                        maxnodes = 15,
                        replace = TRUE,
                        importance = TRUE)

# Variable Importance
barplot(sort(r.forest.optimal$importance[,4], decreasing = TRUE),
        main = "Importancia Relativa Var en % de Decreased Gini", col = "lightblue",
        horiz = TRUE, las = 1, cex.names = .6, xlim = c(-1, 150))

# Prediccion
y_hat_train = predict(r.forest.optimal, x_train, type = 'prob')[,2]
y_hat = predict(r.forest.optimal, x_valid, type = 'prob')[,2]

# Metricas de performance
metricas(y_hat_train, 0.9, y_train)
metricas(y_hat, 0.9, y_valid)


### Iteracion 3 - Modelo base + Grid Search + SMOTE
smotedTrain <- SMOTE(RESULT ~ ., data = modelTrain, perc.over = 200)
prop.table(table(smotedTrain$RESULT))
prop.table(table(modelTrain$RESULT))

x_train <- smotedTrain ; x_train$RESULT <- NULL ; x_train$fl_RESULT <- NULL
y_train = smotedTrain[,'RESULT'] ; y_train <- as.factor(ifelse(y_train == 'FUNDED', 1, 0))

rnf_grid <- rnf_random_grid(size = 30,
                            min_subset = 0.7, max_subset = 1,
                            min_mtry = 6, max_mtry = 12,
                            min_ntree = 300, max_ntree = 500,
                            min_maxnodes = 15, max_maxnodes = 30)

rnf_models <- train_rnf(x_train, y_train, rnf_grid, x_valid, y_valid, 0.9)

# Analisis de las salidas
res_table <- result_table(rnf_models)
print(res_table)

# Modelo optimo
r.forest.optimal = randomForest(x = x_train,
                                y = y_train,
                                subset = 0.73668,
                                mtry = 11,
                                ntree = 414, 
                                maxnodes = 26,
                                replace = TRUE,
                                importance = TRUE)

# Variable Importance
barplot(sort(r.forest.optimal$importance[,4], decreasing = TRUE),
        main = "Importancia Relativa Var en % de Decreased Gini", col = "lightblue",
        horiz = TRUE, las = 1, cex.names = .6, xlim = c(-1, 150))

# Prediccion
y_hat_train = predict(r.forest.optimal, x_train, type = 'prob')[,2]
y_hat = predict(r.forest.optimal, x_valid, type = 'prob')[,2]

# Metricas de performance
metricas(y_hat_train, 0.9, y_train)
metricas(y_hat, 0.9, y_valid)

##------------------------------------------------------------------------------------##
## Gradient Boosting Machines                                                         ##
##------------------------------------------------------------------------------------##

### Iteracion 1 - Modelo base
modelTrain$RESULT <- as.numeric(modelTrain$RESULT)
modelTrain = transform(modelTrain, RESULT = RESULT-1)

g.boosting = gbm(RESULT ~ .,
                 data = modelTrain,
                 distribution="bernoulli",
                 n.trees=5000,
                 shrinkage=0.01,
                 cv.folds=5,
                 interaction.depth=4)

summary(g.boosting)
best.iter = gbm.perf(g.boosting, method="cv")

# Prediccion
y_hat_train = predict(g.boosting, newdata = x_train, n.trees=5000, type = "response")
y_hat = predict(g.boosting, newdata = x_valid, n.trees=5000, type = "response")

# Metricas de performance
metricas(y_hat_train, 0.03, y_train)
metricas(y_hat, 0.03, y_valid)

### Iteracion 2 - Random Search + Modelo base

## Random Grid Search
gbm_grid <- gbm_random_grid(size = 30,
                            min_ntree = 3000, max_ntree = 5000,
                            min_depth = 3, max_depth = 6,
                            min_shrinkage = 0.001, max_shrinkage = 0.01)

gbm_models <- train_gbm(modelTrain, gbm_grid, x_valid, y_valid, 0.03)

# Analisis de las salidas
res_table <- result_table(gbm_models)
print(res_table)

# Modelo optimo
g.boosting.optimal = gbm(RESULT ~ .,
                 data = modelTrain,
                 distribution="bernoulli",
                 n.trees=3186,
                 shrinkage=0.00654,
                 cv.folds=5,
                 interaction.depth=5)

summary(g.boosting.optimal)
best.iter = gbm.perf(g.boosting.optimal, method="cv")

# Prediccion
y_hat_train = predict(g.boosting.optimal, newdata = x_train, n.trees=3186, type = "response")
y_hat = predict(g.boosting.optimal, newdata = x_valid, n.trees=3186, type = "response")

# Metricas de performance
metricas(y_hat_train, 0.03, y_train)
metricas(y_hat, 0.03, y_valid)

### Iteracion 3 - SMOTE + Random Search + Modelo base
smotedTrain <- SMOTE(RESULT ~ ., data = modelTrain, perc.over = 200)
prop.table(table(modelTrain$RESULT))
prop.table(table(smotedTrain$RESULT))

smotedTrain$RESULT <- as.numeric(smotedTrain$RESULT)
smotedTrain = transform(smotedTrain, RESULT = RESULT-1)

gbm_grid <- gbm_random_grid(size = 3,
                            min_ntree = 3000, max_ntree = 5000,
                            min_depth = 3, max_depth = 6,
                            min_shrinkage = 0.001, max_shrinkage = 0.01)

gbm_models <- train_gbm(smotedTrain, gbm_grid, x_valid, y_valid, 0.03)

# Analisis de las salidas
res_table <- result_table(gbm_models)
print(res_table)

# Modelo optimo
g.boosting.optimal = gbm(RESULT ~ .,
                         data = smotedTrain,
                         distribution="bernoulli",
                         n.trees=3528,
                         shrinkage=0.00724,
                         cv.folds=5,
                         interaction.depth=5)

summary(g.boosting.optimal)
best.iter = gbm.perf(g.boosting.optimal, method="cv")

# Prediccion
y_hat_train = predict(g.boosting.optimal, newdata = x_train, n.trees=3528, type = "response")
y_hat = predict(g.boosting.optimal, newdata = x_valid, n.trees=3528, type = "response")

# Metricas de performance
metricas(y_hat_train, 0.1, y_train)
metricas(y_hat, 0.1, y_valid)

##------------------------------------------------------------------------------------##
## Prediccion Final                                                                   ##
##------------------------------------------------------------------------------------##

modelTest  <- dfTest[,varsSelected[1:length(varsSelected)-1]]

y_preds = predict(r.forest.optimal, modelTest, type = 'prob')[,2]

y_preds <- ifelse(y_preds > 0.9, 'FUNDED', 'NOT FUNDED') 

table(y_preds)

options(scipen = 999)  # Para evitar que se guarden valores en notacion científica
write.table(y_preds, file = "Predicciones.csv", sep = ",", row.names=TRUE, quote=FALSE)
