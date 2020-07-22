rm(list=ls())
gc()
.rs.restartR()
options(scipen = 999)

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Carga de librerias                                                                 ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

library(data.table)
library(dplyr)
library(ggplot2)
library(reshape2)
library(Matrix)
library(lubridate)
library(gtools)

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Carga de funciones                                                                 ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

# Cargamos las funciones a utilizar, las cuales fueron definidas en un script separado.

setwd('C:/Repo/Github/machine_learning/ctr_prediction/')
source('utils.R')

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Carga de datasets                                                                  ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

data_dir <- "C:/Repo/Github/machine_learning/ctr_prediction/ctr_data/"


### Carga de un 30% de los datos de cada dia
df <- load_train_data(data_dir, 0.3)

format(object.size(df0), unit = "Mb")

attach(df)

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Tratamiento de nulos                                                               ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

# Proporcion de NAs por columna
na_columnas <- data.frame('pct_nulos'= colMeans(is.na(df)))
na_columnas$variable <- rownames(na_columnas)

# Eliminacion de columnas con mas del 50% de nulos
rm_columnas <- na_columnas[na_columnas$pct_nulos > 0.5,]

print(paste('Cantidad de columnas eliminadas:', length(rm_columnas$variable)))
print(rm_columnas$variable)

df <- select(df, -c(rm_columnas$variable))

## Se eliminan las columnas que tengan mas del 50% de nulos, ya que nos resulta poco apropiado imputar
## una proporcion tan grandes de valores faltantes.

# Proporcion de NAs por fila
df$na_count <- apply(is.na(df), 1, sum)

max(df$na_count)/length(df) #Maximo % de NAs en las filas

# El maximo porcentaje de nulos que tienen las filas es un 30%, por lo que no se considera que sea suficiente para
# prescindir de la informacion que aportan el resto de las variables de estas filas.

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Feature Engineering                                                                ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

# Muchas de las variables creadas en este apartado son generadas a partir de analisis realizados en los EDAs.
# Se opto por separar la creacion de las variables del analisis para mejor orden.

## Creacion de variables

# Label como factor para algunos graficos
df$Label_factor <- as.factor(df$Label)

# Transformacion logaritmica del Bidfloor para reducir el efecto de los outliers
df$log_auction_bidfloor <- log(df$auction_bidfloor)

# Tamano del banner
df$tamano_banner <- as.factor(paste(df$creative_width,df$creative_height))
df$tamano_banner_num <- df$creative_width*df$creative_height

## Conversion de Fechas
df$auction_time_offset <- auction_time+timezone_offset
df$auction_datetime_offset <- as_datetime(df$auction_time_offset)
df$auction_weekday_offset <- factor(strftime(df$auction_datetime_offset, format = "%A"), levels = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
df$auction_fin_semana_offset <- factor(ifelse(df$auction_weekday_offset %in% c("Friday", "Saturday", "Sunday"),1,0))
df$auction_hora_offset <- strftime(df$auction_datetime_offset, format = "%H")


df$auction_datetime <- as_datetime(df$auction_time)
df$auction_weekday <- factor(strftime(df$auction_datetime, format = "%A"), levels = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
df$auction_fin_semana <- factor(ifelse(df$auction_weekday %in% c("Friday", "Saturday", "Sunday"),1,0))
df$auction_hora <- strftime(df$auction_datetime, format = "%H")
df$auction_turno <- factor(ifelse(as.numeric(df$auction_hora)<7,'Noche 2',
                                 ifelse(as.numeric(df$auction_hora)<14,'Manana',
                                        ifelse(as.numeric(df$auction_hora)<20,'Tarde', 'Noche 1'))),
                          levels = c('Manana','Tarde','Noche 1','Noche 2'))

## Creacion de categorica concatenando decive_type y has_video
df$device_video <- as.factor(paste(device_id_type,has_video))

####### BIN COUNTING #######
## Nos quedamos con un 10% de los datos para calcular el CTR por dia como una probabilidad a priori y 
## utilizarla como variable.

indexes <- which(!is.na(df[, "Label"])) # Excluimos los indices que corresponden al validation set
bin_counting_index <- sample(indexes, 0.1 * length(indexes))  # Seleccionamos un 10% al azar para realizar bin counting

df_bin_counting <- df[bin_counting_index,] # Nos quedamos con los registros a partir de los cuales realizaremos el bin counting
df_model <- df[-bin_counting_index,] # Filtramos los registros de bin counting del dataframe para modelar, para evitar data leakage

###### CATEGORICAS #####

######## ACTION ########

# Analisis por action_list_1
act_list_1 <- df_bin_counting %>% group_by(action_list_1) %>% summarize(CTR_act_list_1 = mean(Label)*100)
head(arrange(act_list_1,desc(act_list_1$CTR_act_list_1)),30)
summary(act_list_1$CTR_act_list_1)

# Analisis por action_list_2
act_list_2 <- df_bin_counting %>% group_by(action_list_2) %>% summarize(CTR_act_list_2 = mean(Label)*100)
head(arrange(act_list_2,desc(act_list_2$CTR_act_list_2)),30)
summary(act_list_2$CTR_act_list_2)

# Analisis por action_categorical_1
act_categorical_1 <- df_bin_counting %>% group_by(action_categorical_0,action_categorical_1) %>% summarize(CTR_act_cat_1 = mean(Label)*100)
head(arrange(act_categorical_1,desc(act_categorical_1$CTR_act_cat_1)),30)
summary(act_categorical_1$CTR_act_cat_1)

# Analisis por action_categorical_2
act_categorical_2 <- df_bin_counting %>% group_by(action_categorical_0,action_categorical_1,action_categorical_2) %>% summarize(CTR_act_cat_2 = mean(Label)*100)
head(arrange(act_categorical_2,desc(act_categorical_2$CTR_act_cat_2)),30)
summary(act_categorical_2$CTR_act_cat_2)

# Analisis por action_categorical_3
act_categorical_3 <- df_bin_counting %>% group_by(action_categorical_0,action_categorical_1,action_categorical_2,action_categorical_3) %>% summarize(CTR_act_cat_3 = mean(Label)*100)
head(arrange(act_categorical_3,desc(act_categorical_3$CTR_act_cat_3)),30)
summary(act_categorical_3$CTR_act_cat_3)

# Analisis por action_categorical_4
act_categorical_4 <- df_bin_counting %>% group_by(action_categorical_0,action_categorical_1,action_categorical_2,action_categorical_3,action_categorical_4) %>% summarize(CTR_act_cat_4 = mean(Label)*100)
head(arrange(act_categorical_4,desc(act_categorical_4$CTR_act_cat_4)),30)
summary(act_categorical_4$CTR_act_cat_4)

## Merge de ACTION variables con DF principal

df_model <- merge(df_model, act_list_1, by = c("action_list_1"), all.x = TRUE)
df_model <- merge(df_model, act_list_2, by = c("action_list_2"), all.x = TRUE)
df_model <- merge(df_model, act_categorical_1, by = c("action_categorical_0","action_categorical_1"), all.x = TRUE)
df_model <- merge(df_model, act_categorical_2, by = c("action_categorical_0","action_categorical_1","action_categorical_2"), all.x = TRUE)
df_model <- merge(df_model, act_categorical_3, by = c("action_categorical_0","action_categorical_1","action_categorical_2","action_categorical_3"), all.x = TRUE)
df_model <- merge(df_model, act_categorical_4, by = c("action_categorical_0","action_categorical_1","action_categorical_2","action_categorical_3","action_categorical_4"), all.x = TRUE)

rm(act_list_1)
rm(act_list_2)
rm(act_categorical_1)
rm(act_categorical_2)
rm(act_categorical_3)
rm(act_categorical_4)

head(df_model)

######## AUCTION ########

# Analisis por auction_categorical_0
auction_cat_0 <- df_bin_counting %>% group_by(auction_categorical_0) %>% summarize(CTR_auct_cat_0 = mean(Label)*100)
head(arrange(auction_cat_0,desc(auction_cat_0$CTR_auct_cat_0)),30)

# Analisis por auction_categorical_3
auction_cat_3 <- df_bin_counting %>% group_by(auction_categorical_3) %>% summarize(CTR_auct_cat_3 = mean(Label)*100)
head(arrange(auction_cat_3,desc(auction_cat_3$CTR_auct_cat_3)),30)

# Analisis por auction_categorical_6
auction_cat_6 <- df_bin_counting %>% group_by(auction_categorical_6) %>% summarize(CTR_auct_cat_6 = mean(Label)*100)
head(arrange(auction_cat_6,desc(auction_cat_6$CTR_auct_cat_6)),30)

# Analisis por auction_categorical_7
auction_cat_7 <- df_bin_counting %>% group_by(auction_categorical_7) %>% summarize(CTR_auct_cat_7 = mean(Label)*100)
head(arrange(auction_cat_7,desc(auction_cat_7$CTR_auct_cat_7)),30)

# Analisis por auction_categorical_9
auction_cat_9 <- df_bin_counting %>% group_by(auction_categorical_9) %>% summarize(CTR_auct_cat_9 = mean(Label)*100)
head(arrange(auction_cat_9,desc(auction_cat_9$CTR_auct_cat_9)),30)

# Analisis por auction_categorical_10
auction_cat_10 <- df_bin_counting %>% group_by(auction_categorical_10) %>% summarize(CTR_auct_cat_10 = mean(Label)*100)
head(arrange(auction_cat_10,desc(auction_cat_10$CTR_auct_cat_10)),30)

# Analisis por auction_categorical_12
auction_cat_12 <- df_bin_counting %>% group_by(auction_categorical_12) %>% summarize(CTR_auct_cat_12 = mean(Label)*100)
head(arrange(auction_cat_12,desc(auction_cat_12$CTR_auct_cat_12)),30)

## Merge de AUCTION variables con DF principal

df_model <- merge(df_model, auction_cat_0, by = c("auction_categorical_0"), all.x = TRUE)
df_model <- merge(df_model, auction_cat_3, by = c("auction_categorical_3"), all.x = TRUE)
df_model <- merge(df_model, auction_cat_6, by = c("auction_categorical_6"), all.x = TRUE)
df_model <- merge(df_model, auction_cat_7, by = c("auction_categorical_7"), all.x = TRUE)
df_model <- merge(df_model, auction_cat_9, by = c("auction_categorical_9"), all.x = TRUE)
df_model <- merge(df_model, auction_cat_10, by = c("auction_categorical_10"), all.x = TRUE)
df_model <- merge(df_model, auction_cat_12, by = c("auction_categorical_12"), all.x = TRUE)

rm(auction_cat_0)
rm(auction_cat_3)
rm(auction_cat_6)
rm(auction_cat_7)
rm(auction_cat_9)
rm(auction_cat_10)
rm(auction_cat_12)

head(df_model)
######## CREATIVE ########

# Analisis por creative_categorical_10
creative_cat_10 <- df_bin_counting %>% group_by(creative_categorical_10) %>% summarize(CTR_cre_cat_10 = mean(Label)*100)
head(arrange(creative_cat_10,desc(creative_cat_10$CTR_cre_cat_10)),30)

df_model <- merge(df_model, creative_cat_10, by = c("creative_categorical_10"), all.x = TRUE)

rm(creative_cat_10)
rm(df_bin_counting)
rm(bin_counting_index)
rm(indexes)
rm(df)

#####

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Exploratory Data Analysis (EDA)                                                    ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

# Generacion de dataset para EDA
# Excluimos los registros del validation set
index <- which(!is.na(df[, "Label"]))
df <- df[index, ]

### Share del target
prop.table(table(df$Label))*100

######## Bidfloor ########

### Distribucion

# Se analiza la distribucion de la variable Bidfloor segun la conversion.

### Distribucion de Bidfloor por Label
ggplot(data = df[df$auction_bidfloor > 0]) + 
  geom_boxplot(mapping = aes(x = Label_factor, y = auction_bidfloor)) +
  ggtitle('Distribucion de Bidfloor por Label') +
  xlab('Label') + 
  ylab('Bidfloor')

# Se realiza una transformacion logaritmica al Bidfloor para reducir el efecto de los outliers.

### Distribucion de Bidfloor por Label (en escala logaritmica para reducir efecto de outliers)
ggplot(data = df[auction_bidfloor > 0]) + 
  geom_boxplot(mapping = aes(x = Label_factor, y = log(auction_bidfloor))) +
  ggtitle('Distribucion de Bidfloor por Label') +
  xlab('Label') + 
  ylab('Log(Bidfloor)')

### Tasa de conversion por decil de Bidfloor

df$decil_auction_bidfloor <- quantcut(df$auction_bidfloor, q=10, na.rm=TRUE)
deciles_bidfloor <- df %>% group_by(decil_auction_bidfloor) %>% summarise(prop = round(mean(Label)*100,2))

ggplot(data = deciles_bidfloor) + 
  geom_bar(stat='identity', mapping = aes(x = decil_auction_bidfloor, y = prop), position = 'dodge') +
  ggtitle('Tasa de Conversion por Decil de Bidfloor') +
  xlab('Bidfloor') + 
  ylab('CTR (%)')


######################################################################################################################
# Conclusion: Pareceria que Bidfloor es una variable relevante para el modelo, ya que la tasa de conversion varia
# para los distintos valores de Bidfloor. 
######################################################################################################################

######## Heigth/Width ########

### Scatterplot entre Width y Heigth con Tasa de Conversion
# DataFrame para graficar
heigth_width <- df %>% group_by(creative_height,creative_width) %>% summarise(prop = round(mean(Label)*100,2))

# Grafico de dispersion
# Con este scatterplot se pretende analizar si hay relación entre el ancho y alto del banner y la tasa de conversión.

symbols(x=heigth_width$creative_width,
        y=heigth_width$creative_height,
        circles=heigth_width$prop,inches=0.3,
        fg="white",bg="orange", 
        main="CTR segun Heigth y Width del banner", xlab="Width", ylab="Heigth")+
  text(heigth_width$creative_width,heigth_width$creative_height,
       heigth_width$prop, cex=0.8)

# Si bien no parece haber una relación lineal entre las 2 variables y la conversión, sí se ve que hay 3 pares (Width, Height) 
# que concentran las mayores tasas de conversión. Se analizara la tasa de conversión para cada par (Width, Height), 
# agregando también la división si el banner tiene video o no. 

# Tasa de Conversion segun Tamano de Banner y Tipo (con o sin video)
heigth_width_video <- df %>% group_by(type = paste(creative_width,creative_height),has_video) %>% summarise(prop = round(mean(Label)*100,2))

ggplot(data = heigth_width_video) + 
  geom_bar(stat='identity', mapping = aes(x = type, y = prop, fill = has_video), position = 'dodge') +
  coord_flip() +
  ggtitle('CTR por Tamaño y Tipo de Banner') +
  xlab('Tamaños Banner (Width-Heigth)') + 
  ylab('CTR (%)')

######################################################################################################################
# Conclusion: Pareceria que el tamano del banner incide en la tasa de conversion. Asimismo, agregar un video en el banner 
# mejora notablemente la conversión, por lo que probablemente sea una variable relevante para el modelo. 
# En general la combinación de las 3 variables parece tener incidencia en el CTR.
######################################################################################################################

######## Device Type ########
# DataFrame para graficar
devices <- df %>% group_by(device_id_type) %>% summarise(prop = round(mean(Label)*100,2))

# Distribucion CTR por Device Type 
ggplot(data = devices) + 
  geom_bar(stat='identity', mapping = aes(x = device_id_type, y = prop), position = 'dodge') +
  ggtitle('Distribucion de CTR por Device Type') +
  xlab('Device Type') + 
  ylab('CTR')

# El Device Type 42080e25 tiene un CTR ampliamente superior al resto de los dispositivos 
# Se analizara si en combinacion con otras variables se puede discretizar mejor grupos con alta concentracion de CTR

### Device Type con Has_Video
# DataFrame para graficar
devices_video <- df %>% group_by(device_id_type, has_video) %>% summarise(prop = round(mean(Label)*100,2))

# Distribucion CTR por Device Type y Has_Video
ggplot(data = devices_video) + 
  geom_bar(stat='identity', mapping = aes(x = device_id_type, y = prop), position = 'dodge') +
  facet_wrap(~ has_video, nrow = 1) +
  ggtitle('Distribucion de CTR por Device Type y Has_Video') +
  xlab('Device Type') + 
  ylab('CTR')

# Al igual que para el tamano del banner, agregar video mejora notablemente la conversion en todos los dispositivos

### Device Type con Tamano Banner
# DataFrame para graficar
devices_tamano_banner <- df %>% group_by(device_id_type, tamano_banner) %>% summarise(prop = round(mean(Label)*100,2))

# Distribucion CTR por Device Type y Tamano Banner
ggplot(data = devices_tamano_banner) + 
  geom_bar(stat='identity', mapping = aes(x = tamano_banner, y = prop), position = 'dodge') +
  coord_flip() +
  facet_wrap(~ device_id_type, nrow = 1) +
  ggtitle('Distribucion de CTR por Tamano Banner y Device Type') +
  xlab('Tamano Banner') + 
  ylab('CTR')

# Pareceria que el CTR para los distintos tamanos de banner varia segun el dispositivo que se utilice.

######################################################################################################################
# Conclusion: El CTR varia segun el tipo de dispositivo. Este comportamiento se acentua al agregar video en el banner.
# Ademas, el CTR varia notablemente dentro de cada dispositivo segun el tamano del banner.
######################################################################################################################

######## Fechas ########
##### Sin Offset
# DataFrame para graficar
weekday <- df %>% group_by(auction_weekday) %>% summarise(prop = round(mean(Label)*100,2))

# Distribucion CTR por Weekday
ggplot(data = weekday) + 
  geom_bar(stat='identity', mapping = aes(x = auction_weekday, y = prop), position = 'dodge') +
  ggtitle('Distribucion de CTR por Weekday') +
  xlab('Weekday') + 
  ylab('CTR')

# DataFrame para graficar
weekday_hora <- df %>% group_by(auction_weekday,auction_hora) %>% summarise(prop = round(mean(Label)*100,2))

# Distribucion CTR por Hora y Weekday
ggplot(data = weekday_hora) + 
  geom_line(mapping = aes(x = as.numeric(auction_hora), y = prop)) +
  facet_wrap(~ auction_weekday, nrow = 3) +
  ggtitle('CTR por Hora y Weekday') +
  xlab('Hora') + 
  ylab('CTR')

# Discretizacion de las horas del dia
weekday_hora$turno <- factor(ifelse(as.numeric(weekday_hora$auction_hora)<7,'Noche 2',
                             ifelse(as.numeric(weekday_hora$auction_hora)<14,'Manana',
                                    ifelse(as.numeric(weekday_hora$auction_hora)<20,'Tarde', 'Noche 1'))),
                             levels = c('Manana','Tarde','Noche 1','Noche 2'))


ggplot(data = weekday_hora) + 
  geom_bar(stat='identity', mapping = aes(x = turno, y = prop), position = 'dodge') +
  facet_wrap(~ auction_weekday, nrow = 3) +
  ggtitle('Distribucion de CTR por Weekday') +
  xlab('Weekday') + 
  ylab('CTR')

# Se ve que discretizando las horas del dia se podria encapsular valores altos de CTR dentro de cada dia.

##### Con Offset
# DataFrame para graficar
weekday_offset <- df %>% group_by(auction_weekday_offset) %>% summarise(prop = round(mean(Label)*100,2))

# Distribucion CTR por Weekday
ggplot(data = weekday_offset) + 
  geom_bar(stat='identity', mapping = aes(x = auction_weekday_offset, y = prop), position = 'dodge') +
  ggtitle('Distribucion de CTR por Weekday') +
  xlab('Weekday') + 
  ylab('CTR')

# DataFrame para graficar
weekday_hora_offset <- df %>% group_by(auction_weekday_offset, auction_hora_offset) %>% summarise(prop = round(mean(Label)*100,2))

# Distribucion CTR por Hora y Weekday
ggplot(data = weekday_hora_offset) + 
  geom_line(mapping = aes(x = as.numeric(auction_hora_offset), y = prop)) +
  facet_wrap(~ auction_weekday_offset, nrow = 3) +
  ggtitle('CTR por Hora y Weekday') +
  xlab('Hora') + 
  ylab('CTR')

######################################################################################################################
# Conclusion: El CTR parece aumentar el fin de semana. Al tener una sola semana de datos, no podemos saber si este
# comportamiento es asi todas las semanas, o si puede ser tendencia o alguna promocion esta semana especifica (como un
# Black Friday). En cuanto al offset, no parece aportar diferencias significativas incluirlo o no. Se discretizan las horas
# de cada dia para encapsular los valores altos de CTR de cada dia.
######################################################################################################################

######## Categoricas ########
# Se realizaron sucesivos grafico para cada variable categorica, analizando como cada categoria agrupaba el CTR

# DataFrame para graficar
categorica <- df %>% group_by(creative_categorical_11) %>% summarise(prop = round(mean(Label)*100,2))

# Distribucion CTR por Categoria
ggplot(data = categorica) + 
geom_bar(stat='identity', mapping = aes(x = creative_categorical_11, y = prop), position = 'dodge') +
ggtitle('Distribucion de CTR por Categorica') +
xlab('Categorias') + 
ylab('CTR')

# Distribucion CTR por Categoria y Subcategoria
ggplot(data = categorica) + 
geom_bar(stat='identity', mapping = aes(x = action_categorical_5, y = prop), position = 'dodge') +
facet_wrap(~ action_categorical_6, nrow = 3) +
ggtitle('Distribucion de CTR por Categorica') +
xlab('Categorias') + 
ylab('CTR')


######################################################################################################################
# Conclusion: Las variables categoricas que parecen separar el CTR en grupos significativos son:

# - action_categorical_0
# - action_categorical_5
# - action_categorical_6
# - action_list_0
# - auction_boolean_0
# - auction_boolean_1
# - auction_boolean_2
# - auction_categorical_1
# - auction_categorical_4 (poco impacto)
# - auction_categorical_5
# - auction_categorical_8
# - creative_categorical_1
# - creative_categorical_4
# - creative_categorical_8
# - creative_categorical_9
# - creative_categorical_10
# - creative_categorical_11

# Para el resto de las variables categoricas no se encontro que los distintos niveles separen significativamente grupos 
# con altas concentraciones de CTR, o bien, en el caso de subcategorias, la categoria superior discretizaba el mismo valor 
# de CTR pero con menos niveles.
######################################################################################################################

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Balanceo del dataset                                                               ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

df2 <- balanceo_dataframe(df)

df_sampling <- prop.table(table(df %>% select(weekday_turno)))
hola_df <- df[df$weekday_turno == "domingo Manana",]
list_samples <- rownames(df_sampling) 
df_resampled <- data.frame()

sample_size <- as.integer(sample_ratio * rows) 
dt <- dt[sample(.N, sample_size)] 

combinatoria = c("lunes manana","lunes tarde","lunes noche1","lunes noche2",
                 "martes manana","martes tarde","martes noche1","martes noche2",
                 "miércoles manana","miércoles tarde","miércoles noche1","miércoles noche2",
                 "jueves manana","jueves tarde","jueves noche1","jueves noche2",
                 "viernes manana","viernes tarde","viernes noche1","viernes noche2",
                 "sábado manana","sábado tarde","sábado noche1","sábado noche2",
                 "domingo manana","domingo tarde","domingo noche1","domingo noche2")
necesario = c(103487,	196583,	84529,	101953,
              143183,	138234,	84561,	129333,
              138285,	96176,	67920,	127025,
              85708,	103732,	60136,	98533,
              76402,	119042,	70163,	115079,
              134717,	133065,	76859,	133320,
              103678,	148119,	37529,	118256)

balanceo_df <- data.frame (cbind(combinatorias,necesario))

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Feature Selection                                                                  ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

## Seleccion de variables para modelar

VARS_TO_KEEP <- c('Label',
                  # Continuas
                  'log_auction_bidfloor', 
                  # Categoricas - Action
                  'action_categorical_0', 'action_categorical_5', 'action_categorical_6', 'action_list_0',
                  # Categoricas - Auction
                  'auction_boolean_0','auction_boolean_1', 'auction_boolean_2', 'auction_categorical_1',
                  'auction_categorical_4', 'auction_categorical_5', 'auction_categorical_8',
                  'auction_weekday', 'auction_turno',
                  # Categoricas - Creative
                  'creative_categorical_1', 'creative_categorical_4', 'creative_categorical_8',
                  'creative_categorical_9', 'creative_categorical_10', 'creative_categorical_11', 
                  # Categoricas - Otras
                  'tamano_banner', 'device_video', 'device_id_type')

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Separacion en Train / Validation / Test                                            ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

## Separacion del dataset en Train/Validate/Test

df_separar <- as.data.frame(df)

# Division aleatoria de Train/Test
ev_index <- which(is.na(df_separar[, "Label"]))
eval_data <- df_separar[ev_index, VARS_TO_KEEP]

tr_index <- which(!is.na(df_separar[, "Label"]))  # Con matrices ralas, es molesto elegir columnas por nombre
vd_index <- sample(tr_index, 0.3 * length(tr_index))  # Selecciono un 10% al azar como conjunto de validación

train <- df_separar[setdiff(tr_index, vd_index),VARS_TO_KEEP]
test <- df_separar[vd_index,VARS_TO_KEEP]

# Division Train (Domingo, Lunes, Martes, Miercoles) y Test (Jueves, Viernes y Sabado)
ev_index <- which(is.na(df_separar[, "Label"]))
eval_data <- df_separar[ev_index, VARS_TO_KEEP]

train <- df_separar[df_separar$auction_weekday %in% c('Monday','Tuesday','Wednesday','Thursday') & !is.na(df_separar$Label), VARS_TO_KEEP]
test <- df_separar[!is.na(df_separar$Label) & df_separar$auction_weekday %in% c('Friday','Saturday','Sunday'), VARS_TO_KEEP]

#rm(df)
rm(df_separar)

# One-Hot Encoding
df_one_hot_train <- one_hot_sparse(as.data.table(train)) 
df_one_hot_test <- one_hot_sparse(as.data.table(test))

rm(train)
rm(test)

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Modelado                                                                           ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

#### Log de resultados obtenidos

# ITERACION 1
#--------------------------------------------------------------------------------------#
#### Variables ####
# c('Label','log_auction_bidfloor','auction_bidfloor', 'has_video','tamano_banner','gender','device_id_type',
#  'auction_age', 'action_list_0','creative_categorical_1','creative_categorical_4',
#  'creative_categorical_10','creative_categorical_11')]

#### Train/Test ####
# Datos = 100%
# Train = Aleatorio
# Test = 10% del train

#### Modelo #####
#xgb.train(data=dtrain, objective = "binary:logistic", nrounds = 300, watchlist = watchlist, eval.metric = "auc",
#          max_leaf_nodes = 5, min_samples_split = 3, learning_rate = 0.1)

#### Score ####
# Train AUC: 0.84
# Test AUC: 0.84
# Kaggle: 0.79

####--------------------------------------------------------------------------------####

# ITERACION 2
#--------------------------------------------------------------------------------------#
#### Variables ####
# c('Label',
#      Continuas
#        'log_auction_bidfloor', 
#         Categoricas - Action
#        'action_categorical_0', 'action_categorical_5', 'action_categorical_6', 'action_list_0',
#         Categoricas - Auction
#        'auction_boolean_0','auction_boolean_1', 'auction_boolean_2', 'auction_categorical_1',
#        'auction_categorical_4', 'auction_categorical_5', 'auction_categorical_8',
#        'auction_weekday', 'auction_turno',
#         Categoricas - Creative
#        'creative_categorical_1', 'creative_categorical_4', 'creative_categorical_8',
#        'creative_categorical_9', 'creative_categorical_10', 'creative_categorical_11', 
#         Categoricas - Otras
#        'tamano_banner', 'device_video', 'device_id_type')

#### Train/Test ####
# Datos = 50%
# Train = Lunes, Martes, Miercoles, Jueves
# Test = Viernes, Sabado, Domingo

#### Modelo #####
#xgb.train(data=dtrain, objective = "binary:logistic", nrounds = 200, watchlist = watchlist, eval.metric = "auc",
#          learning_rate = 0.1)

#### Score ####
# Train AUC: 0.85
# Test AUC: 0.83
# Kaggle: 0.79

####--------------------------------------------------------------------------------####

# ITERACION 2
#--------------------------------------------------------------------------------------#
#### Variables ####
#c('Label',
#  # Continuas
#  'log_auction_bidfloor', 
#  # Categoricas - Action
#  'action_categorical_0', 'action_categorical_5', 'action_categorical_6', 'action_list_0',
#  # Categoricas - Auction
#  'auction_boolean_0','auction_boolean_1', 'auction_boolean_2', 'auction_categorical_1',
#  'auction_categorical_4', 'auction_categorical_5', 'auction_categorical_8',
#  'auction_weekday', 'auction_turno',
#  # Categoricas - Creative
#  'creative_categorical_1', 'creative_categorical_4', 'creative_categorical_8',
#  'creative_categorical_9', 'creative_categorical_10', 'creative_categorical_11', 
#  # Categoricas - Otras
#  'tamano_banner', 'device_video', 'device_id_type',
#  # CTR Bin Counting
#  'CTR_weekday', 'CTR_video', 'CTR_act_list_1', 'CTR_act_list_2', 'CTR_act_cat_1','CTR_act_cat_2', 
#  'CTR_act_cat_3', 'CTR_act_cat_4', 'CTR_auct_cat_0', 'CTR_auct_cat_3', 'CTR_auct_cat_6',
#  'CTR_auct_cat_7', 'CTR_auct_cat_9', 'CTR_auct_cat_10', 'CTR_auct_cat_12', 'CTR_cre_cat_10')

#### Train/Test ####
# Datos = 100%
# Train = Aleatorio
# Test = 10% del train

#### Modelo #####
# xgb.train(data=dtrain, nrounds = 150,
#          watchlist = watchlist,
#          objective = "binary:logistic", 
#          eval.metric = "auc",
#          max_depth = 8,
#          colsample_bytree = 0.75,
#          min_child_weight = 1.504,
#          subsample = 0.75,
#          learning_rate = 0.074,
#          print_every_n = 5)

#### Score ####
# Train AUC: 0.88
# Test AUC: 0.87
# Kaggle: 0.49

# Creemos que de alguna manera se filtro informacion al target, generando data leakage.

####--------------------------------------------------------------------------------####


#### Separacion (X_train, y_train) y (X_test, y_test)

library('xgboost')

dtrain <- xgb.DMatrix(data = df_one_hot_train[ , colnames(df_one_hot_train) != "Label"],
                      label = df_one_hot_train[ , colnames(df_one_hot_train) == "Label"])

dvalid <- xgb.DMatrix(data = df_one_hot_test[ , colnames(df_one_hot_test) != "Label"],
                      label = df_one_hot_test[ , colnames(df_one_hot_test) == "Label"])

rm(df_one_hot_train)
rm(df_one_hot_test)


## Entrenamiento
watchlist <- list(train=dtrain, test=dvalid)

vanilla_model <- xgb.train(data=dtrain, nrounds = 150,
                           watchlist = watchlist,
                           objective = "binary:logistic",  # Es la función objetivo para clasificación binaria
                           eval.metric = "auc",
                           max_depth = 8,
                           colsample_bytree = 0.75,
                           min_child_weight = 1.504,
                           subsample = 0.75,
                           learning_rate = 0.074,
                           print_every_n = 5)


## Feature Importance
head(xgb.importance(model=vanilla_model), 20)

feat_importance <- head(xgb.importance(model=vanilla_model), 25)

ggplot(data = feat_importance) + 
  geom_bar(stat='identity', mapping = aes(x = Feature, y = Gain)) +
  coord_flip() +
  ggtitle('XGBoost Feature Importance') +
  xlab('Features') + 
  ylab('Gain')

### Analisis de Resutados

pred_test <- predict(vanilla_model, newdata = dvalid, type="response")

ggplot(test, aes(x=pred_test, color=Label)) + geom_density()

# Matriz de Confusion

ctab.test <- table(Predicho=pred_test>0.5, Real=test$Label)
ctab.test

# Calculo de Precision y Recall

precision <- ctab.test[2,2]/sum(ctab.test[2,])
precision

recall <- ctab.test[2,2]/sum(ctab.test[,2])
recall

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Random Search                                                                      ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

rgrid <- random_grid(size = 30,
                     min_nrounds = 50, max_nrounds = 300,
                     min_max_depth = 2, max_max_depth = 12,
                     min_eta = 0.001, max_eta = 0.125,
                     min_gamma = 0, max_gamma = 1,
                     min_colsample_bytree = 0.5, max_colsample_bytree = 1,
                     min_min_child_weight = 0, max_min_child_weight = 2,
                     min_subsample = 0.5, max_subsample = 1)

predicted_models <- train_xgboost(dtrain, dvalid, rgrid)

# Analisis de las salidas
res_table <- result_table(predicted_models)
print(res_table)

##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##
## Prediccion final y Exportacion                                                     ##
##------------------------------------------------------------------------------------##
##------------------------------------------------------------------------------------##

df_one_hot_eval <- one_hot_sparse(as.data.table(eval_data))

preds <- predict(vanilla_model, df_one_hot_eval[, colnames(df_one_hot_eval) != "Label"])
ids <- df_one_hot_eval[, colnames(df_one_hot_eval) == "id",]

resultados <- data.frame(id = as.matrix(ids), pred = as.matrix(preds))

options(scipen = 999)  # Para evitar que se guarden valores en notacion científica
write.table(resultados, file = "predicciones_v8.csv", sep = ",", row.names=TRUE, quote=FALSE)
