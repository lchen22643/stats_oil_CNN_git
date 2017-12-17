

setwd("C:/Users/User/Downloads/440")
train = readRDS(file='traindata.rds')

test = readRDS(file='testdata.rds')



install.packages("keras")
install.packages("tensorflow")
install.packages("devtools")
devtools::install_github("rstudio/reticulate")
devtools::install_github("rstudio/jsonlite")

library(reticulate)
library(keras)
library(tensorflow)

#dat = jsonlite::fromJSON("train.json")
#train=dat
x_train_1 = train$band_1
x_train_2 = train$band_2
y_train   = train$is_iceberg

y_train <- to_categorical(y_train, 2)

x_test_1 = test$band_1
x_test_2 = test$band_2
# reshape
x_train_1 = array_reshape(unlist(x_train_1), c(1604,75,75,1))
x_train_2 = array_reshape(unlist(x_train_2), c(1604,75,75,1))
x_train= abind(x_train_1, x_train_2, along = 4)
library(abind)

x_test_1 = array_reshape(unlist(x_test_1), c(8424,75,75,1))
x_test_2 = array_reshape(unlist(x_test_2), c(8424,75,75,1))
x_test = abind(x_test_1, x_test_2, along = 4)


#rescale
#??rescale
library(scales)
x_train = rescale(x_train, to = c(0, 1), from = range(x_train, na.rm = TRUE, finite = TRUE))
x_train_1=rescale(x_train_1, to = c(0, 1), from = range(x_train_1, na.rm = TRUE, finite = TRUE))
x_train_2=rescale(x_train_2, to = c(0, 1), from = range(x_train_2, na.rm = TRUE, finite = TRUE))
x_test_1=rescale(x_test_1, to = c(0, 1), from = range(x_test_1, na.rm = TRUE, finite = TRUE))
x_test_2=rescale(x_test_2, to = c(0, 1), from = range(x_test_2, na.rm = TRUE, finite = TRUE))
x_test=rescale(x_test, to = c(0, 1), from = range(x_test, na.rm = TRUE, finite = TRUE))
image(x_train_1[1,,,1])

xtrain  = array(NA, dim=c(1604,75,75,2))
for (i in 1:1604){
  #x = x_train[i,,,1]
  #y = x_train[i,,,2]
  #xtrain[i,,,1] = ( x - min(x)) / (max(x) - min(x) ) 
  #xtrain[i,,,2] = ( y - min(y)) / (max(y) - min(y) ) 
  fft1 = fft(x_train[i,,,1])
  fft2 = fft(x_train[i,,,2])
  xtrain[i,,,1] = abs(fft1)
  xtrain[i,,,2] = abs(fft2)
  #   xtrain[i,,,5] = Re(fft1)
  #   xtrain[i,,,6] = Re(fft2)
  #   xtrain[i,,,7] = Im(fft1)
  #   xtrain[i,,,8] = Im(fft2)
}

#x_train= xtrain
  #abind(x_train, xtrain, along = 4)
###############test data combine




xtest  = array(NA, dim=c(8424,75,75,2))
for (i in 1:8424){
  #x = x_train[i,,,1]
  #y = x_train[i,,,2]
  #xtrain[i,,,1] = ( x - min(x)) / (max(x) - min(x) ) 
  #xtrain[i,,,2] = ( y - min(y)) / (max(y) - min(y) ) 
  fft1 = fft(x_test[i,,,1])
  fft2 = fft(x_test[i,,,2])
  xtest[i,,,1] = abs(fft1)
  xtest[i,,,2] = abs(fft2)
  #   xtrain[i,,,5] = Re(fft1)
  #   xtrain[i,,,6] = Re(fft2)
  #   xtrain[i,,,7] = Im(fft1)
  #   xtrain[i,,,8] = Im(fft2)
}

#x_test= xtest#abind(x_test, xtest, along = 4)
input_shape = c(75,75,2)
model <- keras_model_sequential()
model%>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(6,6), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(5,5), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  
  #layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 36, activation = 'relu') %>% 
  #layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dense(units = 2, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)
summary(model)
history <- model %>% fit(
  x_train, y_train, 
  epochs =50, batch_size = 128, 
  validation_split = 0.2
)

plot(history)

#loss_and_metrics <- model %>% evaluate(x_train_1, y_train)
#loss_and_metrics
#??keras_predict
#pred <- keras_predict(mod, normalize(X_test))
#classes <- model %>% predict_classes(x_train_1)
#classes

probabilities  <- model %>% predict_proba(x_test) 
prb = as.numeric(probabilities[,2])
probabilities[,1]
test[,1]
star = data.frame(id = test[,1], is_iceberg = prb )

colnames(star)=c("id","is_iceberg")

#write.csv(star,'original100.csv',row.names=FALSE)
#write.csv(star,'original2times50.csv',row.names=FALSE)
#write.csv(star,'original3times50.csv',row.names=FALSE)
write.csv(star,'original5times50.csv',row.names=FALSE)
#write.csv(star,'original250.csv',row.names=FALSE)
#write.csv(star,'original500.csv',row.names=FALSE)





