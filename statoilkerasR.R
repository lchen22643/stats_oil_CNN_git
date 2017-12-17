library(kerasR)
mode = kerasR::Sequential()
reticulate::py_available()
reticulate::import("keras.models")


pixels = 
