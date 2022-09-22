using MLDatasets
using Flux
using FluxTraining

# load data
train,test = MNIST.traindata(),MNIST.testdata()