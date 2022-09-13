using Flux
include("./src/surrogate.jl")

using MLDatasets
using FluxTraining
using Flux:onehotbatch


# load training set
train_x, train_y = MNIST.traindata()

# load test set
test_x,  test_y  = MNIST.testdata()

# EDA
using Plots

# Preprocess the data
xtrain = float(Array(reshape(train_x,(:,size(train_x)[3]))))
xtest = float(Array(reshape(test_x,(:,size(test_x)[3]))))

# One hot y label 
ytrain = float(Matrix(onehotbatch(train_y,0:9)))
ytest = float(Matrix(onehotbatch(test_y,0:9)))


# Dataloade
# trainloader = Flux.DataLoader((xtrain,ytrain),batchsize=20,shuffle=true)
# testloader = Flux.DataLoader((xtest,ytest),batchsize=10,shuffle=false)

model = Chain(
                Conv((5,5), channels=>16, pad=(2,2), relu),
                Conv((5,5), channels=>32, pad=(2,2), relu),
                Conv((5,5), channels=>32, pad=(2,2), stride=2, relu),
                Conv((5,5), channels=>64, pad=(2,2),stride=2,  relu),
                Conv((5,5), channels=>64, pad=(2,2), stride=2, relu),
                Conv((5,5), channels=>128, pad=(2,2), stride=2, relu),
                flatten,
                Dense(4*128, 1)
)


# # # loss function 
function loss(noise, ground_truth, model)
    batch_size = size(prediction)[0]
    loss=sum(model(prediction)-model(ground_truth))/batch_size
    loss
end

# # opt 
opt = Flux.Adam(0.01)
nepochs = 100

# # learner 
data = [(xtrain,ytrain)]
params = Flux.params(model)
cb() = @show loss(xtrain,ytrain)

# train
Flux.@epochs 100 Flux.train!(loss,params,data,opt,cb=cb)

typeof(xtrain),typeof(ytrain)