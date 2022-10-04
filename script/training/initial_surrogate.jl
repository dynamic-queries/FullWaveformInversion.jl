using NeuralOperators
using Flux
using FluxTraining
using BSON
using HDF5
include("../src/train.jl")

filename = "bigdata/static/BOUNDARY"
file = h5open(filename)

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

batches = 1:180
BS = length(batches)

xdata = Array{Float64,4}(undef,3,nx,ny,BS)
ydata = Array{Float64,4}(undef,1,nx,ny,BS)

for (ib,b) in enumerate(batches)
    xdata[1,:,:,ib] .= read(file["b$(b)"])
    xdata[2,:,:,ib] .= X
    xdata[3,:,:,ib] .= Y
    ydata[1,:,:,ib] .= read(file["$(b)"])
end 

print("Read Data ... \n")

train,test = Flux.splitobs((xdata,ydata),at=0.9)
train_loader = Flux.DataLoader(train,batchsize=2,shuffle=true)
test_loader = Flux.DataLoader(test,batchsize=1,shuffle=false)

DL = 64
nmodes = 12

model = Chain(
        Dense(3,DL),
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
        Dense(DL,DL),
        Dense(DL,1)
)

# Optimizer params
lossfunction = l₂loss
data = (train_loader,test_loader)

print("Training model... \n")

# Optimizer params
lossfunction = l₂loss
data = (train_loader,test_loader)
foldername = "weights/is/"

if !isdir(foldername)
    mkdir(foldername)
end 

print("Training model... \n")

model = gpu(model)
logger = TBLogger("script/logs/is/110_epochs/")

lr = 1e-2
nepochs = 10
opt = Flux.Adam(lr)
learn(model,lossfunction,data,opt,nepochs,foldername,logger)

lr = 1e-3
nepochs = 100
opt = Flux.Adam(lr)
learn(model,lossfunction,data,opt,nepochs,foldername,logger)