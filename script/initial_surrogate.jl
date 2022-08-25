using NeuralOperators
using Flux
using FluxTraining
using BSON
using HDF5

filename = "data/static/BOUNDARY"
file = h5open(filename)

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

batches = 1:80
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
train_loader = Flux.DataLoader(train,batchsize=20,shuffle=true)
test_loader = Flux.DataLoader(test,batchsize=2,shuffle=false)


DL = 16
nmodes = 8

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

learning_rate=1e-2
nepochs = 60
opt = Flux.ADAM(learning_rate)
learner = Learner(model,data,opt,lossfunction,Checkpointer(joinpath(@__DIR__,"is_checkpoints")))
fit!(learner,nepochs)

learning_rate=1e-3
nepochs = 100
opt = Flux.ADAM(learning_rate)
learner = Learner(model,data,opt,lossfunction,Checkpointer(joinpath(@__DIR__,"is_checkpoints")))
fit!(learner,nepochs)