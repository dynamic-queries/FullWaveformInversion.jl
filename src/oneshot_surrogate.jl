include("../src/data-gen.jl")
using NeuralOperators
using Flux
using Flux:splitobs 
using FluxTraining
using BSON: @load

# We know apriori that the problem was solved in the domain.
# x ∈ [0,1]
# y ∈ [0,1]
# With a resolution of 199 internal points.

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0

X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

## Data
filename = "./data/GROUND_TRUTH"

batches = 1:80
BS = length(batches)
nx = length(x)
ny = length(y)

xdata = Array{Float64,4}(undef,3,nx,ny,BS)
ydata = Array{Float64,4}(undef,1,nx,ny,BS)

for (i,b) in enumerate(batches)
    file = h5open(string(filename,b),"r")
    xdata[1,:,:,i] .= read(file["1"])
    xdata[2,:,:,i] .= X 
    xdata[3,:,:,i] .= Y
    ydata[1,:,:,i] .= read(file["100"])
end     

print("Read Data...\n")

# DataLoader
train,test = splitobs((xdata,ydata),at=0.9)
train_loader = Flux.DataLoader(train,batchsize=20,shuffle=true)
test_loader = Flux.DataLoader(test,batchsize=1,shuffle=false)

# Model
nmodes = 8
DL = 16
 
# model = Chain(
#         Dense(3,DL),
#         OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
#         OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
#         OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
#         OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
#         Dense(DL,DL),
#         Dense(DL,1)
# )


@load "src/os_checkpoints/checkpoint_epoch_139_loss_0.10179136244195063.bson" model

# Optimizer params
lossfunction = l₂loss
data = (train_loader,test_loader)

print("Training model... \n")

# learning_rate=1e-2
# nepochs = 30
# opt = Flux.ADAM(learning_rate)
# learner = Learner(model,data,opt,lossfunction,Checkpointer(joinpath(@__DIR__,"os_checkpoints")))
# fit!(learner,nepochs)

learning_rate=1e-3
nepochs = 100
opt = Flux.ADAM(learning_rate)
learner = Learner(model,data,opt,lossfunction,Checkpointer(joinpath(@__DIR__,"os_checkpoints")))
fit!(learner,nepochs)