include("../src/data-gen.jl")
using NeuralOperators
using Flux
using Flux:splitobs 
using FluxTraining

# We know apriori that the problem was solved in the domain.
# x ∈ [0,1]
# y ∈ [0,1]
# With a resolution of 199 internal points.

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0

X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))


# read the input data.
p = 1:10
b = [1,2,4,5,6,7,9,11,12,13]
BS = 10
nx = length(x)
ny = length(y)
np = length(p)
filename = "/home/dynamic-queries/.julia/dev/InverseProblems.jl/data/GROUND_TRUTH"
xdata = Array{Float64,4}(undef,BS,nx,ny,np+2)
ydata = Array{Float64,4}(undef,BS,nx,ny,1)

for (i,label) in enumerate(b)
    file = h5open(string(filename,label),"r")
    for (j,t) ∈ enumerate(p)
        xdata[i,:,:,j] .= read(file["$(t)"])      
    end 
    xdata[i,:,:,end-1] .= X
    xdata[i,:,:,end] .= Y
    e = Int(p[end]+1)
    ydata[i,:,:,1] .= read(file["$(e)"])

end 


xdata = permutedims(xdata,(4,2,3,1))
ydata = permutedims(ydata,(4,2,3,1))

# DataLoader
train,test = splitobs((xdata,ydata),at=0.9)
train_loader = Flux.DataLoader(train,batchsize=30,shuffle=true)
test_loader = Flux.DataLoader(test,batchsize=3,shuffle=false)

# Model
nmodes = 16
DL = 20
model = Chain(
        Dense(np+2,DL),
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
        Dense(DL,DL),
        Dense(DL,1)
)


# Optimizer params
learning_rate=1e-3
nepochs = 100
opt = Flux.ADAM(learning_rate)
loss(x,y) = l₂loss
data = (train_loader,test_loader)

learner = Learner(model,data,opt,loss,Checkpointer(joinpath(@__DIR__,"checkpoints")))
fit!(learner,100)