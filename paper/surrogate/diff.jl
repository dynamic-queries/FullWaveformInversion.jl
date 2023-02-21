using FullWaveformInversion
using Flux
using NeuralOperators
using BSON:@load
using Plots
using Optim
using LinearAlgebra
using HDF5
using CUDA
using FiniteDiff
using ForwardDiff
using AbstractDifferentiation

CUDA.allowscalar(false)

const FWI = FullWaveformInversion

# Load test_case
file = h5open("paper/surrogate/test_case")
defect = read(file["Defect"])
U_actual = read(file["U_actual"])


## This is an RBF constrained optimization
# Define RBF object
x = 0.0:0.005:2π
rbf = RBF(x)
k = rbf.coefficients

# Define a surrogate model
ntsteps = 70
nx = 200
ny = 200
boundary = BitMatrix(undef,nx+1,ny+1)
boundary .= 0
k = 200*k
FWI.spline2obstacle!(boundary,(Vector(x),rbf(k)))

model = SurrogateModel(ntsteps,nx,ny,boundary)

function forward_model(k)
    rbf.coefficients = k
    FWI.spline2obstacle!(boundary,(Vector(x),rbf(k)))
    Upred = evaluate(model,boundary,FWI.Cheap())
    Upred
end 

function loss(k)
    norm(forward_model(k)-U_actual)
end 

# Differentiating everything takes too long.
# I am going to break this code down an build it back up.