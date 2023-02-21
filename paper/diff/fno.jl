using HDF5
using BSON:@load
using Flux
using NeuralOperators
using CUDA
CUDA.allowscalar(false)
using FullWaveformInversion
using Zygote

using AbstractDifferentiation

nx = 100
ny = 100

x = 0.0:(1.0)/nx:1.0
y = 0.0:(1.0)/ny:1.0
X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

# Sensor locations
xsensors = Vector(5:floor(Int,(length(x)-5)/9):length(x))
ysensors = 3*ones(Int,length(xsensors))

## This is an RBF constrained optimization
# Define RBF object
x = 0.0:0.005:2π
rbf = RBF(x)
k = rbf.coefficients

boundary = BitMatrix(undef,nx+1,ny+1)
boundary .= 0
k = 200*k
spline2obstacle!(boundary,(Vector(x),rbf(k)))


input = CUDA.zeros(3,nx+1,ny+1,1)
input[1,:,:,1] = boundary
input[2,:,:,1] = X
input[3,:,:,1] = Y
inter = zeros(nx+1,ny+1)
output = CUDA.zeros(Float64,1,nx+1,ny+1,1)

SurBs = []
filename = "best/os/"
ntsteps = 70
for i=1:ntsteps
    @load string(filename,"$(i)") modeltemp
    push!(SurBs,modeltemp |> gpu)
end 

# Initial Surrogate
@load "best/is/is_smooth" modeltemp
SurA = modeltemp |> gpu

function eval_fno(input::CuArray)
    return SurA(input)
end 

ab = AD.Zygote
D_eval_fno = ab.jacobian(eval_fno,input)