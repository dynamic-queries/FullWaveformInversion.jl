using HDF5
using BSON:@load
using Flux
using NeuralOperators

# Load Surrogates
filename = "best/is/"
@load string(filename,"is_6l_ 16d") modeltemp
SurA = modeltemp
SurBs = []
filename = "best/os/"
ntsteps = 70
for i=1:ntsteps
    @load string(filename,"$(i)") modeltemp
    push!(SurBs,modeltemp)
end 

# Load precomputed initial conditions
test_set = 1905
nx = 200
ny = 200
boundary = nothing
x = 0.0:(1.0)/nx:1.0
y = 0.0:(1.0)/ny:1.0
X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

filename = "consolidated/"
file = h5open(string(filename,"static/BOUNDARY"),"r")
boundary = read(file["b$(test_set)"])
pboundary = read(file["b3"])

# Sensor locations
xsensors = Vector(5:20:196)
ysensors = 3*ones(Int,length(xsensors))