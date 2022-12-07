using FullWaveformInversion
using Flux
using NeuralOperators
using BSON
using Plots

# Test script for no boundary
nx = 200
ny = 200
boundary = BitMatrix(undef,nx+1,ny+1)
boundary .= 0 
boundary_name = "experiments/empty"
# FullWaveformInversion.standalone_test(boundary,boundary_name)

# Load Surrogate
x = 0.0:(1.0)/nx:1.0
y = 0.0:(1.0)/ny:1.0

X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

if isnothing(boundary)
    _,boundary,u = FullWaveformInversion.solver(nx,ny,TwoD())
else
    _,_,u = FullWaveformInversion.solver(nx,ny,TwoD(),boundary)
end 

f1 = heatmap(boundary,title=boundary_name)

anim = @animate for ts=1:nsteps
    heatmap(u[:,:,ts],title="$(ts) time step")
end 
f2 = gif(anim,"figures/$(boundary_name).gif",fps=10)

batches = 1:1
BS = length(batches)
nx = length(x)
ny = length(y)

xdata = Array{Float64,4}(undef,3,nx,ny,BS)

xdata[1,:,:,1] = float(boundary)
xdata[2,:,:,1] = X
xdata[3,:,:,1] = Y 

BSON.@load "best/is/is_6l_ 16d" modeltemp
surA = modeltemp
uinit = surA(xdata)
heatmap(uinit[1,:,:,1],title="Initial Surrogate") 
