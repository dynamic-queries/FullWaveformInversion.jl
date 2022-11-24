using FullWaveformInversion
using FullWaveformInversion:TwoD
using LinearAlgebra
using Plots
using BSON
using NeuralOperators
using HDF5
using Flux


# Full Order solution

nx = 200
ny = 200

# Load Surrogate
x = 0.0:(1.0)/nx:1.0
y = 0.0:(1.0)/ny:1.0

X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

nsteps = 39
nsensors = 10
boundary = BitMatrix(undef,nx+1,ny+1)
boundary .= 0
nxmid = 101
nymid = 101
radius = 40
for i=1:nx
    for j=1:ny
        if sqrt((i-nxmid)^2 + (j-nymid)^2) <= radius 
            boundary[i,j] = 1
        end 
    end 
end 
heatmap(boundary,title="Circular Boundary")
savefig("figures/circle_boundary.png")

_,_,u = FullWaveformInversion.solver(nx,ny,TwoD(),boundary)

anim = @animate for ts=1:nsteps
    heatmap(u[:,:,ts],title="$(ts) time step")
end 
gif(anim,"figures/circle.gif",fps=10)



batches = 1:1
BS = length(batches)
nx = length(x)
ny = length(y)

xdata = Array{Float64,4}(undef,3,nx,ny,BS)
ydata = Array{Float64,4}(undef,1,nx,ny,nsteps)

xdata[1,:,:,1] = float(boundary)
xdata[2,:,:,1] = X
xdata[3,:,:,1] = Y

BSON.@load "best/is/is_6l_ 16d" modeltemp
surA = modeltemp
uinit = surA(xdata)
heatmap(uinit[1,:,:,1]) 

for ts = 1:nsteps
    ydata[1,:,:,ts] .= u[:,:,ts]
end 

ysur = zero(ydata)
for k = 1:nsteps
    sur = FullWaveformInversion.surrogate(k)
    ysur[1,:,:,k] = sur(xdata)[1,:,:,1]
end

anim = @animate for ts=1:nsteps
    heatmap(ysur[1,:,:,ts],title="$(ts) time step")
end 
gif(anim,"figures/circle_surrogate.gif",fps=10)