
function surrogate(timestep::Int)
    # initial surrogate
    @load "best/is/is_6l_ 16d" modeltemp
    SurA = modeltemp
    
    # surrogate corresponding to "timestep"
    @load "best/os/$(timestep)" modeltemp
    SurB = modeltemp

    # compose the two
    return x -> SurB(SurA(x))
end 

function sample(Y::Array, sensorlocs::Vector)
    Z = zeros(length(sensorlocs))
    k = 1
    for coords in sensorlocs
        x,y = coords
        Z[k] = Y[x,y]
    end 
    Z
end 

function timeseries!(nsteps::Int,initial::Array, final::Array, TS::Matrix, sensorlocs::Vector)
    # We train a maximum of 100 surrogates for the first 100 time steps.
    for j=1:nsteps
        sur = surrogate(j)
        final .= sur(initial)
        TS[:,j] .= sample(final,sensorlocs)
    end 
end 



using HDF5 
using NeuralOperators
using FullWaveformInversion
using Flux
using FullWaveformInversion:learn
using TensorBoardLogger
using BSON:@load
using Plots

# We know apriori that the problem was solved in the domain.
# x ∈ [0,1]
# y ∈ [0,1]
# With a resolution of 199 internal points.

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0

X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

## Data
filename = "data/dynamic/GROUND_TRUTH"

batches = 1:1
BS = length(batches)
nx = length(x)
ny = length(y)

xdata = Array{Float64,4}(undef,3,nx,ny,BS)
ydata = Array{Float64,4}(undef,1,nx,ny,BS)

for (i,b) in enumerate(batches)
    file = h5open(string(filename,b),"r")
    xdata[1,:,:,i] .= read(file["$(k)"])
    xdata[2,:,:,i] .= X 
    xdata[3,:,:,i] .= Y
    ydata[1,:,:,i] .= read(file["$(k)"])
end

# IN datagenerated from the surrogate
sol = zero(ydata)
ntsteps =39
nsensors = 10
sensor_range = 0.0:(1.0)/(nsensors-1):1.0
coords = coordinates()
T = zeros(nsensors,nsteps)
timeseries!(nsteps,xdata,sol,T,coords)