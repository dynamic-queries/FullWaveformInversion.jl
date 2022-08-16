using NeuralOperators
using BSON
using BSON : @load
include("src/utils.jl")

# Filenames of models
isfilename = "src/is_checkpoints/checkpoint_epoch_030_loss_0.04952367390419685.bson"
osfilename = "src/os_checkpoints/checkpoint_epoch_035_loss_0.08063310637318366.bson"

# Load models
@load inputsurrogate isfilename
@load outputsurrogate osfilename

# Input grid
nx = 200
ny = 200
x = 0.0:(1.0/nx):1.0
y = 0.0:(1.0/ny):1.0
y = 0.0:(1.0)/20:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

# Initial condition
s = spline()
A = BitMatrix(undef,nx+1,ny+1)
A .= 0
spline2obstacle!(A,s)

# Munge data
xdata = Array{Float64,3}(undef,3,nx,ny)
xdata[1,:,:] .= A 
xdata[2,:,:] .= X
xdata[3,:,:] .= Y

# Predict output
ypredict = outputsurrogate(inputsurrogate(xdata))

# Visualize the input and the outputs
fig1 = heatmap(A,title="Input obstacle field.")
fig2 = heatmap(ypredict[1,:,:],title="Displacement/Pressure field at t=1ms")
plot(fig1,fig2,layout=(2,1))
savefig("./figures/comparison.png")


# If this works well, make this a function.
# Create a simulation parameters object.
# Put all the non-hard coded simulation paramters in that struct. 
# The query function should look like : 
# function query(simulation::Simulation,A::Initial Obstacle Field.)
