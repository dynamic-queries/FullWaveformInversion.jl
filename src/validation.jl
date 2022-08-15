using Flux
using Flux:splitobs 
using BSON: @load
using HDF5
using NeuralOperators
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
filename = "./data/GROUND_TRUTH"

batches = 11:14
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

@load "src/os_checkpoints/checkpoint_epoch_139_loss_0.10179136244195063.bson" model 

ypredict = model(xdata)

# Validation
for i=1:4
    p1 = heatmap(x,y,ypredict[1,:,:,i],title="FNO model")
    p2 = heatmap(x,y,ydata[1,:,:,i],title="Original")
    display(plot(p1,p2,layout=(2,1),legend=false))
    savefig("validation/$(i).png")
end 