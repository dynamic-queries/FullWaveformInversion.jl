using NeuralOperators
using Flux
using Plots
using FluxTraining
using BSON:@load
using HDF5

filename = "big_data/static/BOUNDARY"
file = h5open(filename)

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

batches = 181:200
BS = length(batches)

xdata = Array{Float64,4}(undef,3,nx,ny,BS)
ydata = Array{Float64,4}(undef,1,nx,ny,BS)

for (ib,b) in enumerate(batches)
    xdata[1,:,:,ib] .= read(file["b$(b)"])
    xdata[2,:,:,ib] .= X
    xdata[3,:,:,ib] .= Y
    ydata[1,:,:,ib] .= read(file["$(b)"])
end 

print("Read Data ... \n")

@load "script/is_checkpoints/16/0.01/checkpoint_epoch_075_loss_0.05436417911179097.bson" model 

ypredict = model(xdata)

# Validation
for i=1:20
    crack = heatmap(x,y,xdata[1,:,:,i],title="Defect");
    p1 = heatmap(x,y,ypredict[1,:,:,i],title="FNO model");
    p2 = heatmap(x,y,ydata[1,:,:,i],title="Original");
    error = heatmap(x,y,ypredict[1,:,:,i] .- ydata[1,:,:,i], title="Error");
    plot(crack,p1,p2,error,layout=4,legend=false);
    savefig("validation/initial/16/0.01/$(i).png")
end 