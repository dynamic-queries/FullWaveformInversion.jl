using Flux
using Flux:splitobs 
using BSON: @load
using HDF5
using NeuralOperators
using Plots
using CUDA

# We know apriori that the problem was solved in the domain.
# x ∈ [0,1]
# y ∈ [0,1]
# With a resolution of 199 internal points.

k = 150

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0

X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

## Data
filename = "big_data/dynamic/GROUND_TRUTH"

batches = 181:185
BS = length(batches)
nx = length(x)
ny = length(y)

xdata = CuArray{Float64,4}(undef,3,nx,ny,BS)
ydata = CuArray{Float64,4}(undef,1,nx,ny,BS)

for (i,b) in enumerate(batches)
    file = h5open(string(filename,b),"r")
    xdata[1,:,:,i] .= read(file["$(k)"])
    xdata[2,:,:,i] .= X 
    xdata[3,:,:,i] .= Y
    ydata[1,:,:,i] .= read(file["$(k)"])
end     

dir = "weights/os/$(k)/"
filename = readdir(dir)[end]
@load string(dir,filename) model 

@show model typeof(model)

# ypredict = model(xdata)

# # Validation
# for i=1:20
#     p1 = heatmap(x,y,ypredict[1,:,:,i],title="FNO model");
#     p2 = heatmap(x,y,ydata[1,:,:,i],title="Original");
#     plot(p1,p2,layout=(2,1),legend=false);
#     foldername = "script/validation_images/$(k)/"
#     if !isdir(foldername)
#         mkdir(foldername)
#     end 
#     savefig(string(foldername,"$(i).png"))
# end 