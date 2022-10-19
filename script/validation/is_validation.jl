using HDF5 
using NeuralOperators
using FullWaveformInversion
using Flux
using FullWaveformInversion:learn
using TensorBoardLogger

filename = "/tmp/ge96gak/consolidated_data/static/BOUNDARY"
file = h5open(filename,"r")

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

batches = 746:750
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
foldername = "weights/is/16/"
if isdir(foldername)
    files = readdir(foldername)
end 
filename = files[end]

@load "weights/is/16/Epoch_100_TrainLoss_0.014171972255736812" modeltemp
model = gpu(modeltemp)
ypredict = model(xdata|>gpu) |> cpu

# Validation
for i=1:5
    crack = heatmap(x,y,xdata[1,:,:,i],title="Defect");
    p1 = heatmap(x,y,ypredict[1,:,:,i],title="FNO model");
    p2 = heatmap(x,y,ydata[1,:,:,i],title="Original");
    error = heatmap(x,y,ypredict[1,:,:,i] .- ydata[1,:,:,i], title="Error");
    plot(crack,p1,p2,error,layout=4,legend=false);
    savefig("script/validation_images/is/16/$(i).png")
end