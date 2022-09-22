using BSON: @load
using NeuralOperators
using LinearAlgebra
using Plots
using Flux
using HDF5

dir = "script/ts_checkpoints/"
filename = readdir(dir)[end]
@load string(dir,filename) model

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

batches = 181:185
BS = length(batches)
slices = 1:100:500
bs = length(slices)

N = bs*BS

filename = "./big_data/dynamic/GROUND_TRUTH"

xdata = Array{Float32,4}(undef,5,nx,ny,N)
ydata = Array{Float32,4}(undef,1,nx,ny,N)

print("Reading Data ... \n\n")

for (nb,b) in enumerate(batches)
    file = h5open(string(filename,nb),"r")
    for (it,slice) in enumerate(slices)
        xdata[1,:,:,(nb-1)*bs + it] .= read(file["$(slice)"])
        xdata[2,:,:,(nb-1)*bs + it] .= read(file["$(slice+1)"])
        xdata[3,:,:,(nb-1)*bs + it] .= read(file["$(slice+2)"])
        xdata[end-1,:,:,(nb-1)*bs + it] .= X
        xdata[end,:,:,(nb-1)*bs + it] .= Y
        ydata[1,:,:,(nb-1)*bs + it] .= read(file["$(slice+3)"]) 
    end 
end 

# Prediction
for j = 1:BS
    for i=1:bs
        ypred = model(reshape(xdata[:,:,:,(j-1)*bs + i],(5,nx,ny,1)))
        yactual = reshape(ydata[:,:,:,(j-1)*bs + i],(1,nx,ny,1))
        e = ypred .- yactual
        e = reshape(e,(nx,ny))
        p1 = heatmap(ypred[1,:,:,1],title="FNO t=$(slices[i])")
        p2 = heatmap(yactual[1,:,:,1],title="Original t=$(slices[i])")
        p3 = heatmap(e,title="error = $(norm(e))")
        plot(p1,p2,p3,layout=3)
        savefig("validation/ts/$(i).png")
    end 
end