using BSON: @load
using NeuralOperators

dir = "src/ts_checkpoints"
filename = readdir(dir)[end]
@load string(@__DIR__,"/ts_checkpoints/",filename) model

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

batches = 1:1
BS = length(batches)
bs = BS*501 - 1

filename = "./data/dynamic/GROUND_TRUTH"

xdata = Array{Float32,4}(undef,3,nx,ny,bs)
ydata = Array{Float32,4}(undef,1,nx,ny,bs)

print("Reading Data ... \n\n")

for (nb,b) in enumerate(batches)
    file = h5open(string(filename,nb),"r")
    for it=1:bs
        xdata[1,:,:,(nb-1)*bs + it] .= read(file["$(it)"])
        xdata[2,:,:,(nb-1)*bs + it] .= X
        xdata[3,:,:,(nb-1)*bs + it] .= Y
        ydata[1,:,:,(nb-1)*bs + it] .= read(file["$(it+1)"]) 
    end 
end 

# Prediction
anim = @animate for i=1:500
    e = model(reshape(xdata[:,:,:,i],(3,nx,ny,1))) .- reshape(ydata[:,:,:,i],(1,nx,ny,1))
    e = reshape(e,(nx,ny))
    heatmap(e,title="$(i)",cmap=(-1,1))
end 
gif(anim,"ts_error.gif",fps=10)