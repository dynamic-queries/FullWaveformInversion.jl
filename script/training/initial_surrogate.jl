using HDF5 
using NeuralOperators
using FullWaveformInversion
using Flux
using FullWaveformInversion:learn
using TensorBoardLogger


filename = "consolidated_data/static/BOUNDARY"
file = h5open(filename)

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

batches = 1:745
BS = length(batches)

filename =  "consolidated_data/rbf_defects"
file_rbf = h5open(filename)
bs_rbf = length(file_rbf)


xdata = Array{Float64,4}(undef,3,nx,ny,BS+bs_rbf)
ydata = Array{Float64,4}(undef,1,nx,ny,BS+bs_rbf)

for (ib,b) in enumerate(batches)
    xdata[1,:,:,ib] .= read(file["b$(b)"])
    xdata[2,:,:,ib] .= X
    xdata[3,:,:,ib] .= Y
    ydata[1,:,:,ib] .= read(file["$(b)"])
end 

batches = 1:990
for i=1:batches
    xdata[1,:,:,BS+i] .= read(file["$(i)"])
    xdata[2,:,:,BS+i] .= X
    xdata[3,:,:,BS+i] .= Y
    ydata[1,:,:,BS+i] .= read(file["$(b)"])
end 

print("Read Data ... \n")

traind,testd = Flux.splitobs((xdata,ydata),at=0.9)
train_loader = Flux.DataLoader(traind,batchsize=1,shuffle=true)
test_loader = Flux.DataLoader(testd,batchsize=1,shuffle=false)

DLs = [16,24,32]
nmodes = 12

for DL in DLs
    model = Chain(
            Dense(3,DL),
            OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
            OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
            OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
            OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
            OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
            OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu),
            Dense(DL,DL),
            Dense(DL,1)
    )

    # Optimizer params
    lossfunction = l₂loss
    data = (train_loader,test_loader)

    # Optimizer params
    lossfunction = l₂loss
    data = (train_loader,test_loader)
    foldername = "weights/is/$(DL)/"

    if !isdir(foldername)
        mkdir(foldername)
    end 

    print("Training model... \n")

    model = gpu(model)
    logger = TBLogger("script/logs/is/$(DL)/")

    lr = 1e-2
    nepochs = 50
    opt = Flux.Adam(lr)
    learn(model,lossfunction,data,opt,nepochs,foldername,logger)

    lr = 1e-3
    nepochs = 100
    opt = Flux.Adam(lr)
    learn(model,lossfunction,data,opt,nepochs,foldername,logger)

    lr = 1e-4
    nepochs = 100
    opt = Flux.Adam(lr)
    learn(model,lossfunction,data,opt,nepochs,foldername,logger)
end