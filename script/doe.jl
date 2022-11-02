# Design of Experiments for training surrogates.
# Target:
# Timesteps : 1:50

## Imports 
using Flux
using HDF5
using NeuralOperators
using TensorBoardLogger
using FullWaveformInversion:learn
using CUDA

K = 10:10:50
## Timestep of Interest
for k in K

    print("=================================================================\n")
    print("Training a surrogate for the $(k)th time step.\n")
    print("=================================================================\n\n")

    ## Get data
    x = 0.0:(1.0)/200:1.0
    y = 0.0:(1.0)/200:1.0
    X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
    Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

    filename = "/tmp/ge96gak/consolidated_data/dynamic/GROUND_TRUTH"

    batches = 1:700
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
        ydata[1,:,:,i] .= read(file["$(k)"])
    end

    print("Data has been read...\n")

    ## DataLoader
    traind,testd = Flux.splitobs((xdata,ydata),at=0.9)
    train_loader = Flux.DataLoader(traind,batchsize=2,shuffle=true)
    test_loader = Flux.DataLoader(testd,batchsize=1,shuffle=false)


    ## Models
    function FourierLayer(DL,nmodes)
        OperatorKernel(DL=>DL, (nmodes,nmodes), FourierTransform, gelu)
    end

    function MultipleFourierLayers(nlayers,DL,nmodes)
        [FourierLayer(DL,nmodes) for i=1:nlayers]
    end 

    DLs = [16]
    nmodes = 12
    nfourier_layers = 6

    for DL in DLs

        print("-------------------------------------\n")
        print("Using a layer depth of $(DL)\n")
        print("-------------------------------------\n\n")

        file = "doe/os/$(k)/$(nfourier_layers)/$(DL)/"
        if !ispath(file)
            mkpath(file)
        end 

        model = Chain(
                    Dense(3,DL),
                    MultipleFourierLayers(nfourier_layers,DL,nmodes)...,
                    Dense(DL,1)
                    )
        
        lossfunction = l₂loss
        data = (train_loader,test_loader)
        
        print("Training model... \n")

        model = gpu(model)
        logger = TBLogger(string(file,"logs/"))
        foldername = string(file,"weights/")
        if !ispath(foldername)
            mkpath(foldername)
        end 

        lr = 1e-2
        nepochs = 20
        opt = Flux.Adam(lr)
        learn(model,lossfunction,data,opt,nepochs,foldername,logger)

        lr = 1e-3
        nepochs = 150
        opt = Flux.Adam(lr)
        learn(model,lossfunction,data,opt,nepochs,foldername,logger)

        lr = 1e-4
        nepochs = 30
        opt = Flux.Adam(lr)
        learn(model,lossfunction,data,opt,nepochs,foldername,logger)
    end 

end