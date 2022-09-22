using Flux
using FluxOptTools
using Optim
using TensorBoardLogger
using ProgressMeter
using Zygote
using BSON
using CUDA
CUDA.allowscalar(false)

# FluxOpt Tools upgrade to Complex weights
Base.zeros(grads::Zygote.Grads) = zeros(ComplexF64,veclength(grads))
Base.zeros(pars::Flux.Params) = zeros(ComplexF64,veclength(pars))


function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

function callback(losses,model,epoch,foldername)
    # Checkpoint
    BSON.@save string(foldername,"Epoch_$(epoch)_TrainLoss_$(losses[1])") model
    
    # Log on Tensorboard
    dict = Dict{String,Any}()
    fill_param_dict!(dict,model,"")
    TensorBoardLogger.with_logger(logger) do 
        @info "train" losses[1] log_step_increment=0
        @info "test" losses[2]
    end 

    # Print losses
    println("Training Loss = $(losses[1]) \t Testing Loss = $(losses[2])\n")
end 

function train(model,loss,data,opt,epoch)
    trainloader,testloader = data
    local trainloss = 0.0
    local trainagg = 0.0
    ps = Flux.Params(Flux.params(model))

    printstyled("Training Phase\n";color=:green)
    ntrain = length(trainloader)
    p = Progress(ntrain)
    for (x,y) in trainloader
        xtrain,ytrain = gpu(x),gpu(y)
        gs = Flux.gradient(ps) do
            trainloss = loss(model(xtrain),ytrain)
            return trainloss
        end 
        Flux.Optimise.update!(opt,ps,gs)
        trainagg += loss(model(xtrain),ytrain)
        next!(p)
    end 
    trainagg /= ntrain

    printstyled("Validaiton Phase\n";color=:red)
    ntest = length(testloader)
    local testagg = 0.0 
    for (x,y) in testloader
        xtest,ytest = gpu(x),gpu(y)
        testagg += loss(model(xtest),ytest)
    end 
    testagg /= ntest

    losses = (trainagg,testagg)    
    if mod(epoch,1) == 0
        callback(losses,model,epoch,foldername)
    end 
end 

function learn(model,loss,data,opt,nepochs,foldername)
    for epoch in 1:nepochs
        printstyled("\nEpoch $(epoch)\n";color=:blue)
        train(model,loss,data,opt,epoch)
    end 
end 