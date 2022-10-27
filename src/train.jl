# FluxOpt Tools upgrade to Complex weights
Base.zeros(grads::Zygote.Grads) = CUDA.zeros(ComplexF64,veclength(grads))
Base.zeros(pars::Flux.Params) = CUDA.zeros(ComplexF64,veclength(pars))


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

function callback(losses,model,epoch,foldername,logger)
    # Checkpoint
    modeltemp = model |> cpu
    BSON.@save string(foldername,"Epoch_$(epoch)_TrainLoss_$(losses[1])") modeltemp 
    
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

function train(model,loss,data,opt,epoch,foldername,logger)
    trainloader,testloader = data
    local trainloss = 0.0
    local trainagg = 0.0
    ps = Flux.Params(Flux.params(model))

    if mod(epoch,10) == 0
        printstyled("Training Phase\n";color=:green)
    end 
    
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

    if mod(epoch,10) == 0
        printstyled("Validaiton Phase\n";color=:red)
    end 

    ntest = length(testloader)
    local testagg = 0.0 
    for (x,y) in testloader
        xtest,ytest = gpu(x),gpu(y)
        testagg += loss(model(xtest),ytest)
    end 
    testagg /= ntest

    losses = (trainagg,testagg)    
    if mod(epoch,10) == 0
        callback(losses,model,epoch,foldername,logger)
    end 
end 

function learn(model,loss,data,opt,nepochs,foldername,logger)
    for epoch in 1:nepochs
        printstyled("\nEpoch $(epoch)\n";color=:blue)
        train(model,loss,data,opt,epoch,foldername,logger)
    end 
end 