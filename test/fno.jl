using HDF5
using NeuralOperators
using FullWaveformInversion
using Flux
using FullWaveformInversion:surrogate
using TensorBoardLogger
using BSON
using Plots
using KissSmoothing


function coordinates(k)
    n = 3:(199/10):199
    n = floor.(Int,n)
    [(ni,k) for ni in n]
end 


function solve_instance(instance,vis::Bool=false)
    # We know apriori that the problem was solved in the domain.
    # x ∈ [0,1]
    # y ∈ [0,1]
    # With a resolution of 199 internal points.

    x = 0.0:(1.0)/200:1.0
    y = 0.0:(1.0)/200:1.0

    X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
    Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

    ## Data
    filename = "data/2022-11-24T18/p_data/1/static/BOUNDARY"
    filename2 = "data/2022-11-24T18/p_data/1/dynamic/GROUND_TRUTH$(instance)"

    nsteps = 39
    nsensors = 10


    batches = 1:1
    BS = length(batches)
    nx = length(x)
    ny = length(y)

    xdata = Array{Float64,4}(undef,3,nx,ny,BS)
    ydata = Array{Float64,4}(undef,1,nx,ny,nsteps)

    file = h5open(filename,"r")

    for (i,b) in enumerate(batches)
        xdata[1,:,:,i] .= read(file["b$(instance)"])
        xdata[2,:,:,i] .= X 
        xdata[3,:,:,i] .= Y
    end

    filed = h5open(filename2,"r")
    for j=1:nsteps
        ydata[1,:,:,j] .= read(filed["$(j)"])
    end 

    # IN data generated from the surrogate
    sol = zero(ydata)
    ndatapoints = length(y)
    sensor_range = 0.0:(1.0)/(nsensors-1):1.0
    T = zeros(nsensors,nsteps)

    # Get error plots
    y = zero(ydata)
    for i=1:nsteps
        sur = surrogate(i)
        temp = sur(xdata)
        if vis
            f1 = heatmap(temp[1,:,:,1],title="FNO")
            f2 = heatmap(ydata[1,:,:,i],title="Simulation")
            f3 = heatmap((temp[1,:,:,1]-ydata[1,:,:,i]),title="Abs Error")
            @show maximum(ydata[1,:,:,i])
            f4 = heatmap((temp[1,:,:,1]-ydata[1,:,:,i])/maximum(ydata[1,:,:,i]),title="Rel Error")
            plot(f1,f2,f3,f4)
            savefig("figures/ts_val/$(i).png")
        end
        y[1,:,:,i] = temp[1,:,:,1]
    end 

    if vis
        anim2 = @animate for ts = 1:39
            heatmap(ydata[1,:,:,ts],title="$(ts)th timestep")
        end
        gif(anim2,"figures/$(instance)_simualtion.gif",fps=5)

        anim3 = @animate for ts = 1:39
            heatmap(y[1,:,:,ts],title="$(ts)th timestep")
        end
        gif(anim3,"figures/$(instance)_surrogate.gif",fps=5)
        bound = heatmap(xdata[1,:,:,1],title="Defect")
    end
    
    if vis
        K = 101:101
        anim = @animate for k in K
            coord = coordinates(k)
            Tsur = zero(T)
            for j=1:nsteps
                Tsur[:,j] = sample(y[1,:,:,j],coord)
            end

            Tactual = zero(T)
            for j=1:nsteps
                Tactual[:,j] = sample(ydata[1,:,:,j],coord)
            end 

            figures = []
            for i=1:nsensors
                f = plot(Tactual[i,:],xlabel="ntsteps",ylabel="U",label="Actual",size=(1000,1000))
                scatter!(Tsur[i,:],label="Surrogate",title="$(i)")
                push!(figures,f)
            end 
            plot(figures...,bound)
            plot!(title="$(k)th layer")

            if !isdir("figures/$(instance)/timeseries/")
                mkpath("figures/$(instance)/timeseries/")
            end 

            savefig("figures/$(instance)/timeseries/$(k).png")
        end 
    else
        k=101
        coord = coordinates(k)
        Tsur = zero(T)
        for j=1:nsteps
            Tsur[:,j] = sample(y[1,:,:,j],coord)
        end
    
        Tactual = zero(T)
        for j=1:nsteps
            Tactual[:,j] = sample(ydata[1,:,:,j],coord)
        end 
    end 
    Tsur,Tactual
end

# Denoise the data
Tsurrogate,Tactual = solve_instance(1,true)

DN = zero(Tsurrogate)
anim = @animate for i=1:10
    D = Tsurrogate[i,:]
    Dn,N = denoise(D)
    DN[i,:] .= Dn
    scatter(D,label="Noisy data",title="Sensor $(i)")
    plot!(Dn,label="Cleaned surrogate data")
end
gif(anim,"figures/gifs/denoising.gif",fps=3)

FIGS = []
for i=1:10
    f= plot(DN[i,:],label="Cleaned surrogate",title="Sensor $(i)")
    plot!(Tactual[i,:],label="Simulation data",size=(1000,1000))
    push!(FIGS,f)
end 
plot(FIGS...)
savefig("figures/comparison_smoothsur_vs_simulation.png")