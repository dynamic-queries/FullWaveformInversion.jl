using HDF5
using NeuralOperators
using FullWaveformInversion
using Flux
using FullWaveformInversion:learn
using TensorBoardLogger
using BSON
using Plots

function surrogate(timestep::Int)
    # initial surrogate
    BSON.@load "best/is/is_6l_ 16d" modeltemp
    SurA = modeltemp
    
    # surrogate corresponding to "timestep"
    BSON.@load "best/os/$(timestep)" modeltemp
    SurB = modeltemp
    sur = function(xdata)
        temp = zero(xdata)
        temp .= SurA(xdata)
        final = SurB(temp)
        return final
    end 
    # compose the two
    return sur
end 

function sample(Y::Array, sensorlocs::Vector)
    Z = zeros(length(sensorlocs))
    k = 1
    for coords in sensorlocs
        x,y = coords
        Z[k] = Y[x,y]
        k += 1 
    end 
    Z
end 

function timeseries!(nsteps::Int,initial::Array, final::Array, TS::Matrix, sensorlocs::Vector)
    # We train a maximum of 100 surrogates for the first 100 time steps.
    for j=1:nsteps
        sur = surrogate(j)
        final .= sur(initial)
        TS[:,j] = sample(final[1,:,:,1],sensorlocs)
    end 
end 

function coordinates(k)
    n = 3:(199/10):199
    n = floor.(Int,n)
    [(ni,k) for ni in n]
end 

function Plots.plot(M::Matrix,title::String)
    t = 1:size(M,2)
    fig = []
    for i=1:size(M,1)
        push!(fig,Plots.plot(t,M[i,:]));
    end
    Plots.plot(fig...,title=title)
end 

function Plots.plot(M1::Matrix,M2::Matrix,title::String)
    t = 1:size(M1,2)
    fig = []
    for i=1:size(M1,1)
        temp = Plots.plot(t,M1[i,:],title="Surrogate")
        Plots.plot!(t,M2[i,:],title="Simulation")
        push!(fig,temp);
    end
    Plots.plot(fig...,title=title)
end

begin
    # We know apriori that the problem was solved in the domain.
    # x ∈ [0,1]
    # y ∈ [0,1]
    # With a resolution of 199 internal points.

    x = 0.0:(1.0)/200:1.0
    y = 0.0:(1.0)/200:1.0

    X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
    Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

    ## Data
    filename = "data/2022-11-24T13/p_data/1/static/BOUNDARY"
    filename2 = "data/2022-11-24T13/p_data/1/dynamic/GROUND_TRUTH1"

    nsteps = 39
    nsensors = 10


    batches = 1:1
    BS = length(batches)
    nx = length(x)
    ny = length(y)

    xdata = Array{Float64,4}(undef,3,nx,ny,BS)
    ydata = Array{Float64,4}(undef,1,nx,ny,nsteps)

    file = h5open(filename,"r")

    k = 1
    for (i,b) in enumerate(batches)
        xdata[1,:,:,i] .= read(file["b$(k)"])
        xdata[2,:,:,i] .= X 
        xdata[3,:,:,i] .= Y
    end

    filed = h5open(filename2,"r")
    for j=1:nsteps
        ydata[1,:,:,j] .= read(filed["$(j)"])
    end 
end 

begin
    # IN data generated from the surrogate
    sol = zero(ydata)
    ndatapoints = length(y)
    sensor_range = 0.0:(1.0)/(nsensors-1):1.0
    T = zeros(nsensors,nsteps)
end 

# Get error plots
y = zero(ydata)
for i=1:nsteps
    sur = surrogate(i)
    temp = sur(xdata)
    f1 = heatmap(temp[1,:,:,1],title="FNO")
    f2 = heatmap(ydata[1,:,:,i],title="Simulation")
    f3 = heatmap((temp[1,:,:,1]-ydata[1,:,:,i]),title="Abs Error")
    @show maximum(ydata[1,:,:,i])
    f4 = heatmap((temp[1,:,:,1]-ydata[1,:,:,i])/maximum(ydata[1,:,:,i]),title="Rel Error")
    plot(f1,f2,f3,f4)
    savefig("figures/ts_val/$(i).png")
    y[1,:,:,i] = temp[1,:,:,1]
end 

bound = heatmap(xdata[1,:,:,1],title="Defect")

K = Int.(2:200)
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
    savefig("figures/timeseries/$(k).png")
end 
gif(anim,"figures/sampling_layers.gif",fps=10)