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

function standalone_test(boundary,boundary_name)
    nx = 200
    ny = 200
    nsteps = 39
    nsensors = 10
    boundary = BitMatrix(undef,nx+1,ny+1)
    boundary .= 0
    
    # Load Surrogate
    x = 0.0:(1.0)/nx:1.0
    y = 0.0:(1.0)/ny:1.0

    X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
    Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))
    
    if isnothing(boundary)
        _,boundary,u = FullWaveformInversion.solver(nx,ny,TwoD())
    else
        _,_,u = FullWaveformInversion.solver(nx,ny,TwoD(),boundary)
    end 

    f1 = heatmap(boundary,title=boundary_name)

    anim = @animate for ts=1:nsteps
        heatmap(u[:,:,ts],title="$(ts) time step")
    end 
    f2 = gif(anim,"figures/$(boundary_name).gif",fps=10)

    batches = 1:1
    BS = length(batches)
    nx = length(x)
    ny = length(y)

    xdata = Array{Float64,4}(undef,3,nx,ny,BS)
    ydata = Array{Float64,4}(undef,1,nx,ny,nsteps)

    xdata[1,:,:,1] = float(boundary)
    xdata[2,:,:,1] = X
    xdata[3,:,:,1] = Y

    for ts = 1:nsteps
        ydata[1,:,:,ts] .= u[:,:,ts]
    end 

    BSON.@load "best/is/is_6l_ 16d" modeltemp
    surA = modeltemp
    uinit = surA(xdata)
    f3 = heatmap(uinit[1,:,:,1],title="Initial Surrogate") 

    ysur = zero(ydata)
    for k = 1:nsteps
        sur = FullWaveformInversion.surrogate(k)
        ysur[1,:,:,k] = sur(xdata)[1,:,:,1]
    end

    anim = @animate for ts=1:nsteps
        heatmap(ysur[1,:,:,ts],title="$(ts) time step")
    end 
    f4 = gif(anim,"figures/$(boundary_name)_surrogate.gif",fps=10)
    # plot(f1,f3,f2,f4)
    ysur,ydata
end 