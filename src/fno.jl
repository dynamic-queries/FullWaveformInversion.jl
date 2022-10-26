function surrogate(timestep::Int)
    # initial surrogate
    @load "best/is/is_6l_ 16d" modeltemp
    SurA = modeltemp
    
    # surrogate corresponding to "timestep"
    @load "best/os/$(timestep)" modeltemp
    SurB = modeltemp

    # compose the two
    return x -> SurB(SurA(x))
end 

function sample(Y::Array, sensorlocs::Vector)
    Z = zeros(length(sensorlocs))
    k = 1
    for coords in sensorlocs
        x,y = coords
        Z[k] = Y[x,y]
    end 
    Z
end 

function timeseries!(initial::Array, final::Array, TS::Matrix, sensorlocs::Vector)
    # We train a maximum of 100 surrogates for the first 100 time steps.
    nmax = 100
    for j=1:nmax
        sur = surrogate(j)
        final .= sur(initial)
        TS[:,j] .= sample(out,sensorlocs)
    end 
end 