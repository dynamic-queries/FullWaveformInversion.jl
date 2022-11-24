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

