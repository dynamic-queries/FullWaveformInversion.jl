mutable struct SurrogateModel
    ntsteps
    nx
    ny
    boundary
    test_set_number
    x
    y
    X
    Y
    xsensors
    ysensors
    cache
    surrogates

    function SurrogateModel(nt,nx,ny,boundary,test_set_number=1905)
        # Space
        x = 0.0:(1.0)/nx:1.0
        y = 0.0:(1.0)/ny:1.0
        X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
        Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

        # Sensor locations
        xsensors = Vector(5:20:196)
        ysensors = 3*ones(Int,length(xsensors))

        # Cache
        cache = SurrogateModelCache(nx,ny,X,Y,boundary)

        # Surrogates
        SurBs = []
        filename = "best/os/"
        ntsteps = 70
        for i=1:ntsteps
            @load string(filename,"$(i)") modeltemp
            push!(SurBs,modeltemp)
        end 

        new(nt,nx,ny,boundary,test_set_number,x,y,X,Y,xsensors,ysensors,cache,SurBs)
    end 
end 

mutable struct SurrogateModelCache
    input
    inter
    output

    function SurrogateModelCache(nx,ny,X,Y,boundary)
        input = Array{Float64,4}(undef,3,nx+1,ny+1,1)
        input[1,:,:,1] = boundary
        input[2,:,:,1] = X
        input[3,:,:,1] = Y
        inter = zeros(nx+1,ny+1)
        output = zeros(Float64,1,nx+1,ny+1,1)
        new(input,inter,output)
    end 
end 

function evaluate(sur::SurrogateModel)
    @unpack input,inter,output = sur.cache
    @unpack ntsteps,nx,ny,boundary,surrogates = sur
    @unpack xsensors,ysensors = sur
    
    ns = length(xsensors)
    
    timeseries = zeros(ns,ntsteps)
    inter = solver_init(nx,ny,TwoD(),boundary)
    input[1,:,:,1] = inter
    for i=1:ntsteps
        sur = surrogates[i]
        output = sur(input)
        timeseries[:,i] = output[1,:,:,1][xsensors,3]
    end 
    return timeseries
end 

function evaluate(sur::SurrogateModel,boundary)
    sur.boundary = boundary
    evaluate(sur)
end 

function evaluate_verbose(sur::SurrogateModel)
    @unpack input,inter,output = sur.cache
    @unpack ntsteps,nx,ny,boundary,surrogates = sur
    @unpack xsensors,ysensors = sur

    solutions = []
    inter = solver_init(nx,ny,TwoD(),boundary)
    input[1,:,:,1] = inter
    for i=1:ntsteps
        sur = surrogates[i]
        output = sur(input)
        push!(solutions,copy(output[1,:,:,1]))
    end    
    solutions
end 