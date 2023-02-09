abstract type Eval end 
struct Cheap <: Eval end 
struct Expensive <: Eval end

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
    initial_surrogate 

    function SurrogateModel(nt,nx,ny,boundary,test_set_number=1905)
        # Space
        x = 0.0:(1.0)/nx:1.0
        y = 0.0:(1.0)/ny:1.0
        X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
        Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

        # Sensor locations
        xsensors = Vector(5:floor(Int,(length(x)-5)/8):length(x))
        ysensors = 3*ones(Int,length(xsensors))

        # Cache
        cache = SurrogateModelCache(nx,ny,X,Y,boundary)

        # Surrogates
        SurBs = []
        filename = "best/os/"
        ntsteps = 70
        for i=1:ntsteps
            @load string(filename,"$(i)") modeltemp
            push!(SurBs,modeltemp |> gpu)
        end 

        # Initial Surrogate
        BSON.@load "best/is/is_smooth" modeltemp
        SurA = modeltemp |> gpu

        new(nt,nx,ny,boundary,test_set_number,x,y,X,Y,xsensors,ysensors,cache,SurBs,SurA)
    end 
end 

mutable struct SurrogateModelCache
    input
    inter
    output

    function SurrogateModelCache(nx,ny,X,Y,boundary)
        input = CUDA.zeros(3,nx+1,ny+1,1)
        input[1,:,:,1] = boundary
        input[2,:,:,1] = X
        input[3,:,:,1] = Y
        inter = zeros(nx+1,ny+1)
        output = CUDA.zeros(Float64,1,nx+1,ny+1,1)
        new(input,inter,output)
    end 
end 

# Versions that make use of an input solver.
function evaluate(sur::SurrogateModel,evol::Eval)
    @unpack inter = sur.cache
    @unpack ntsteps,nx,ny,boundary,surrogates = sur
    @unpack xsensors,ysensors = sur
    
    ns = length(xsensors)
    timeseries = zeros(ns,ntsteps)
    solutions = evaluate_verbose(sur,evol)[2]
    for i=1:ntsteps
        timeseries[:,i] = solutions[i][1,:,:,1][xsensors,3]
    end 
    return timeseries
end 

function evaluate(sur::SurrogateModel,boundary,evol::Eval)
    sur.boundary = boundary
    evaluate(sur,evol)
end 

function evaluate_verbose(sur::SurrogateModel,evol::Expensive)
    @unpack input,inter,output = sur.cache
    @unpack ntsteps,nx,ny,boundary,surrogates = sur
    @unpack xsensors,ysensors = sur

    solutions = []
    inter = solver_init(nx,ny,TwoD(),boundary)
    input[1,:,:,1] = inter

    for i=1:ntsteps
        sur = surrogates[i]
        output = sur(input)
        push!(solutions,output|>cpu)
    end    
    inter|>cpu,solutions
end

# Smoother
function smooth_defect(nx,ny,mean,σ)

    # Domain
    x = 0.0:1/(nx-1):1.0
    y = 0.0:1/(ny-1):1.0

    # Gaussian function
    g(x,y,mx,my,σ) = exp(-((x-mx)/σ)^2-((y-my)/σ)^2)

    function gaussian_filter(mx,my,σ)
        na,nb = length(x),length(y)
        Z = zeros(na,nb)
        for i=1:na
            for j=1:nb
                Z[i,j] = g(x[i],y[j],mx,my,σ)
            end 
        end 
        Z
    end 

    # Smoothed defect
    D = zeros(Float64,nx,ny)

    # Smoothen defect
    for i=1:nx 
        for j=1:ny
            if mean[i,j] != 0
                D+=gaussian_filter(x[i],y[j],σ)
            end 
        end
    end

    D = D./ maximum(D)
    D
end 

# Versions that make use of the surrogate
function evaluate_verbose(sur::SurrogateModel,::Cheap)
    @unpack input,inter,output = sur.cache
    @unpack ntsteps,nx,ny,boundary,surrogates,initial_surrogate = sur
    @unpack xsensors,ysensors = sur

    sigma = 0.035
    boundary = smooth_defect(nx+1,ny+1,boundary,sigma)
    input[1,:,:,1] = boundary
    
    inter = initial_surrogate(input)[1,:,:,1]

    solutions = []
    input[1,:,:,1] = inter 
    for i=1:ntsteps
        sur = surrogates[i] |> gpu
        output = sur(input)
        push!(solutions,output|>cpu)
    end    
    inter|>cpu,solutions
end 