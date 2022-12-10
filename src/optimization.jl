abstract type AbstractSurrogate end 
abstract type AbstractRegularizer end 

struct FNO <: AbstractSurrogate end 
struct GAR <: AbstractRegularizer end 

struct DefectReconstructionProblem
    input_signals::Matrix
    sensor_locations::Vector{Tuple}
    ndims::Int
    dims::Vector
    limits::Vector
    niters::Int
    eps::Float64

    DefectReconstructionProblem(signals,locs,dims,limits) = new(signals,locs,length(dims),dims,limits,1e6,1e-3)
    DefectReconstructionProblem(signals,locs,dims,limits,niters,eps) = new(signals,locs,length(dims),dims,limits,niters,eps)
end 

mutable struct DefectReconstructionSolution
    prob::DefectReconstructionProblem
    defect::Matrix
    learning_rate::Float64

    DefectReconstructionSolution(prob,lr) = new(prob,nothing,lr)
end 

function __solve(sol::DefectReconstructionSolution, sur::FNO, reg::GAR)
    # Forward problem parameters
    dims = sol.prob.dims
    limits = sol.prob.limits
    x = 0:(limits[1]/dims[1]):limits[1]
    y = 0:(limits[2]/dims[2]):limits[2]
    X = reshape([xi for xi in x for _ in y],(nx,ny))
    Y = reshape([yi for _ in x for yi in y],(nx,ny))
    
    # Initial guess for the defect
    sol.defect = zeros(nx,ny)
    xdata = Array{eltype(sol.prob.input_signals),3}(undef,3,nx,ny)
    xdata[1,:,:] .= sol.defect
    xdata[2,:,:] .= X
    xdata[3,:,:] .= y

    # Optimization parameters
    n = sol.prob.niters
    eps = sol.prob.eps
    a = sol.learning_rate
    iter = 1
    res = Inf
    tm = sol.prob.signals
    uf = Array{eltype(sol.prob.input_signals),3}(undef,1,nx,ny)
    TS = Matrix{eltype(sol.prob.input_signals)}(length(sol.prob.sensor_locations))


    # TODO : Add regularizer here.
    function loss(um,uf)
        norm(um-uf)
    end 

    # TODO : Implement the gradient of the loss function using pullback.
    function gradient()

    end 

    # Optimization loop - Simple stochastic gradient descent
    # TODO : Replace with optim.
    while iter <= n & res > eps
        timeseries!(sol.defect,uf,TS,sol.prob.sensor_locations)
        l = loss(uf,um)
        grad = gradient(l,sol.defect)
        sol.defect -= a*grad
    end
    return sol         
end 

function CommonSolve.solve(prob::DefectReconstructionProblem,sur::AbstractSurrogate,reg::AbstractRegularizer)
    sol = DefectReconstructionSolution(prob)
    __solve(sol,sur,reg)
end 


