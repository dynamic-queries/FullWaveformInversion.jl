module FullWaveformInversion

using Interpolations
using LinearAlgebra
using Distributions
using SparseArrays
using LinearAlgebra
using Plots
using OrdinaryDiffEq 
using OrdinaryDiffEq:solve
using HDF5
using MPI
using Flux
using FluxOptTools
using Optim
using TensorBoardLogger
using ProgressMeter
using Zygote
using CUDA
using NeuralOperators
using BSON
using BSON:@load
CUDA.allowscalar(false)

abstract type Dimension end 
struct OneD <: Dimension end 
struct TwoD <: Dimension end 

include("utils.jl")
include("solvers.jl")
include("data-gen.jl")
include("train.jl")
include("fno.jl")
include("gar.jl")
include("optimization.jl")

export surrogate, sample, timeseries
# export DefectReconstructionProblem, solve, __solve

end 