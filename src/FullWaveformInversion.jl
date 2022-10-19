module FullWaveformInversion

using Interpolations
using LinearAlgebra
using Distributions
using SparseArrays
using Plots
using OrdinaryDiffEq 
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
using BSON : @load
CUDA.allowscalar(false)

abstract type Dimension end 
struct OneD <: Dimension end 
struct TwoD <: Dimension end 

include("utils.jl")
include("solvers.jl")
include("data-gen.jl")
include("train.jl")
include("surrogate.jl")
include("optimization.jl")

end 