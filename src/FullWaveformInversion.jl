module FullWaveformInversion

using Interpolations
using LinearAlgebra
using Distributions
using SparseArrays
using LinearAlgebra
using OrdinaryDiffEq
using CommonSolve
using HDF5
using UnPack
using MPI
using Flux
using Optim
using TensorBoardLogger
using ProgressMeter
using Zygote
using CUDA
using NeuralOperators
using BSON
using BSON:@load
using Base.Threads
CUDA.allowscalar(false)

include("utils.jl")
include("solvers.jl")
include("defects.jl")
include("data-gen.jl")
include("train.jl")
include("surrogate.jl")
include("regularization.jl")
include("optimization.jl")

export solver, solver_init, TwoD
export RBF,spline2obstacle!
export SurrogateModel, evaluate ,evaluate_verbose
export Cheap, Expensive
end 