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
export SurrogateModel, evaluate, evaluate_verbose

end 