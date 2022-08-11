include("../src/data-gen.jl")
using NeuralOperators

# We know apriori that the problem was solved in the domain.
# x ∈ [0,1]
# y ∈ [0,1]
# With a resolution of 199 internal points.

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0

X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

# read the input data.
n = 10
filename = "/home/dynamic-queries/.julia/dev/InverseProblems.jl/data/GROUND_TRUTH"

i = 1
xdata = Array{Float64,4}(undef,)
file = h5open(joinpath(filename,i),"r")
U = []
for t=1:10
    push!(U,read(file["t"]))
end 
