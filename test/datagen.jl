# Mother of all scripts
include("../src/data-gen.jl")

# Get one simulation instance
nx = 20
ny = 20
_,boundary,sol = solver(nx,ny,TwoD());
@assert length(size(sol)) == 3 
@assert size(sol)[1]== nx+1 && size(sol)[2] == ny+1
@assert length(boundary) == (nx+1)*(ny+1)

# Test multiple simulations
n = 10
sensordata,boundary,solutions = generate_data(nx,ny,n)
@assert length(sensordata) == 3
@assert length(solutions) == n


# Test writing routines
filename = "./data/GROUND_TRUTH"
data_extraction(solutions,filename)


# write the boundary data and the associated initial conditions
filename = "./data/Boundary"
boundary_extraction(Array(boundary),solutions,filename)

# Sampling routine
samples = samplesensor(sensordata,solutions[1])
@assert size(samples,1) == 10