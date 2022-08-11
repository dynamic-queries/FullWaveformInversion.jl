# Mother of all scripts
include("../src/data-gen.jl")

# Get one simulation instance
nx = 20
ny = 20
_,sol = solver(nx,ny,TwoD());
@assert length(size(sol)) == 3 
@assert size(sol)[1]== nx+1 && size(sol)[2] == ny+1

# Test multiple simulations
n = 10
sensordata,solutions = generate_data(nx,ny,n)
@assert length(sensordata) == 3
@assert length(solutions) == n

# Test writing routines
filename = "./data/GROUND_TRUTH"
data_extraction(solutions,filename)

# Sampling routine
samples = samplesensor(sensordata,solutions[1])
@assert size(samples,1) == 10