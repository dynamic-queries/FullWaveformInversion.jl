include("../src/data-gen.jl")

nx = 200
ny = 200
N = 100
@time sensordata,solutions = generate_data(nx,ny,N)
filename = "./data/GROUND_TRUTH"
data_extraction(solutions,filename)