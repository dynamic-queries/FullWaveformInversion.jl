using FullWaveformInversion

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) + 1

nx = 200
ny = 200
N = 30
@time sensordata,boundaries,solutions = generate_data(nx,ny,N)

foldername = "/tmp/ge96gak/p_data/$(rank)/dynamic/"
if !isdir(foldername)
    mkpath(foldername)
end 
filename = "/tmp/ge96gak/p_data/$(rank)/dynamic/GROUND_TRUTH"
data_extraction(solutions,filename)


foldername = "/tmp/ge96gak/p_data/$(rank)/static/"
if !isdir(foldername)
    mkpath(foldername)
end 
filename = "/tmp/ge96gak/p_data/$(rank)/static/BOUNDARY"
boundary_extraction(boundaries,solutions,filename)

MPI.Barrier(comm)