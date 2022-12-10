using FullWaveformInversion
using MPI
using Dates


date_time = string(now())[1:13]

const FWI = FullWaveformInversion

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) + 1

nx = 200
ny = 200
# Number of instances of the problem.
N = [1,20,20,20,20]
Defects = [Empty(),Point(),Linear(),Ellipse(),Spline()]
sensordata,boundaries,solutions = FWI.generate_data(nx,ny,N,Defects)

foldername = "data/$(date_time)/p_data/$(rank)/dynamic/"
if !isdir(foldername)
    mkpath(foldername)
end 
filename = "data/$(date_time)/p_data/$(rank)/dynamic/GROUND_TRUTH"
FWI.data_extraction(solutions,filename)


foldername = "data/$(date_time)/p_data/$(rank)/static/"
if !isdir(foldername)
    mkpath(foldername)
end 
filename = "data/$(date_time)/p_data/$(rank)/static/BOUNDARY"
FWI.boundary_extraction(boundaries,solutions,filename)

MPI.Barrier(comm)