using FullWaveformInversion
using MPI
const FWI = FullWaveformInversion

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) + 1
#
print("Generating data in rank $(rank)\n")
nx = 200
ny = 200
filename = "data/rbf/$(rank)/"
if !isdir(filename)
    mkpath(filename)
end 
ninstances = 50
FWI.data_gen_init(nx,ny,ninstances,string(filename,"BOUNDARY"))
#
MPI.Barrier(comm)
MPI.Finalize()