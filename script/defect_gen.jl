using FullWaveformInversion
using MPI
const FWI = FullWaveformInversion

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm) + 1
#
print("Generating data in rank $(rank)\n")
nx = 50
ny = 50
filename = "data/rbf/$(rank)/"
if !isdir(filename)
    mkpath(filename)
end 
ninstances = 1000
FWI.data_gen_init(nx,ny,ninstances,string(filename,"BOUNDARY"))
#
MPI.Barrier(comm)
MPI.Finalize()