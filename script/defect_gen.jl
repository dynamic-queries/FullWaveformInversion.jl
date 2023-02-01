using FullWaveformInversion
const FWI = FullWaveformInversion

nx = 50
ny = 50
filename = "consolidated/rbf/BOUNDARY"
ninstances = 20
FWI.data_gen_init(nx,ny,ninstances,filename)
