using Plots
using HDF5 
gr()

foldername = "data/"

## Dynamic Field Plots
filename = string(foldername,"dynamic/GROUND_TRUTH")
for n=1:1
    file = h5open(string(filename,"$(n)"))
    heatmap()
    anim = @animate for i=1:501
        heatmap(read(file["$(i)"]),title="$(i)",clim=(-2,2))
    end 
    gif(anim,"./figures/wave_prop/$(n).gif",fps=20)
end 