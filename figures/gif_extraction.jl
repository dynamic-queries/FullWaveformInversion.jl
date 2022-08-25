using Plots
using HDF5 

foldername = "data/"

## Dynamic Field Plots
filename = string(foldername,"dynamic/GROUND_TRUTH")
for n=1:1
    file = h5open(string(filename,"$(n)"))
    anim = @animate for i=1:501
        heatmap(read(file["$(i)"]),title="$(i)")
    end 
    gif(anim,"./figure/wave_prop/$(n).gif",fps=20)
end 