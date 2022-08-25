using Plots
using HDF5 

foldername = "data/"

# Initial Conditions
## Crack
filename = string(foldername,"static/BOUNDARY")
file = h5open(filename)

ninstances = 100
for n in 1:ninstances
    boundary = read(file["b$(n)"])
    heatmap(boundary);
    savefig("./figures/cracks/$(n)b")

    initial = read(file["$(n)"])
    heatmap(initial);
    savefig("./figures/initials/$(n)")
end 