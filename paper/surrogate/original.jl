using FullWaveformInversion
include("preamble.jl")

# Boundary
boundary = boundary

# FOM1 
solutionA = read(file["$(test_set)"])

# FOM2
solutionB = [] 
ntsteps = 39
for i=1:ntsteps
    file = h5open(string(filename,"dynamic/GROUND_TRUTH$(test_set)"))
    push!(solutionB,read(file["$(i)"]))
end 


# # Visualize original
# fig1 = heatmap(boundary,title="Defect")
# fig2 = heatmap(solutionA,title="t=s+1")
# fig3 = heatmap(solutionB[end],title="t=s+k")
# plot(fig1,fig2,fig3,layout=(2,2),surtitle="Numerical Simulation")
