include("original.jl")
include("surrogate.jl")
using Plots
using LinearAlgebra
using Measures

## Timeseries
# FOM
UFO = []
push!(UFO,solutionA[xsensors,3])
for i=2:ntsteps
    push!(UFO,solutionB[i][xsensors,3])
end 
UFO = reduce(hcat,UFO) 

# ROM
URO = []
push!(URO,sur_solutionA[xsensors,3])
for i=2:ntsteps
    push!(URO,sur_solutionB[i][xsensors,3])
end 
URO = reduce(hcat,URO)

# Visualize the defect, simulation and surrogate results
theme = :coolwarm
fs = 12
f1 = heatmap(boundary,title="Defect",c=:algae,titlefontsize=fs)
f2 = heatmap(solutionA,c=theme,title="Simulation : t=ts+k δt",titlefontsize=fs)
f3 = heatmap(solutionB[end],c=theme,title="Simulation :t=ts+k δt",titlefontsize=fs)

f4 = heatmap(sur_solutionA,c=theme,title="Surrogate : t=ts+k δt",titlefontsize=fs)
f5 = heatmap(sur_solutionB[end],c=theme,title="Surrogate : t=ts+k δt",titlefontsize=fs)
f6 = heatmap(sur_solutionA-solutionA,c=theme,title="Error: t=ts+k δt",titlefontsize=fs)
f7 = heatmap(sur_solutionB[end]-solutionB[end],c=theme,title="Error: t=ts+k δt",titlefontsize=fs)

plot(f1,f2,f6,f4,xaxis=false,yaxis=false,axis=nothing,size=(500,500))
savefig("paper/figures/TS_1_sur_vs_sim.svg")

plot(f1,f3,f7,f5,xaxis=false,yaxis=false,axis=nothing,size=(500,500))
savefig("paper/figures/TS_k_sur_vs_sim.svg")

# Visualize time series
plt = []
for i=1:10
    f = plot(UFO[i,:],label="Simulation",title="Sensor $(i)")
    scatter!(URO[i,:],label="Surrogate")
    push!(plt,f)
end 
plot(plt...,layout=(2,5),size=(1400,750),margin=7.5mm)
savefig("paper/figures/Timeseries_sim_vs_sur_$(test_set).svg")