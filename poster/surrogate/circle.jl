include("preamble.jl")
using Plots

# Munge data to pass to the FNO
input = Array{Float64,4}(undef,3,nx+1,ny+1,1)
input[1,:,:,1] = boundary
input[2,:,:,1] = X
input[3,:,:,1] = Y

# Solve wave propagation with a circle
boundary = BitMatrix(undef,nx+1,ny+1)
boundary .= 0
nxmid = 101
nymid = 101
radius = 40
for i=1:nx
    for j=1:ny
        if sqrt((i-nxmid)^2 + (j-nymid)^2) <= radius 
            boundary[i,j] = 1
        end 
    end 
end 
heatmap(boundary,title="Circular Boundary")
savefig("figures/circle_boundary.png")

solA = FullWaveformInversion.solver_init(nx,ny,FullWaveformInversion.TwoD(),boundary)
sur_solutionB = []
input[1,:,:,1] = solA
@time for i=1:ntsteps
    sur = SurBs[i]
    output = sur(input)
    push!(sur_solutionB,output[1,:,:,1])    
end
theme = :coolwarm
fs = 12
f1 = heatmap(solA,c=theme,title="Surrogate : t=ts+k δt",titlefontsize=fs)
figures = []
for i=1:10:70
    f = heatmap(sur_solutionB[i],c=theme,title="Surrogate : t=ts+k δt",titlefontsize=fs)
    push!(figures,f)
end 
plot(f1,figures...)
savefig("paper/figures/circle.svg")