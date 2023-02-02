include("preamble.jl")

# Munge data to pass to the FNO
input = Array{Float64,4}(undef,3,nx+1,ny+1,1)
input[1,:,:,1] = boundary
input[2,:,:,1] = X
input[3,:,:,1] = Y

# Solve wave propagation with a circle
boundary = BitMatrix(undef,nx+1,ny+1)
boundary .= 0
xstart = 51
ystart = 51
for i=1:50
    for j=1:50
        boundary[xstart+i,ystart+j]=1.0 
    end 
end 
heatmap(boundary,title="Square Boundary")
savefig("figures/square_boundary.svg")

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
savefig("paper/figures/square.svg")