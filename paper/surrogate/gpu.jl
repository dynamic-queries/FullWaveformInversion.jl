using CUDA
using BSON:@load
using Flux
using NeuralOperators
using FullWaveformInversion
using Plots

SurBs = []
filename = "best/os/"
ntsteps = 70
for i=1:ntsteps
    @load string(filename,"$(i)") modeltemp
    push!(SurBs,modeltemp)
end 

nx = 200
ny = 200
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

x = 0.0:(1.0)/nx:1.0
y = 0.0:(1.0)/ny:1.0
X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

temp = solver_init(nx,ny,TwoD(),boundary)
input = CUDA.zeros(3,nx+1,ny+1,1)
input[1,:,:,1] = temp
input[2,:,:,1] = X
input[3,:,:,1] = Y

output = CUDA.zeros(1,nx+1,ny+1,1)

outputs = []

for i=1:70
    sur = SurBs[i]
    sur = sur |> gpu
    output = sur(input)
    push!(outputs,output|>cpu)
end 

# figs = []
# for i=1:10:70
#     f = heatmap(outputs[i][1,:,:,1],c=:coolwarm,title="Time step = $(i)",fontsize=8)
#     push!(figs,f)
# end 
# plot(figs...)
# savefig("paper/surrogate/gpu.svg")