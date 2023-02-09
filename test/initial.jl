using HDF5
using Plots
using BSON
using NeuralOperators
using Flux
using CUDA
using FullWaveformInversion

const FWI = FullWaveformInversion

# 
function smooth_defect(nx,ny,mean,σ)

    # Domain
    x = 0.0:1/(nx-1):1.0
    y = 0.0:1/(ny-1):1.0

    # Gaussian function
    g(x,y,mx,my,σ) = exp(-((x-mx)/σ)^2-((y-my)/σ)^2)

    function gaussian_filter(mx,my,σ)
        na,nb = length(x),length(y)
        Z = zeros(na,nb)
        for i=1:na
            for j=1:nb
                Z[i,j] = g(x[i],y[j],mx,my,σ)
            end 
        end 
        Z
    end 

    # Smoothed defect
    D = zeros(Float64,nx,ny)

    # Smoothen defect
    for i=1:nx 
        for j=1:ny
            if mean[i,j] != 0
                D+=gaussian_filter(x[i],y[j],σ)
            end 
        end
    end

    D = D./ maximum(D)
    D
end 

# 
filename = "consolidated/rbf/BOUNDARYs"
file = h5open(filename,"r")

# 
boundary = read(file["b400"])
pred = read(file["400"])

# 
nx = 200
ny = 200
x = 0.0:(1.0)/nx:1.0
y = 0.0:(1.0)/ny:1.0
X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

# 
input = CuArray{Float64,4}(undef,3,nx+1,ny+1,1)
input[1,:,:,1] = boundary
input[2,:,:,1] = X
input[3,:,:,1] = Y

# 
BSON.@load "best/is/is_smooth" modeltemp
sur = modeltemp |> gpu
output = sur(input) |> cpu

# 
b = heatmap(boundary,title="Defect")
p1 = heatmap(output[1,:,:,1],title="Prediction")
p2 = heatmap(pred,title="Actual")
plot(b,p1,p2)
savefig("paper/figures/initials/initial.png")

# Other boundary
for i=1:30
    nx = 200
    ny = 200
    boundary = BitMatrix(undef,nx+1,ny+1)
    boundary .= 0
    xs = Vector(0.0:0.005:2π)
    rbf = FWI.RBF(xs)
    k = rbf.coefficients
    FWI.spline2obstacle!(boundary,(xs,rbf(k)))
    nx,ny = size(boundary)
    sigma = 0.035
    boundary = smooth_defect(nx,ny,boundary,sigma)

    input[1,:,:,1] = boundary
    output = sur(input) |> cpu

    # 
    b = heatmap(boundary,title="Defect")
    p1 = heatmap(output[1,:,:,1],title="Prediction")
    plot(b,p1)
    savefig("paper/figures/initials/randoms/initial_random_$(i).png")
end 