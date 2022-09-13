include("./src/solvers.jl")
include("./src/utils.jl")

using Flux

# Define boundary extents
nx = 200
ny = 200

# Generate random defect / If you feel like it, initialize a random bit matrix as well.
s = spline()
A = BitMatrix(undef,nx+1,ny+1)
A .= 0
spline2obstacle!(A,s)

# Solve problem
(xsensors,_,_),boundary,u = solver(nx,ny,TwoD(),A)

# Get the input matrix for the reg. network.
# Note : M \in  \Re^{nsensors,nt}
input_reg = regularizer_N_input(u,xsensors,nx+1,ny+1)


measurements=add_noise(input_reg,0,1e-3)



plot(input_reg',layout=10)
savefig("test/vis/sensors.png")


plot(measurements', layout=10)
savefig("test/vis/sensors_no.png")



W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

θ = Flux.params(W, b)
grads = gradient(() -> loss(x, y), θ)

