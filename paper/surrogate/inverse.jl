include("plots.jl")
include("rbf.jl")
using LinearAlgebra

# Defects in matrix form and their corresponding parameteric versions
true_boundary = boundary
guess_boundary = pboundary

# Show the relative difference between the two defects
h1 = heatmap(true_boundary)
h2 = heatmap(guess_bounary)
plot(h1,h2)

# Actual data
TS_actual = URO

# Fix the basis of the RBF kernel that has to be used for obtaining the boundaries
n = 100
x = 0.0:(1.0/(n-1)):1.0
rbf = RBF(x)

# Define loss function
function eval(params)
    spline = rbf(params)
    boundary = lift(spline,nx,ny)
    input[1,:,:,1] = boundary
    temp = SurA(input)
    input[1,:,:,1] = temp[1,:,:,1]
    ts = zero(TS_actual)
    for i=1:ntsteps
        sur = SurBs[i]
        output = sur(input)[1,:,:,1]
        ts[:,i] = output[xsensors,3]    
    end 
    norm(TS_actual - ts)
end

# Evaluate loss function
error = eval(approx.coefficients)

# Compute the gradient of the loss function
