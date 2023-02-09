using FullWaveformInversion
using Flux
using NeuralOperators
using BSON:@load
using Plots
using Optim
using LinearAlgebra
using HDF5
using CUDA
using FiniteDiff
using ForwardDiff

CUDA.allowscalar(false)

const FWI = FullWaveformInversion

# Load test_case
file = h5open("paper/surrogate/test_case")
defect = read(file["Defect"])
U_actual = read(file["U_actual"])


## This is an RBF constrained optimization
# Define RBF object
x = 0.0:0.005:2π
rbf = RBF(x)
k = rbf.coefficients

# Count
count = 0

# Define a surrogate model
ntsteps = 2
nx = 200
ny = 200
boundary = BitMatrix(undef,nx+1,ny+1)
boundary .= 0
k = 200*k
FWI.spline2obstacle!(boundary,(Vector(x),rbf(k)))

model = SurrogateModel(ntsteps,nx,ny,boundary)

# Compare evaluation times with and without an initial surrogate.
@time solutions_2 = evaluate_verbose(model,FWI.Expensive());
@time solutions_1 = evaluate_verbose(model,FWI.Cheap());

# Compare speed and accuracy
init_1 = solutions_1[1] |> cpu
init_2 = solutions_2[1] |> cpu
error = init_1 - init_2
b = heatmap(boundary,title="Boundary")
f1 = heatmap(init_1,title="Cheap")
f2 = heatmap(init_2,title="Expensive")
f3 = heatmap(error ,title="Error")
plot(b,f1,f2,f3)
savefig("paper/surrogate/test.png")


function forward_model(boundary) 
    global count += 1 
    Upred = evaluate(model,boundary,FWI.Cheap())
end 




# function loss(k)
#     # Update coefficients parameterizing the boundary
#     rbf.coefficients = k
#     @show count
    
#     # Create a boundary
#     y = rbf(k)
#     spline2obstacle!(model.boundary,(Vector(x),y))

#     # Predict outcome
#     Upred = forward_model(model.boundary)
    
#     # L2 loss
#     norm(Upred-U_actual)
# end 

# # Compute loss
# params = copy(k)
# l = loss(params)

# # Setup derivative free optimization
# optimize(loss,params,NelderMead())