using AbstractDifferentiation
using LinearAlgebra
using CUDA
using ForwardDiff

f(x) = x*x*x
g(x) = norm(x*x*x)

# Evaluate gradient
x = CUDA.rand(100,100)


ab = AD.ForwardDiff
@time Jf = ab.jacobian(f, x)