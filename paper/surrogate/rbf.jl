using LinearAlgebra
using ForwardDiff
using Plots
using Optimization
using OptimizationOptimJL

# Radial basis struct
mutable struct RBF
    basis
    coefficients
    
    function RBF(x;σ=1.0)
        f(x,y) = exp(-((x-y)^2)/σ^2)
        n = length(x)
        K = zeros(n,n)
        for i=1:n
            for j=1:n
                K[i,j] = f(x[i],x[j])
            end 
        end
        basis = [K[i,:] for i=1:n]
        k = rand(n)
        new(basis,k)
    end 
end 

function Base.show(io::IO,rbf::RBF)
    print(io,"RBF")
end 

function (b::RBF)(k)
    b.coefficients = k
    n = length(k)
    Z = zero(k)
    for i=1:n
        Z = Z + b.coefficients[i] * b.basis[i]
    end 
    Z
end 

# Tests
function test(k)
    x = 0.0:k:2.0
    ydata = sin.(6*x)
    rbf = RBF(x)

    # Optimization Loop
    coeff_guess = rand(length(x))

    function loss(k,p)
        norm(rbf(k) - ydata)
    end 

    func = OptimizationFunction(loss,Optimization.AutoForwardDiff())
    prob = OptimizationProblem(func,coeff_guess,[])
    @time sol = solve(prob,LBFGS())
    f = plot(rbf(sol.u),label="Simulated",title="$(k)")
    plot!(ydata,label="Actual")
    display(f)
end 