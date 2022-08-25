include("../src/solvers.jl")

# Define boundary extents
nx = 200
ny = 200

# Generate random defect / If you feel like it, initialize a random bit matrix as well.
s = spline()
A = BitMatrix(undef,nx+1,ny+1)
A .= 0
spline2obstacle!(A,s)

# Solve problem
(xsensors,_,_),_,u = solver(nx,ny,TwoD(),A)

# Get the input matrix for the reg. network.
# Note : M \in  \Re^{nsensors,nt}
input_reg = regularizer_N_input(u,xsensor,nx,ny)