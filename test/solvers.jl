using FullWaveformInversion

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
plot(input_reg',layout=10)
savefig("test/vis/sensors.png")

heatmap(boundary,title="Boundary")
savefig("test/vis/boundary.png")

heatmap(u[:,:,1],title="t_{emit}")
savefig("test/vis/initial.png")

heatmap(u[:,:,end],title="t_{end}")
savefig("test/vis/final.png")