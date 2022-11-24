# Test script for diagonal boundary
boundary = BitMatrix(undef,nx+1,ny+1)
boundary .= 0
nxmid = 101
nymid = 101
radius = 40
for i=50:nx-50
    for j=50:ny-50
        if i==j 
            boundary[i,j] = 1
        end 
    end 
end 
boundary_name = "circle"
FullWaveformInversion.standalone_test(boundary,boundary_name)