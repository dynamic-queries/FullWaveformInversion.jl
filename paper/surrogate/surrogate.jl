include("preamble.jl")

# Munge data to pass to the FNO
input = Array{Float64,4}(undef,3,nx+1,ny+1,1)
input[1,:,:,1] = boundary
input[2,:,:,1] = X
input[3,:,:,1] = Y

# ROM-1
temp = SurA(input)
sur_solutionA = temp[1,:,:,1]

# ROM-2
sur_solutionB = []
input[1,:,:,1] = sur_solutionA
for i=1:ntsteps
    sur = SurBs[i]
    output = sur(input)
    push!(sur_solutionB,output[1,:,:,1])    
end