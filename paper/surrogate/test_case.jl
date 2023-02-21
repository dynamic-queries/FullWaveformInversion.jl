using HDF5
using Plots
using Measures

# Get the defect
filename = "consolidated/static/BOUNDARY"
file = h5open(filename)
defect = read(file["b1905"])

# Get full waveforms 
filename = "consolidated/dynamic/GROUND_TRUTH1905"
file = h5open(filename)
ntsteps = 70
actual_sols = []
for i=1:ntsteps
    arr = read(file["$(i)"])
    push!(actual_sols,arr)
end


# Sample the full waveforms
xsensors = Vector(5:21:196)
Uactual = zeros(length(xsensors),ntsteps)
for i=1:ntsteps
    Uactual[:,i] = actual_sols[i][xsensors,3]
end 

# Plot functions
function visualize(matrix::Matrix)
    plt = []
    for i=1:size(matrix,1)
        f = plot(matrix[i,:],label="Prediction",title="Sensor $(i)",fontsize=8)
        push!(plt,f)
    end 
    p = plot(plt...,size=(1400,750),margin=7.5mm)
    return p
end 

# Write the actual time series out
filename = "paper/surrogate/test_case"
file = h5open(filename,"w")
file["Defect"] = Array(defect)
file["U_actual"] = Uactual

# Plot the defect and the timeseries - Uactual
f1 = heatmap(defect,c=:thermal,title="Actual Defect")
f2 = visualize(Uactual)
plot(f1,f2)
savefig("paper/figures/TestCase/testcase.svg")