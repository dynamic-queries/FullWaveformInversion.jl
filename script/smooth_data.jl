using HDF5
using Plots

function smooth_defect(nx,ny,mean,σ)

    # Domain
    x = 0.0:1/(nx-1):1.0
    y = 0.0:1/(ny-1):1.0

    # Gaussian function
    g(x,y,mx,my,σ) = exp(-((x-mx)/σ)^2-((y-my)/σ)^2)

    function gaussian_filter(mx,my,σ)
        na,nb = length(x),length(y)
        Z = zeros(na,nb)
        for i=1:na
            for j=1:nb
                Z[i,j] = g(x[i],y[j],mx,my,σ)
            end 
        end 
        Z
    end 

    # Smoothed defect
    D = zeros(Float64,nx,ny)

    # Smoothen defect
    for i=1:nx 
        for j=1:ny
            if mean[i,j] != 0
                D+=gaussian_filter(x[i],y[j],σ)
            end 
        end
    end

    D = D./ maximum(D)
    D
end 

filename = "consolidated/rbf/BOUNDARYs"
file_smooth = h5open(filename,"w")

filename = "consolidated/rbf/BOUNDARY"
file = h5open(filename)

sample = read(file["b1"])
nx, ny = size(sample)
for i=1:floor(Int,length(file)/2)
    D = smooth_defect(nx,ny,Array(read(file["b$(i)"])),0.035)
    file_smooth["b$(i)"] = D
    file_smooth["$(i)"] = read(file["$(i)"])
end 
close(file)