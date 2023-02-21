using HDF5
using Plots
using FFTW
using Measures

filename = "consolidated/dynamic/GROUND_TRUTH"
ninstances = 1920
ntimessteps = 70

D = Array{ComplexF64,4}(undef,ninstances,ntimessteps,12,12)

for i=1:ninstances
    file = h5open(string(filename,"$(i)"),"r")
    for j=1:ntimessteps
        data = read(file["$(j)"])
        modes = rfft(data)[1:12,1:12]
        D[i,j,:,:] .= modes
    end 
end 


P = Array{ComplexF64,4}(undef,5,ntimessteps,12,12)
filenames = ["circle","composite","diagonal","empty","square"]
for (l,name) in enumerate(filenames)
    file = h5open(string("unseen/$(name)/$(name)"))
    for ts = 1:ntimessteps
        data = rfft(read(file["$(ts)"]))[1:12,1:12]
        P[l,ts,:,:] .= data
    end 
end 

# Generate images
colors = [:red,:blue,:green,:black,:yellow]
filenames = ["circle","composite","diagonal","empty","square"]

for i=1:12
    for j=1:12 
        k = (i-1)*12+j
        filename = "analysis/modes/mode_#$(k).svg"
        f = D[1,:,i,j]
        fig1 = plot(abs.(f),linecolor=:black,linealpha=0.025,xlabel="t (in time steps)",ylabel="|R_$(k)|",margin=20mm)
        for p=2:ninstances
            f = D[p,:,i,j]
            plot!(abs.(f),linecolor=:black,linealpha=0.025)
        end 
        for l=1:5
            f = P[l,:,i,j]
            plot!(abs.(f),linecolor=colors[l],label="$(filenames[l])")
        end 
        plot(fig1,title="Mode $(k)",legend=false)
        savefig(filename)
    end 
end 