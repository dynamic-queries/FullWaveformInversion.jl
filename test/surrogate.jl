using FullWaveformInversion
using Flux
using NeuralOperators
using BSON:@load
using Plots
using Measures

function visualize(vec,sequence,filename)
    theme = :coolwarm
    fs = 8
    figures = []
    for i in sequence
        f = heatmap(vec[i],c=theme,title="Surrogate : t=ts+$(i) δt",titlefontsize=fs)
        push!(figures,f)
    end 
    Plots.plot(figures...)
    savefig(filename)
end 

function visualize(matrix::Matrix,filename)
    plt = []
    for i=1:size(matrix,1)
        f = plot(matrix[i,:],label="Prediction",title="Sensor $(i)")
        push!(plt,f)
    end 
    plot(plt...,layout=(2,5),size=(1400,750),margin=7.5mm)
    savefig(filename)
end 

function model(defect_label::String, boundary)
    # Create a folder to visualize
    if !isdir("paper/figures/$(defect_label)/")
        mkpath("paper/figures/$(defect_label)/")
    end 

    # Define surrogate and visualize predictions
    surrogate = SurrogateModel(ntsteps,nx,ny,boundary)
    predictions = evaluate_verbose(surrogate)
    saveat = 1:10:70
    visualize(predictions,saveat,"paper/figures/$(defect_label)/$(defect_label).svg")

    # Evaluate the timeseries from this model
    timeseries = evaluate(surrogate)
    visualize(timeseries,"paper/figures/$(defect_label)/ts_$(defect_label).svg")
end

# Setup surrogate
ntsteps = 70
nx = 200
ny = 200

begin
    # Circle-in-the-middle defect
    boundary = BitMatrix(undef,nx+1,ny+1)
    boundary .= 0
    nxmid = 101
    nymid = 101
    radius = 40
    for i=1:nx
        for j=1:ny
            if sqrt((i-nxmid)^2 + (j-nymid)^2) <= radius 
                boundary[i,j] = 1
            end 
        end 
    end 

    # Evaluate and visualize model
    @time model("circle",boundary)
end 