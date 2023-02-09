using FullWaveformInversion
using Flux
using NeuralOperators
using BSON:@load
using Plots
using Measures
using KissSmoothing

const FWI = FullWaveformInversion

function visualize(vec,sequence,filename)
    theme = :coolwarm
    fs = 8
    figures = []
    for i in sequence
        f = heatmap(vec[i][1,:,:,1],c=theme,title="Surrogate : t=ts+$(i) δt",titlefontsize=fs)
        push!(figures,f)
    end 
    Plots.plot(figures...)
    savefig(filename)
end 

function visualize(matrix::Matrix,filename)
    plt = []
    for i=1:size(matrix,1)
        f = plot(matrix[i,:],label="Prediction",title="Sensor $(i)")
        plot!(denoise(matrix[i,:])[1],label="Smoothed prediction")
        push!(plt,f)
    end 
    plot(plt...,layout=(2,5),size=(1400,750),margin=7.5mm)
    savefig(filename)
end 

function visualize(actual::Array,pred::Vector,saveat,foldername)
    if !isdir(foldername)
        mkpath(foldername)
    end 

    n = length(pred)
    theme = :coolwarm
    fs = 8
    for j in saveat
        f1 = heatmap(actual[:,:,j],c=theme,title="Simulation : t=ts+$(j) δt",titlefontsize=fs)
        f2 = heatmap(pred[j][1,:,:,1],c=theme,title="Surrogate",titlefontsize=fs)
        error = actual[:,:,j] - pred[j][1,:,:,1]
        f3 = heatmap(error,c=theme,title="Error",titlefontsize=fs)
        plot(f1,f2,f3)
        savefig(string(foldername,"$(j).svg"))
    end 
end 

function model_setup(defect_label::String, boundary)
    # Create a folder to visualize
    if !isdir("paper/figures/$(defect_label)/")
        mkpath("paper/figures/$(defect_label)/comparisons/")
    end 

    # Define surrogate and visualize predictions
    surrogate = SurrogateModel(ntsteps,nx,ny,boundary)
    _,predictions = evaluate_verbose(surrogate,Expensive())

    # # Evaluate the timeseries from this model
    # timeseries = evaluate(surrogate,Expensive())

    # Evaluate full model
    _,_,actual = FWI.solver(nx,ny,FWI.TwoD(),boundary)
    actual = actual[:,:,1:70]

    saveat = 1:10:70
    visualize(actual,predictions,saveat,"paper/figures/$(defect_label)/comparisons/")
    # visualize(predictions,saveat,"paper/figures/$(defect_label)/$(defect_label).svg")
    # visualize(timeseries,"paper/figures/$(defect_label)/ts_$(defect_label).svg")
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
    @time model_setup("circle",boundary)
end 

begin
    # Diagonal defect
    boundary = BitMatrix(undef,nx+1,ny+1)
    boundary .= 0
    for i=1:100
        boundary[50+i,50+i]=1
    end 

    # Evaluate and visualize model
    @time model_setup("diagonal",boundary)
end 

begin
    # Composite defect
    boundary = BitMatrix(undef,nx+1,ny+1)
    boundary .= 0
    nxmid = 101
    nymid = 101
    radius = 10
    for i=1:nx
        for j=1:ny
            if sqrt((i-nxmid)^2 + (j-nymid)^2) <= radius 
                boundary[i,j] = 1
            end 
        end 
    end 

    nxmid = 50
    nymid = 101
    radius = 10
    for i=1:nx
        for j=1:ny
            if sqrt((i-nxmid)^2 + (j-nymid)^2) <= radius 
                boundary[i,j] = 1
            end 
        end 
    end 

    nxmid = 101
    nymid = 50
    radius = 10
    for i=1:nx
        for j=1:ny
            if sqrt((i-nxmid)^2 + (j-nymid)^2) <= radius 
                boundary[i,j] = 1
            end 
        end 
    end 

    # Evaluate and visualize model
    @time model_setup("composite",boundary)
end 

begin
    # Square in the domain
    boundary = BitMatrix(undef,nx+1,ny+1)
    boundary .= 0
    xstart = 51
    ystart = 51
    for i=1:50
        for j=1:50
            boundary[xstart+i,ystart+j]=1.0 
        end 
    end 

    # Evaluate and visualize model
    @time model_setup("square",boundary)
end 

begin
    # Square in the domain
    boundary = BitMatrix(undef,nx+1,ny+1)
    boundary .= 0
    boundary[1,:] .= 1
    boundary[end,:] .= 1
    boundary[:,1] .= 1
    boundary[:,end] .= 1 

    # Evaluate and visualize model
    @time model_setup("empty",boundary)
end 