include("solvers.jl")

"""
    Input : Number of instances of the problem that we want to solve.
    Output : Vector of Arrays.
""" 
function generate_data(nx::Int,ny::Int,N::Int)
    print("Generating data ...\n")
    
    solutions = []
    sensordata,sol = solver(nx,ny,TwoD())
    push!(solutions,sol)
    
    pro = Progress(N,1)
    for i=2:N
        push!(solutions,solver(nx,ny,TwoD())[2])
        next!(pro)
    end 
    sensordata,solutions
end 

""" 
    Input : Vector of Arrays
    Output : NIL
    Implicit Output : HDF5 file that contains time data for the pressure over the whole domain.
""" 
function data_extraction(solutions::Vector,filename::String)
    for (f,solution) in enumerate(solutions)
        file = h5open(string(filename,f),"w")
        ts = size(solution,3)
        for i=1:ts
            file["$(i)"] = Matrix(solution[:,:,i])
        end 
        close(file)
    end 
end 

"""
    Input : Arrays
    Output : Time series for the value at all the sensors.
"""
function samplesensor(sensordata,solution::Array)
    ts = size(solution,3)
    xsensor,nx,ny = sensordata
    sol = reshape(solution,(:,ts))
    series = sample_sensor_reading(sol,xsensor,nx,ny)
    series
end 
