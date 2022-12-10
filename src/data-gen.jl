"""
    Input : Number of instances of the problem that we want to solve.
    Output : Vector of Arrays.
""" 
function generate_data(nx::Int,ny::Int,N::Int)
    print("Generating data ...\n")
    
    solutions = []
    boundaries = []
    sensordata,boundary,sol = solver(nx,ny,TwoD())
    push!(solutions,sol)
    push!(boundaries,boundary)

    pro = Progress(N,1)
    for i=2:N
        _,b,s = solver(nx,ny,TwoD())
        push!(boundaries,b)
        push!(solutions,s)
        next!(pro)
    end 
    sensordata,boundaries,solutions
end 


function generate_data(nx::Int,ny::Int,N::Vector{Int},defects::Vector{AbstractDefect}) 
    print("Generating data ...\n")
    
    solutions = []
    boundaries = []
    for nbtypes = 1:length(N)
        typ = defects[nbtypes]
        push!(boundaries,boundary(nx,ny,typ))
    end 

    for boundary in boundaries
        sensordata,_,sol = solver(nx,ny,TwoD())
        push!(solutions,sol)
    end 
    boundaries,solutions
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
    Input : boundary
    Output : NIL
    Implicit Output : HDF5 file with boundary as a bit vector and the input to the main simulation.
""" 
function boundary_extraction(boundaries::Vector,solutions::Vector,filename::String)
    file = h5open(filename,"w")
    for (i,solution) in enumerate(solutions)
        file["b$(i)"] = Array(boundaries[i])
        file["$(i)"] = Matrix(solution[:,:,1])
    end 
    close(file)
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