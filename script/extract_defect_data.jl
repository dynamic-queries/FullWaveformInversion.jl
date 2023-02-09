using HDF5

source = "data/rbf/"
destination = "consolidated/rbf/"

nranks = 24
file = h5open(string(destination,"BOUNDARY"),"w")
it = 1
for rank=1:nranks
    file_local =  h5open(string(source,"$(rank)/BOUNDARY"),"r")
    n = floor(Int,length(file_local)/2)
    for j=1:n
        global file["$(it)"] =  read(file_local["$(j)"])
        global file["b$(it)"] = read(file_local["b$(j)"])
        global it += 1
    end 
    close(file_local)
end 
close(file)