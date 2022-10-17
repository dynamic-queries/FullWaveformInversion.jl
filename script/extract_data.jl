using HDF5

# Split up data files
input_foldername = "/tmp/ge96gak/p_data/"
target_foldername = "/tmp/ge96gak/consolidated_data/"

nprocesses = 30
nproblems = 25

staticfilename = string(target_foldername,"static/BOUNDARY")
file = h5open(staticfilename,"w")
k = 1
for i=1:nprocesses
    local_file = h5open(string(input_foldername,i,"/static/BOUNDARY"))
    for j=1:nproblems
        init_data = read(local_file["b$(j)"])
        final_data = read(local_file["$(j)"])
        global file["b$(k)"] = init_data
        global file["$(k)"] = final_data
        global k = k+1
    end 
    close(local_file) 
end 
close(file)


dynamic_filename = string(target_foldername,"dynamic/")
k = 1
for i=1:nprocesses
    for j=1:nproblems
        file = h5open(string(dynamic_filename,"GROUND_TRUTH$(k)"),"w")
        local_file = h5open(string(input_foldername,i,"/dynamic/GROUND_TRUTH$(j)"))
        for t in 1:501
            global file["$(t)"] = read(local_file["$(t)"])
        end 
        close(local_file)
        close(file)
        global k = k + 1
    end 
end 