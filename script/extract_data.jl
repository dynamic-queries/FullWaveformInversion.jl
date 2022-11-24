using HDF5

sourcefolder = "data/"
folders = readdir(sourcefolder)

staticfile = h5open("consolidated/static/BOUNDARY","w")
k = 1 
for folder in folders
    subfolders = readdir(string(sourcefolder,folder,"/p_data/"))
    nprocs = length(subfolders)
    ninstances = 20
    for i=1:nprocs
        local_file = h5open(string(sourcefolder,folder,"/p_data/$(i)/static/BOUNDARY"))
        for j=1:ninstances
            init = read(local_file["b$(j)"])
            final  = read(local_file["$(j)"])
            global staticfile["b$(k)"] = init
            global staticfile["$(k)"] = final
            global k = k + 1
        end 
        close(local_file)
    end 
end 
close(staticfile)

k = 1
folders = readdir(sourcefolder)
for folder in folders
    subfolders = readdir(string(sourcefolder,folder,"/p_data/"))
    nprocs = length(subfolders)
    ninstances = 20
    for i=1:nprocs
        for j=1:ninstances
            source_file = h5open(string(sourcefolder,folder,"/p_data/$(i)/dynamic/GROUND_TRUTH$(j)"))
            target_file = h5open("consolidated/dynamic/GROUND_TRUTH$(k)","w")
            for ts = 1:501
                target_file["$(ts)"] = read(source_file["$(ts)"])
            end 
            close(source_file)
            close(target_file)
            global k += 1 
        end 
    end 
end