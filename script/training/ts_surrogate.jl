using FullWaveformInversion

# We use a Markov Neural Operator
# Meaning 
# u_{t-1} is the input.
# u_{t} is the output.

x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0
nx,ny = length(x),length(y)
X = reshape([xi for xi in x for _ in y],(nx,ny))
Y = reshape([yi for _ in x for yi in y],(nx,ny))

batches = 1:1
BS = length(batches)
bs = 250*BS - 1

filename = "./data/dynamic/GROUND_TRUTH"

xdata = Array{Float32,4}(undef,3,nx,ny,bs)
ydata = Array{Float32,4}(undef,1,nx,ny,bs)

print("Reading Data ... \n\n")

for (nb,b) in enumerate(batches)
    file = h5open(string(filename,nb),"r")
    for it=1:bs
        xdata[1,:,:,(nb-1)*bs + it] .= read(file["$(it)"])
        xdata[2,:,:,(nb-1)*bs + it] .= X
        xdata[3,:,:,(nb-1)*bs + it] .= Y
        ydata[1,:,:,(nb-1)*bs + it] .= read(file["$(it+1)"]) 
    end 
end 

train,test = Flux.splitobs((xdata,ydata),at=0.9)
train_loader = Flux.DataLoader(train,batchsize=20,shuffle=true)
test_loader = Flux.DataLoader(test,batchsize=2,shuffle=false)

DL = 16
nmodes = 8
model = MarkovNeuralOperator(ch=(3,DL,DL,DL,DL,DL,1),modes=(nmodes,nmodes), σ = gelu)

data = (train_loader,test_loader)
loss_func = l₂loss

print("Training model... \n\n")

learning_rate = 0.01
nepochs = 50
opt = Flux.ADAM(learning_rate)
learner = Learner(model,data,opt,loss_func,Checkpointer(joinpath(@__DIR__,"ts_checkpoints")))
fit!(learner,nepochs)