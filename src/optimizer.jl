using Flux
include("./src/surrogate.jl")

function optimizer(measurements, model, nx=200, ny=200, epochs=100)
    opt = Flux.Optimiser.Adam(0.001, (0.9, 0.8))
    crack_no_init = crack_initialization(nx,ny)  
    params = Flux.params(crack_no_init)
    for epoch in epochs        
        xsensor_no = model(crack_no_init, nx, ny)
        gradients = gradient(ps) do
            loss(xsensor_no, measurements)
        end
        lossmse = loss(xsensor_no, measurements)
        println("Loss at Epoch $epoch is: $lossmse")
        Flux.Optimise.update!(opt, params, gradients)
    end
    crack_no_init
end

function loss(prediction, truth)
    mse(prediction, truth; agg = mean)
end

