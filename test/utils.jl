using FullWaveformInversion

begin
    nx = 100
    ny = 100
    array = BitArray(undef,nx,ny)
    array .= 0

    x = x = 0.0:0.01:2π
    σ = 1
    K = zeros(length(x),length(x))
    for i=1:length(x)
        for j=1:length(x) 
            K[i,j] = exp(-((x[i]-x[j])^2)/σ^2)
        end 
    end 
    K .= K .+ 1e-6I(length(x)) 
    dist = MvNormal(zero(x),K)
    s = rand(dist,1)[:]
    display(plot(x,s,xlabel="x",ylabel="y",title="Gaussian sample"))
    spline2obstacle!(array,(x,s))
    display(heatmap(array',title="Discrete Embedding"))
end 