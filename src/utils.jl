"""
    Input : nothing
    Output : (x,s) 
"""
function spline()
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
    x,s
end 


"""
    Input: 
        (x,y): Coordinates x_i and y_i of a spline defined in unit domain x ∈ (0,1)
        nx : 
"""
function spline2obstacle!(domain::BitArray,spline::Tuple)
    x,y = spline
    nx,ny = size(domain)
    xmin = minimum(x)
    xmax = maximum(x)

    itp = LinearInterpolation(x,y)
    xdomain = xmin:(xmax-xmin)/(nx-1):xmax
    ydomain = itp.(collect(xdomain))
    
    nhalf = floor(Int,nx/2)
    start = rand(1:nhalf)
    idxobs = start:start+nhalf
    xobs = xdomain[idxobs]
    yobs = ydomain[idxobs]

    ymin = minimum(yobs)
    ymax = maximum(yobs)
    xobs = (xobs .- xmin) ./ (xmax - xmin)
    yobs = (yobs .- ymin) ./ (ymax - ymin)

    xobs = floor.(Int,10 .+ (nx-20).*xobs)
    yobs = floor.(Int,10 .+ (ny-20).*yobs)

    for i=1:length(xobs)
        domain[xobs[i],yobs[i]] = 1
    end 
end 

""" 
    Input: 
        - nx 
        - ny 
        - spline as a tuple 
    Output: BitArray containing the location of the elements to which a Neumann Boundary condition has to be applied.
"""
function mask_boundary(nx::Int,ny::Int,spline=nothing)
    
    boundary = BitArray{2}(undef,nx,ny)
    boundary .= 0
    boundary[1,:] .= 1
    boundary[end,:] .= 1
    boundary[:,1] .= 1
    boundary[:,end] .= 1 

    if !isnothing(spline)
        spline2obstacle!(boundary,spline)
    end 
    boundary
end 

"""
    Input: 
        - x : x domain
        - y : ydomain
        - n : number of sensors/transducers
        - amplitudes: 
        - sds : 
    Output:
        - xsensor : location of the sensors (Vector)
        - source  : SparseArray of source terms. It is assumed a Gaussian
""" 
function compute_source(x::StepRangeLen,y::StepRangeLen,n::Int,amplitudes::Vector,sds::Vector)
    xmin = x[1]
    xmax = x[end]
    xprov = xmin:(xmax-xmin)/(n+1):xmax
    mean = Vector(xprov[2:end-1])
    @assert length(mean) == n
    source = spzeros(length(x),length(y))
    gaussian(x,y,amp,sd,mean) = amp*(exp(-((x-mean)/sd)^2) + exp(-(y/sd)^2))

    for ns in 1:n
        for (i,xi) in enumerate(x) 
            for (j,yj) in enumerate(y)  
                source[i,j] += gaussian(xi,yj,amplitudes[ns],sds[ns],mean[ns])
            end 
        end 
    end 
    mean,source
end 

"""
    Input: 
        - u field 
        - xsensor : x positions of the sensors (ysensors have positions = zero(xsensor))
    
    Output: 
        - u field samples at all locations specified by (xsensor_i,y_sensor_i)
"""
function sample_sensor_reading(ufield,xsensor,Nx,Ny)
    samples = zeros(length(xsensor),size(ufield,2))
    for (i,x) in enumerate(xsensor)
        xcord = floor(Int,(x/1.0)*Nx)
        ycord = Int(3)
        k = (xcord-1)*Ny + ycord
        samples[i,:] .= ufield[k,:]
    end 
    samples
end 
