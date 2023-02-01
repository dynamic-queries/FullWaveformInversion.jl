function solver(nx::Int,ny::Int,::TwoD,boundary=nothing)
    # Parameters 
    xmin = 0.0
    xmax = 1.0 
    ymin = 0.0 
    ymax = 1.0
    tmin = 0.0 # duh! 
    temit = 100.0
    tsense = 1.25e3
    tsim = temit + tsense
    c = 2e-3
    nsensors = 10

    # Derived quantities
    x = xmin:(xmax-xmin)/nx:xmax
    y = ymin:(ymax-ymin)/ny:ymax 
    δx = x[2]-x[1]
    δy = y[2]-y[1] 
    δt = sqrt(δx^2/c^2)
    t = tmin:δt:tsim

    # Arrays for the simulation
    u = spzeros(nx+1,ny+1)
    v = spzeros(nx+1,ny+1)

    # Exterior boundaries
    nx = length(x)
    ny = length(y)

    u[:,2] .= u[:,1]
    u[:,end-1] .= u[:,end] 
    u[2,:] .= u[1,:]
    u[end-1,:] .= u[end,:] 

    v[:,2] .= v[:,1]
    v[:,end-1] .= v[:,end] 
    v[2,:] .= v[1,:]
    v[end-1,:] .= v[end,:] 

    # Bit Array for the boundary
    if isnothing(boundary)
        # Generate a spline in space
        s = spline()
        # Mask boundary 
        boundary = mask_boundary(nx,ny,s)
    end
    # Compute the source term 
    amplitudes = 1e-5*ones(Float64,nsensors)
    deviations = ones(Float64,nsensors)
    xsensors,source = compute_source(x,y,nsensors,amplitudes,deviations)

    # Setup the inital simulation 
    init = vcat(u[:],v[:])
    tspan = (tmin,temit)
    n = length(u)
    params = [(c/δx)^2,(c/δy)^2,nx,ny,source[:],boundary[:],n]
    tsave = tmin:δt:temit 

        # Forcing function with boundary handing
    function wave!(du,u,p,t)
        k1 = p[1]
        k2 = p[2] 
        nx = p[3]
        ny = p[4]
        source = p[5]
        boundary = p[6]
        n = p[7]

        for j=3:ny-2
            for i=3:nx-2
                k = (j-1)*ny + i
                k̂ = k + n 
                K = [k-1,k+1,k-ny,k+ny]

                if boundary[k]==1 
                    u[k]=0
                    u[k̂]=0
                    du[k]=0
                    du[k̂]=0
                elseif sum(isone.(boundary[K]))>0
                    for l in K 
                        if boundary[l] == 1 
                            du[k] = 0
                            du[k̂] = 0
                            u[k] = u[l]
                            u[k̂] = u[l+n]
                        end
                    end
                else
                @inbounds du[k] = u[k̂]
                @inbounds du[k̂] = k1*(u[K[1]]+u[K[2]]-2*u[k]) + k2*(u[K[3]]+u[K[4]]-2*u[k]) + source[k]
                end
            end 
        end 
    end 

    # Solve the problem using time integration
    prob1 = ODEProblem(wave!,Array(init),tspan,params)
    solution1 = OrdinaryDiffEq.solve(prob1,ORK256(),dt=δt,saveat=tsave,progress=true)

    function wave2!(du,u,p,t)
        k1 = p[1]
        k2 = p[2] 
        nx = p[3]
        ny = p[4]
        source = p[5]
        boundary = p[6]
        n = p[7]
    
        for j=3:ny-2
            for i=3:nx-2
                k = (j-1)*ny + i
                k̂ = k + n 
                K = [k-1,k+1,k-ny,k+ny]
    
                if boundary[k]==1 
                    u[k]=0
                    u[k̂]=0
                    du[k]=0
                    du[k̂]=0
                elseif sum(isone.(boundary[K]))>0
                    for l in K 
                        if boundary[l] == 1 
                            du[k] = 0
                            du[k̂] = 0
                            u[k] = u[l]
                            u[k̂] = u[l+n]
                        end
                    end
                else
                @inbounds du[k] = u[k̂]
                @inbounds du[k̂] = k1*(u[K[1]]+u[K[2]]-2*u[k]) + k2*(u[K[3]]+u[K[4]]-2*u[k])
                end
            end 
        end
    end 
    
    init = solution1.u[end]
    tspan = (temit,tsim)
    prob = OrdinaryDiffEq.ODEProblem(wave2!,init,tspan,params)
    tsave = temit:δt:tsim
    solution2 = OrdinaryDiffEq.solve(prob,ORK256(),dt=δt,saveat=tsave)

    # Munge the data
    sol2 = Array(solution2)
    u = reshape(sol2[1:n,:],(nx,ny,:))
    v = reshape(sol2[n+1:end,:],(nx,ny,:))
    
    (xsensors,nx,ny),reshape(boundary,(nx,ny)),u
end 

function regularizer_N_input(u::Array,xsensor,nx,ny)
    u = reshape(u,(nx*ny,:))
    sample_sensor_reading(u,xsensor,nx,ny)
end 

function solver_init(nx::Int,ny::Int,::TwoD,boundary=nothing)
    # Parameters 
    xmin = 0.0
    xmax = 1.0 
    ymin = 0.0 
    ymax = 1.0
    tmin = 0.0 # duh! 
    temit = 100.0
    tsense = 1.25e3
    tsim = temit + tsense
    c = 2e-3
    nsensors = 10

    # Derived quantities
    x = xmin:(xmax-xmin)/nx:xmax
    y = ymin:(ymax-ymin)/ny:ymax 
    δx = x[2]-x[1]
    δy = y[2]-y[1] 
    δt = sqrt(δx^2/c^2)
    t = tmin:δt:tsim

    # Arrays for the simulation
    u = spzeros(nx+1,ny+1)
    v = spzeros(nx+1,ny+1)

    # Exterior boundaries
    nx = length(x)
    ny = length(y)

    u[:,2] .= u[:,1]
    u[:,end-1] .= u[:,end] 
    u[2,:] .= u[1,:]
    u[end-1,:] .= u[end,:] 

    v[:,2] .= v[:,1]
    v[:,end-1] .= v[:,end] 
    v[2,:] .= v[1,:]
    v[end-1,:] .= v[end,:] 

    # Bit Array for the boundary
    if isnothing(boundary)
        # Generate a spline in space
        s = spline()
        # Mask boundary 
        boundary = mask_boundary(nx,ny,s)
    end
    # Compute the source term 
    amplitudes = 1e-5*ones(Float64,nsensors)
    deviations = ones(Float64,nsensors)
    xsensors,source = compute_source(x,y,nsensors,amplitudes,deviations)

    # Setup the inital simulation 
    init = vcat(u[:],v[:])
    tspan = (tmin,temit)
    n = length(u)
    params = [(c/δx)^2,(c/δy)^2,nx,ny,source[:],boundary[:],n]
    tsave = tmin:δt:temit 

        # Forcing function with boundary handing
    function wave!(du,u,p,t)
        k1 = p[1]
        k2 = p[2] 
        nx = p[3]
        ny = p[4]
        source = p[5]
        boundary = p[6]
        n = p[7]

        for j=3:ny-2
            for i=3:nx-2
                k = (j-1)*ny + i
                k̂ = k + n 
                K = [k-1,k+1,k-ny,k+ny]

                if boundary[k]==1 
                    u[k]=0
                    u[k̂]=0
                    du[k]=0
                    du[k̂]=0
                elseif sum(isone.(boundary[K]))>0
                    for l in K 
                        if boundary[l] == 1 
                            du[k] = 0
                            du[k̂] = 0
                            u[k] = u[l]
                            u[k̂] = u[l+n]
                        end
                    end
                else
                @inbounds du[k] = u[k̂]
                @inbounds du[k̂] = k1*(u[K[1]]+u[K[2]]-2*u[k]) + k2*(u[K[3]]+u[K[4]]-2*u[k]) + source[k]
                end
            end 
        end 
    end 

    # Solve the problem using time integration
    prob1 = ODEProblem(wave!,Array(init),tspan,params)
    solution1 = OrdinaryDiffEq.solve(prob1,ORK256(),dt=δt,saveat=tsave,progress=true)
    sol = Array(solution1)
    sol[end]
end 

mutable struct RBF
    basis
    coefficients
    
    function RBF(x;σ=1.0)
        f(x,y) = exp(-((x-y)^2)/σ^2)
        n = length(x)
        K = zeros(n,n)
        for i=1:n
            for j=1:n
                K[i,j] = f(x[i],x[j])
            end 
        end
        basis = [K[i,:] for i=1:n]
        k = rand(n)
        new(basis,k)
    end 
end 

function Base.show(io::IO,rbf::RBF)
    print(io,"RBF")
end 

function (b::RBF)(k)
    b.coefficients = k
    n = length(k)
    Z = zero(k)
    for i=1:n
        Z = Z + b.coefficients[i] * b.basis[i]
    end 
    Z
end 

function random_rbf_defect()
    x = 0.0:0.005:2π
    rbf = RBF(x)
    k = rbf.coefficients
    x,y = x,rbf(k)
    guess_boundary = BitMatrix(undef,nx+1,ny+1)
    guess_boundary .= 0
    FWI.spline2obstacle!(guess_boundary,(x,y))
    guess_boundary
end

function data_gen_init(nx,ny,ninstances,filename)
    file = h5open(filename,"w")
    pro = Progress(ninstances,1)
    for i=1:ninstances
        bound = random_rbf_defect()
        sol = solver_init(nx,ny,TwoD(),bound)      
        file["b$(i)"] = Array(bound)
        file["$(i)"] = sol
        next!(pro)
    end
    close(file) 
end 