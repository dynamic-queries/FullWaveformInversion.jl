using SparseArrays
using Plots
using OrdinaryDiffEq 

include("utils.jl")

#= Verbose description

This is the simulation of a domain subjected to ultrasonic waves.
The domain is regular and assumed to be rectangular.
The "boundaries of the domain are subjected to homogenous Neumann boundary conditions"
There exist obstacles in the interior of the domain that serves to model the cracks that we want to detect ultimately. 
These obstacles are also assumed to be boundaries and the same boundaey conditions apply here. 
Furthermore, there are "n_{s}" sensors at the top of the domain, that are transducers which actuate the medium in ultrasonic frequency and also serve the purpose of a sensor.
In otherwords, the waves have a source at these sensors and it is the location of these sensors that are our points of interest for measurement.

We make the following assumption to carry out this simulation.
1) The transducers emit waves for t_{emit} (unit)seconds before turning into a receiving mode. This time can be read off the sensor manual that is used for the measurement in the field. 
2) The sources emit constantly. That is, the impulses of the hammer are assumed continuous as opposed to being discrete. 

Assumption 1, helps us decompose the simulation into two parts.
A) Wave Propagation in a 2D medium with multiple sources.
B) Wave Propagation in a 2D medium with an initial condition (output of simulation A).

In addition we make implementational assumptions such as: 
    - Storing the field variables only in the location of the sensors. 
    - Using a fixed time step with a τ that satisfies the Courant Criteron.(Much easier than solving an insanely huge system. (Implicit integrators))
    - 5 point stencil for discretization in 2D. More accurate stencils can be added if the results from this attempt is promising.
=# 


#= Terminology : Simulation parameters

- temit := Time for which the device functions as a transducers
- tsense := Time for which the device functions as a sensor
- t := temit + tsense 
- xmin := minimum of the domain in the x direction
- xmax := maximum of the domain in the x direction 
- ymin := minimum of the domain in the y direction
- ymax := maximum of the domain in the y direction 
- nx := number of internal points in the x direction
- ny := number of internal points in the y direction
- nt := resolution of the time domain
- δx := xstep size
- δy := ystep size
- δt := tstep size
- c := Velocity of wave in that medium
- u := Amplitude of the wave
- v := Velocity of the wave
=# 

#= Units of quantities measured.

- domain - meters
- time - microseconds 
- amplitude - meters
- velocity - meters / microsecond
=# 

# Parameters 
xmin = 0.0
xmax = 1.0 
ymin = 0.0 
ymax = 1.0
tmin = 0.0 # duh! 
temit = 100.0
tsense = 1e3
tsim = temit + tsense
nx = 50
ny = 50
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
# Generate a spline in space
s = spline()

# Mask boundary 
boundary = mask_boundary(nx,ny,s)

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
solution1 = solve(prob1,ORK256(),dt=δt,saveat=tsave,progress=true)
sol1 = Array(solution1)
u1 = sol1[1:n,:]

# # Plot the field 
# anim = @animate for ts=1:size(u1,2)
#     heatmap(reshape(u1[:,ts],(nx,ny)),title="$(ts)")
# end 
# gif(anim,"./gifs/2D/simulation.gif",fps=3)

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
prob = ODEProblem(wave2!,init,tspan,params)
tsave = temit:δt:tsim
solution2 = solve(prob,ORK256(),dt=δt,saveat=tsave)

sol2 = Array(solution2)
u2 = sol2[1:n,:]

# Animate the solution
anim = @animate for i=1:size(u2,2)
    heatmap(reshape(u2[:,i],(nx,ny)),title="$(i)",clim=(minimum(u2),maximum(u2)))
end
gif(anim,"./gifs/2D/simulation_dyn.gif",fps=2)