using OrdinaryDiffEq
using Plots

"""
    This is an attempt to solve the simulation in 1 dimension. 
    Apparently the one dimensional setting works well. 
    What needs to be resolved is transfering the results that we obtain here to two dimension.
"""
function OneD(nx,animate=false)
    nx = 1000
    x = 0.0:(1.0/nx):1.0
    t = 0.0:0.01:0.1
    tend = t[end]:0.01:5.0
    s = exp.(-((x.-0.5)/0.1).^2)
    u = zero(x)
    v = zero(x)
    #
    function wave!(du,u,p,t)
        k = p[1]
        s = p[2]
        n = Int(length(u)/2)
        du[2] = du[1] 
        for i=3:n-2
            du[i] = u[n+i]
            du[n+i] = k*(u[i+1]+u[i-1]-2*u[i]) + s[i]
        end 
        du[n-1] = du[n]
    end 


    init = vcat(u,v)
    δx = x[2]-x[1]
    δt = t[2]-t[1]
    c = 1.0
    k1 = (c/δx)^2
    params = [k1,s]
    tspan = (t[1],t[end])

    prob = ODEProblem(wave!,init,tspan,params)
    sol = solve(prob,Rosenbrock23(),saveat=t)

    if animate
        solution = Array(sol)
        ue = solution[1:length(u),:]
        ve = solution[length(u)+1:end,:]

        anim = @animate for i=1:length(t) 
            plot(x,ue[:,i],label="$(i)",ylim=[-0.1,0.1])
        end 
        gif(anim,"./gifs/1D/trial.gif",fps=10)

        anim = @animate for i=1:length(t) 
            plot(x,ve[:,i],label="$(i)",ylim=[-0.1,0.1])
    end 
    gif(anim,"./gifs/1D/trial_vel.gif",fps=10)
    end 

    function wave2!(du,u,p,t)
        k = p[1]
        s = p[2]
        n = Int(length(u)/2)
        # 
        du[2] = du[1]
        for i=3:n-2
            du[i] = u[n+i]
            du[n+i] = k*(u[i+1]+u[i-1]-2*u[i])
        end  
        du[n-1] = du[n]
    end 

    init2 = sol.u[end]
    tspan = (tend[1],tend[end])
    prob2 = ODEProblem(wave2!,init2,tspan,params)
    sol2 = solve(prob2,Rosenbrock23(),saveat=tend)
    solution2 = Array(sol2)

    if animate 
        ue = solution2[1:length(u),:]
        ve = solution2[length(u)+1:end,:]

        anim = @animate for i=1:length(tend) 
            plot(x,ue[:,i],label="$(i)",ylim=[-0.1,0.1])
        end 
        gif(anim,"./gifs/1D/trialdyn.gif",fps=10)

        anim = @animate for i=1:length(tend) 
            plot(x,ve[:,i],label="$(i)",ylim=[-0.1,0.1])
        end 
        gif(anim,"./gifs/1D/trialdyn_vel.gif",fps=10)
    end 
    sol2 
end

solution = OneD(1000,true)
plot(solution[200,:])