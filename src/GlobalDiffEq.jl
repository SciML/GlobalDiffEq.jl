module GlobalDiffEq

using Reexport: @reexport
@reexport using SciMLBase

using Richardson: extrapolate
using PrecompileTools: @setup_workload, @compile_workload
using SciMLBase: ODEProblem, __solve, AbstractODEProblem, AbstractDAEProblem
using CommonSolve: solve, 

abstract type GlobalDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

struct GlobalRichardson{A} <: GlobalDiffEqAlgorithm
    alg::A
end

function __solve(
        prob::Union{AbstractODEProblem, AbstractDAEProblem},
        alg::GlobalRichardson, args...;
        dt, kwargs...)
    opt = Dict(kwargs)
    otheropts = delete!(copy(opt), :dt)
    tstops = get(opt, :tstops, range(prob.tspan[1], stop = prob.tspan[2], step = dt))
    local sol
    val,
    err = extrapolate(dt, rtol = get(opt, :reltol, 1e-3),
        atol = get(opt, :abstol, 1e-6), contract = 0.5) do _dt
        sol = solve(prob, alg.alg, args...; dt = _dt, adaptive = false, otheropts...)
        # Convert Vector{Vector{T}} to Matrix{T} for Richardson.jl compatibility
        reduce(hcat, sol.(tstops))
    end
    return sol
end

export GlobalRichardson

@setup_workload begin
    # Simple test ODE: exponential decay du/dt = -u
    function f!(du, u, p, t)
        du[1] = -u[1]
    end
    u0 = [1.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(f!, u0, tspan)

    @compile_workload begin
        # Precompile with SSPRK33 (commonly used explicit method)
        #solve(prob, GlobalRichardson(OrdinaryDiffEq.SSPRK33()),
        #    dt = 0.1, reltol = 1e-3, abstol = 1e-6)
    end
end

end
