module GlobalDiffEq

using Reexport
@reexport using DiffEqBase

import OrdinaryDiffEq, Richardson, SciMLBase
using PrecompileTools

abstract type GlobalDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

struct GlobalRichardson{A} <: GlobalDiffEqAlgorithm
    alg::A
end

# Forward algorithm traits to the wrapped algorithm
# This allows GlobalRichardson to inherit capabilities from the inner algorithm
SciMLBase.allows_arbitrary_number_types(alg::GlobalRichardson) =
    SciMLBase.allows_arbitrary_number_types(alg.alg)
SciMLBase.allowscomplex(alg::GlobalRichardson) =
    SciMLBase.allowscomplex(alg.alg)
SciMLBase.isautodifferentiable(alg::GlobalRichardson) =
    SciMLBase.isautodifferentiable(alg.alg)

function DiffEqBase.__solve(
        prob::Union{DiffEqBase.AbstractODEProblem, DiffEqBase.AbstractDAEProblem},
        alg::GlobalRichardson, args...;
        dt, kwargs...
    )
    opt = Dict(kwargs)
    otheropts = delete!(copy(opt), :dt)
    tstops = get(opt, :tstops, range(prob.tspan[1], stop = prob.tspan[2], step = dt))
    local sol
    val,
        err = Richardson.extrapolate(
        dt, rtol = get(opt, :reltol, 1.0e-3),
        atol = get(opt, :abstol, 1.0e-6), contract = 0.5
    ) do _dt
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
        solve(
            prob, GlobalRichardson(OrdinaryDiffEq.SSPRK33()),
            dt = 0.1, reltol = 1.0e-3, abstol = 1.0e-6
        )
    end
end

end
