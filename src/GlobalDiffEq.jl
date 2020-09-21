module GlobalDiffEq

using Reexport
@reexport using DiffEqBase

import OrdinaryDiffEq, Richardson

abstract type GlobalDiffEqAlgorithm <: DiffEqBase.AbstractODEAlgorithm end

struct GlobalRichardson{A} <: GlobalDiffEqAlgorithm
    alg::A
end

function DiffEqBase.__solve(prob::Union{DiffEqBase.AbstractODEProblem,DiffEqBase.AbstractDAEProblem},
                            alg::GlobalRichardson, args...;
                            dt,kwargs...)
    opt = Dict(kwargs)
    otheropts = delete!(copy(opt), :dt)
    tstops = get(opt, :tstops, range(prob.tspan[1], stop=prob.tspan[2], step=dt))
    local sol
    val, err = Richardson.extrapolate(dt, rtol=get(opt, :reltol, 1e-3), atol=get(opt, :abstol, 1e-6), contract=0.5) do _dt
        sol = solve(prob, alg.alg, args...; dt=_dt, adaptive=false, otheropts...)
        sol.(tstops)
    end
    return sol
end

export GlobalRichardson

end
