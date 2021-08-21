module GlobalDiffEq

using Reexport
@reexport using DiffEqBase

import Distributions, LinearAlgebra, ModelingToolkit, OrdinaryDiffEq, Richardson

include("vector_expectation.jl")
include("adjoint_sol_stepsize.jl")

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

struct GlobalAdjoint{A} <: GlobalDiffEqAlgorithm
    alg::A
end

function DiffEqBase.__solve(prob::Union{DiffEqBase.AbstractODEProblem,DiffEqBase.AbstractDAEProblem},
                            alg::GlobalAdjoint, args...;
                            dt,kwargs...)

    opt = Dict(kwargs)
    otheropts = delete!(copy(opt), :dt)
    tstops = get(opt, :tstops, range(prob.tspan[1], stop=prob.tspan[2], step=dt))
    local sol

    sol_init = solve(prob, alg.alg, args...; dt=dt, otheropts...)
    n = length(prob.u0)
    C = 11/6 # for k = 2, 2 random vectors
    E₂ = 2/π
    Eₙ = vec_expt_full(n)
    rtol = get(opt, :reltol, 1e-3)
    atol = get(opt, :abstol, 1e-6)
    gtol = atol # target gtol is abstol? abstol cannot be zero?
    z₁, z₂ = orth_vec(n)
    
    t_ctrl = tstops[1]
    tstops_ctrl = [t_ctrl]
    λ₁ = adjoint_solve(prob, sol_init, z₁, tstops, prob.p, rtol, atol)
    λ₂ = adjoint_solve(prob, sol_init, z₂, tstops, prob.p, rtol, atol)
    while t_ctrl < tstops[end]
        h = adjoint_step_ctrl(λ₁(t_ctrl), λ₂(t_ctrl), E₂, Eₙ, C, gtol)
        t_ctrl += h
        push!(tstops_ctrl, t_ctrl)
    end
    push!(tstops_ctrl, tstops[end])

    sol = solve(prob, alg.alg, args...; dt=dt, tstops=tstops_ctrl, adaptive=false, otheropts...) # control stepsize or LTOL?
    return sol
end

export GlobalRichardson, GlobalAdjoint

end
