module GlobalDiffEq

using Reexport
@reexport using DiffEqBase

import DiffEqSensitivity, Distributions, LinearAlgebra, OrdinaryDiffEq, Richardson, Zygote

include("vector_expectation.jl")

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

    n = length(prob.u0)
    k = 2 # 2 random vectors
    C = 11/6 # for k = 2
    E₂ = 2/π
    Eₙ = vec_expt_full(n)
    gtol_get = get(opt, :abstol, 1e-6) # target gtol is abstol? abstol cannot be zero?
    z₁, z₂ = orth_vec(n)

    function adjoint_loss_z₁(u0, p)
        # g(u), reverse-mode adjoint
        _prob = remake(prob, u0=u0, p=p)
        _sol = solve(_prob, alg.alg; dt=dt, sensealg=DiffEqSensitivity.QuadratureAdjoint())
        z₁' * _sol.u[end]
    end

    function adjoint_loss_z₂(u0, p)
        # g(u), reverse-mode adjoint
        _prob = remake(prob, u0=u0, p=p)
        _sol = solve(_prob, alg.alg; dt=dt, sensealg=DiffEqSensitivity.QuadratureAdjoint())
        z₂' * _sol.u[end]
    end

    # get sensitivities with respect to p at each t
    # only iterate over tstops if p is a function of t
    is_callable(f) = !isempty(methods(f))
    if is_callable(prob.p)
        p_range = prob.p.(tstops)
        λ₁ = Vector(undef, length(tstops)) # typedef?
        λ₂ = Vector(undef, length(tstops)) # typedef?
        for (i, p) ∈ enumerate(p_range) # this does not work because the problem still expects a function for p
            λ₁[i] = Zygote.gradient(adjoint_loss_z₁, prob.u0, p)[1] # u0 sensitivity
            λ₂[i] = Zygote.gradient(adjoint_loss_z₂, prob.u0, p)[1]
        end
    else
        λ₁ = fill(Zygote.gradient(adjoint_loss_z₁, prob.u0, prob.p)[1], length(tstops)) # u0 sensitivity
        λ₂ = fill(Zygote.gradient(adjoint_loss_z₂, prob.u0, prob.p)[1], length(tstops))
    end
    K₁ = LinearAlgebra.norm.(λ₁, 1) .+ LinearAlgebra.norm(λ₁[1])
    K₂ = LinearAlgebra.norm.(λ₂, 1) .+ LinearAlgebra.norm(λ₂[1])
    K = (E₂ / Eₙ) * @. sqrt(K₁^2 + K₂^2)
    h = (gtol_get / (K * C)).^(1/k)
    h_stops_len = length(h)+1
    h_stops = Vector(undef, h_stops_len) # typedef?
    h_stops[1] = prob.tspan[1]
    for i ∈ 2:h_stops_len
        h_stops[i] = h_stops[i-1] + h[i-1]
    end

    sol = solve(prob, alg.alg, args...; dt=maximum(h_stops), tstops=h_stops, adaptive=false, otheropts...)
    return sol
end

export GlobalRichardson, GlobalAdjoint

end
