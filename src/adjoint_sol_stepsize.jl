function adjoint_solve(sol_init, rtol, atol, dg, tstops)
    adj_prob = DiffEqSensitivity.ODEAdjointProblem(sol_init,
        DiffEqSensitivity.QuadratureAdjoint(reltol=rtol, abstol=atol),
        dg, tstops) # λ(T) = z, u₀ = z?
    λ = solve(adj_prob, sol_init.alg, reltol=rtol, abstol=atol, tstops=tstops)
    return λ
end

function condition_num(λₜ)
    K = LinearAlgebra.norm(λₜ, 1) + LinearAlgebra.norm(λₜ[1])
    return K
end

function adjoint_step_ctrl(λ₁ₜ, λ₂ₜ, E₂, Eₙ, C, gtol)
    K₁ = condition_num(λ₁ₜ)
    K₂ = condition_num(λ₂ₜ)
    K = (E₂ / Eₙ) * sqrt(K₁^2 + K₂^2)
    h = sqrt((gtol / (K * C))) # T in denominator?
    return h
end