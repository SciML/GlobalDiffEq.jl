function adjoint_solve(prob, sol_init, u₀, tstops, p, rtol, atol)
    de = ModelingToolkit.modelingtoolkitize(prob)
    sym_jac = eval(ModelingToolkit.generate_jacobian(de)[1])
    function adjoint!(du, u, p, t)
        du = -Base.invokelatest(sym_jac, sol_init(t), p, t)' * u
    end
    rev_tstops = reverse(tstops) # reverse tstops for λ(T) = u₀?
    rev_tspan = (rev_tstops[1], rev_tstops[end])
    adj_prob = ODEProblem(adjoint!, u₀, rev_tspan, p)
    λ = solve(adj_prob, sol_init.alg, reltol=rtol, abstol=atol, tstops=rev_tstops)
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