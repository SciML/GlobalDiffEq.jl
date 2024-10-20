function vec_expt_est(n)
    # Estimate method. Not used, but could substitute for vec_expt_full.
    Eₙ = sqrt(2 / (π*(n - 1/2)))
    return Eₙ
end

function vec_expt_full(n)
    if isodd(n)
        Eₙ = prod(range(1, n-2, step=2)) / prod(range(2, n-1, step=2))
    else
        Eₙ = (2/π) * prod(range(2, n-2, step=2)) / prod(range(1, n-1, step=2))
    end
    return Eₙ
end

function orth_vec(n)
    mvn_1 = Distributions.MvNormal(n, 1)
    z₁ = Distributions.rand(mvn_1)
    z₁ = z₁ / LinearAlgebra.norm(z₁)
    z₁_null = LinearAlgebra.nullspace(Matrix(z₁'))
    mvn_2 = Distributions.MvNormal(n-1, 1)
    v₂ = Distributions.rand(mvn_2)
    v₂ = v₂ / LinearAlgebra.norm(v₂)
    z₂ = vec(sum(v₂' .* z₁_null, dims=2))
    return z₁, z₂
end