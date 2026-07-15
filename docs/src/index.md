# GlobalDiffEq.jl

GlobalDiffEq provides ODE solver wrappers that estimate or control error over
the full integration interval.

## Algorithms

```@autodocs
Modules = [GlobalDiffEq]
```

`GlobalAdjoint` implements adjoint-based a posteriori error estimation. It uses
SciMLSensitivity's public adjoint problem and vector-Jacobian-product machinery,
then integrates the numerical solution's defect against randomized terminal
directions. For example:

```julia
alg = GlobalAdjoint(Tsit5(); gtol = 1.0e-6)
sol = solve(prob, alg)
```

The method follows Yang Cao and Linda Petzold, [*A Posteriori Error Estimation
and Global Error Control for Ordinary Differential Equations by the Adjoint
Method*](https://doi.org/10.1137/S1064827503420969), SIAM Journal on Scientific
Computing 26(2), 2004.
