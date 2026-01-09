using GlobalDiffEq, OrdinaryDiffEq, LinearAlgebra
using Test
import DiffEqBase: SciMLBase

@testset "GlobalDiffEq.jl" begin
    @testset "Basic functionality" begin
        l = 1.0                             # length [m]
        m = 1.0                             # mass[m]
        g = 9.81                            # gravitational acceleration [m/s²]

        function pendulum!(du, u, p, t)
            du[1] = u[2]                    # θ'(t) = ω(t)
            return du[2] = -3g / (2l) * sin(u[1]) + 3 / (m * l^2) * p(t) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
        end

        θ₀ = 0.01                           # initial angular deflection [rad]
        ω₀ = 0.0                            # initial angular velocity [rad/s]
        u₀ = [θ₀, ω₀]                       # initial state vector
        tspan = (0.0, 10.0)                  # time interval

        M = t -> 0.1sin(t)                    # external torque [Nm]

        prob = ODEProblem(pendulum!, u₀, tspan, M)

        v0 = solve(prob, Tsit5(), dt = 0.1, reltol = 1.0e-12, abstol = 0).(1:10)
        ve = solve(
            prob, GlobalRichardson(SSPRK33()), dt = 0.2, reltol = 1.0e-12, abstol = 0).(1:10)
        @test norm(ve - v0) / norm(v0) < 1.0e-10
    end

    @testset "Algorithm traits forwarding" begin
        alg_inner = SSPRK33()
        alg = GlobalRichardson(alg_inner)

        @test SciMLBase.allows_arbitrary_number_types(alg) ==
              SciMLBase.allows_arbitrary_number_types(alg_inner)
        @test SciMLBase.allowscomplex(alg) == SciMLBase.allowscomplex(alg_inner)
        @test SciMLBase.isautodifferentiable(alg) == SciMLBase.isautodifferentiable(alg_inner)
    end

    @testset "BigFloat support" begin
        function f_bf!(du, u, p, t)
            du[1] = -u[1]
        end

        u0_bf = BigFloat[1.0]
        tspan_bf = (BigFloat(0.0), BigFloat(1.0))
        prob_bf = ODEProblem(f_bf!, u0_bf, tspan_bf)

        sol_bf = solve(prob_bf, GlobalRichardson(SSPRK33()),
            dt = BigFloat(0.1), reltol = BigFloat(1e-3), abstol = BigFloat(1e-6))

        @test eltype(sol_bf.u[end]) == BigFloat
        # Check solution is reasonable (e^-1 ≈ 0.368)
        @test isapprox(sol_bf.u[end][1], exp(BigFloat(-1)), rtol = 1e-4)
    end
end
