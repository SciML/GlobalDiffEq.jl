using GlobalDiffEq, OrdinaryDiffEq, LinearAlgebra, Random
using GlobalDiffEq: vec_expt_full, orth_vec
using Test

Random.seed!(123)

@testset "GlobalDiffEq Tests" begin

@testset "Vector Expectation" begin

  @testset "Vector Expectation Full" begin
    @test vec_expt_full(2) ≈ 2/π
  end;

  @testset "Random Orthogonal Vectors" begin
    function compute_vec_series(k, n)
      for _ ∈ 1:k
        z₁, z₂ = orth_vec(n)
        @test norm(z₁) ≈ 1
        @test norm(z₂) ≈ 1
        @test dot(z₁, z₂) ≈ 0 atol=1e-10
      end
    end
    k = 100 # number of random iters
    n = 10 # dimension of the vector
    compute_vec_series(k, n)
  end;
end;

@testset "Global Pendulum" begin
  l = 1.0                             # length [m]
  m = 1.0                             # mass[m]
  g = 9.81                            # gravitational acceleration [m/s²]

  function pendulum!(du,u,p,t)
      du[1] = u[2]                    # θ'(t) = ω(t)
      du[2] = -3g/(2l)*sin(u[1]) + 3/(m*l^2)*p(t) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
  end

  θ₀ = 0.01                           # initial angular deflection [rad]
  ω₀ = 0.0                            # initial angular velocity [rad/s]
  u₀ = [θ₀, ω₀]                       # initial state vector
  tspan = (0.0,10.0)                  # time interval

  M = t->0.1sin(t)                    # external torque [Nm]

  prob = ODEProblem(pendulum!,u₀,tspan,M)

  v0 = solve(prob, Tsit5(), dt=0.1, reltol=1e-12, abstol=0).(1:10)

  @testset "Global Pendulum Richardson" begin
    ve = solve(prob, GlobalRichardson(SSPRK33()), dt=0.2, reltol=1e-12, abstol=0).(1:10)
    @test norm(ve - v0) / norm(v0) < 1e-10
  end;

  @testset "Global Pendulum Adjoint" begin
    # The adjoint tests with pendulum! fail because the parameters are a function rather than a constant term
    sol_adjoint = solve(prob, GlobalAdjoint(SSPRK33()), dt=0.2, reltol=1e-4, abstol=1e-4)
    ve_adjoint = sol_adjoint.(1:10)
    @test norm(ve_adjoint - v0) / norm(v0) < 1e-4
  end;
end;

@testset "Global Lotka Volterra" begin
  function Lotka_Volterra!(du,u,p,t)
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = -p[3]*u[2] + u[1]*u[2]
  end

  u₀_LV = [1.0;1.0]
  p_LV = [1.5,1.0,3.0]
  tspan_LV = (0.0,10.0)
  prob_LV = ODEProblem(Lotka_Volterra!,u₀_LV,tspan_LV,p_LV)

  v0_LV = solve(prob_LV, Tsit5(), dt=0.1, reltol=1e-12, abstol=0).(1:10)

  @testset "Global Lotka Volterra Richardson" begin
    sol_LV_Richardson = solve(prob_LV, GlobalRichardson(SSPRK33()), dt=0.2, reltol=1e-12, abstol=0)
    ve_LV_Richardson = sol_LV_Richardson.(1:10)
    @test norm(ve_LV_Richardson - v0_LV) / norm(v0_LV) < 1e-10
  end;

  @testset "Global Lotka Volterra Adjoint" begin
    sol_LV_adjoint = solve(prob_LV, GlobalAdjoint(Tsit5()), dt=0.2, reltol=1e-10, abstol=1e-10)
    ve_LV_adjoint = sol_LV_adjoint.(1:10)
    @test norm(ve_LV_adjoint - v0_LV) / norm(v0_LV) < 1e-10
  end;
end;

end;