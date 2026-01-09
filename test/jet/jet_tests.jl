using GlobalDiffEq, OrdinaryDiffEq
using JET
using Test

@testset "JET static analysis" begin
    # Test package-level static analysis
    rep = JET.report_package("GlobalDiffEq"; target_defined_modules = true)
    @test length(JET.get_reports(rep)) == 0

    # Test GlobalRichardson constructor type stability
    @test_opt GlobalRichardson(SSPRK33())

    # Test type constraint enforcement
    @test GlobalRichardson{typeof(SSPRK33())} <: GlobalDiffEq.GlobalDiffEqAlgorithm
    @test GlobalRichardson(Tsit5()) isa GlobalDiffEq.GlobalDiffEqAlgorithm
end
