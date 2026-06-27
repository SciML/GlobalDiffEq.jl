using SciMLTesting, GlobalDiffEq, Test
using JET
using OrdinaryDiffEq, OrdinaryDiffEqSSPRK

run_qa(
    GlobalDiffEq;
    explicit_imports = true,
    ei_kwargs = (;
        # `SciMLBase.__solve` is SciMLBase's internal solve entry point (not part of
        # the public API); GlobalDiffEq overloads it via its owner SciMLBase.
        all_qualified_accesses_are_public = (; ignore = (:__solve,)),
    ),
    # `@reexport using DiffEqBase` deliberately reexports DiffEqBase's API, so
    # `DiffEqBase`, `ODEProblem`, and `solve` are inherently implicit. Tracked in
    # https://github.com/SciML/GlobalDiffEq.jl/issues/53
    ei_broken = (:no_implicit_imports,),
)

@testset "GlobalRichardson static analysis" begin
    @test_opt GlobalRichardson(SSPRK33())
    @test GlobalRichardson{typeof(SSPRK33())} <: GlobalDiffEq.GlobalDiffEqAlgorithm
    @test GlobalRichardson(Tsit5()) isa GlobalDiffEq.GlobalDiffEqAlgorithm
end
