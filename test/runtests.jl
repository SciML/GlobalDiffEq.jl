using SafeTestsets
using Test
using SciMLTesting

run_tests(;
    core = function ()
        @safetestset "Basic functionality" begin
            include("basic_functionality_tests.jl")
        end
        @safetestset "Algorithm traits forwarding" begin
            include("algorithm_traits_tests.jl")
        end
        return @safetestset "BigFloat support" begin
            include("bigfloat_tests.jl")
        end
    end,
    groups = Dict(
        # JET runs the static-analysis tests in its own environment. The original
        # dispatcher ran these only for GROUP=="JET" (never as part of "All"), so JET
        # is an env-bearing group kept out of the curated `all`.
        "JET" => (;
            env = joinpath(@__DIR__, "JET"),
            body = function ()
                return @safetestset "JET static analysis" begin
                    include(joinpath(@__DIR__, "JET", "jet_tests.jl"))
                end
            end,
        ),
    ),
    # The original runtests.jl ran the Core body for GROUP=All and GROUP=Core, and
    # ran JET only for GROUP=JET (never under "All"). Curate "All" to Core only.
    all = ["Core"],
)
