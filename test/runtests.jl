using DrWatson
quickactivate(@__DIR__, "AIR")
using Test
using AIR

@testset "decision function" begin
    P = [0.5, 0.5, 0.5]
    D = [0, 0, 0]
    @test AIR.decide(P, D, 0.1, 0.9) == D
    @test AIR.decide(P, D, 0.6, 0.9) == ones(Int, 3)
    @test AIR.decide(P, D, 0.1, 0.4) == 2 .* ones(Int, 3)
    P = [0.05, 0.05, 0.05]
    @test AIR.decide(P, [0, 1, 2], 0.1, 0.9) == [1, 1, 2]
    P = [0.95, 0.05, 0.05]
    @test AIR.decide(P, [0, 1, 2], 0.1, 0.9) == [2, 1, 2]
end
