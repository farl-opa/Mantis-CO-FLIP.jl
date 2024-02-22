module PolynomialsTests

using Test

@time @testset "NodalPolynomials" begin include("NodalPolynomialsTests.jl") end

@testset "BernsteinPolynomial" begin include("BernsteinPolynomialTest.jl") end

end