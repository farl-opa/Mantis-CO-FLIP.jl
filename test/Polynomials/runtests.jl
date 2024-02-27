module PolynomialsTests

using Test

@testset "NodalPolynomials" begin include("NodalPolynomialsTests.jl") end

@testset "BernsteinPolynomial" begin include("BernsteinPolynomialTest.jl") end

end