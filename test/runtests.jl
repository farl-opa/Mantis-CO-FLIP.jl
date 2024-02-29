module MantisTests

using Test

@testset "Polynomials" begin include("Polynomials/runtests.jl") end
@testset "Quadrature" begin include("Quadrature/runtests.jl") end

end