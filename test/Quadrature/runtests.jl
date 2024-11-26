module QuadratureTests

using Test

@testset "ClenshawCurtisQuadrature" begin include("ClenshawCurtisQuadratureTests.jl") end
@testset "GaussQuadrature" begin include("GaussQuadratureTests.jl") end
@testset "NewtonCotesQuadrature" begin include("NewtonCotesQuadratureTests.jl") end
@testset "TensorProductQuadrature" begin include("TensorProductQuadratureTests.jl") end

end