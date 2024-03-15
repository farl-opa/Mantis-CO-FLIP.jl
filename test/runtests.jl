module MantisTests

using Test

@testset "Polynomials" begin include("Polynomials/runtests.jl") end
@testset "Quadrature" begin include("Quadrature/runtests.jl") end
#@testset "ExtractionCoefficients" begin include("ExtractionCoefficients/runtests.jl") end
#@testset "Bases" begin include("Bases/runtests.jl") end
@testset "Mesh" begin include("Mesh/runtests.jl") end

end; nothing