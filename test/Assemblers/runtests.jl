module AssemblersTests

using Test

@testset "AssemblersTest" begin include("AssemblersTests.jl") end
@testset "PolarSplineL2ProjectionTest" begin include("L2ProjectionPolarSplines.jl") end

end