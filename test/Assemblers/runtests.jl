module AssemblersTests

using Test

@testset "AssemblersTest" begin include("AssemblersTests.jl") end
@testset "L2ProjectionConvergenceTests" begin include("L2ProjectionConvergenceTests.jl") end
@testset "PoissonZeroFormConvergenceTests" begin include("PoissonZeroFormConvergenceTests.jl") end
@testset "PolarSplineL2ProjectionTest" begin include("L2ProjectionPolarSplines.jl") end

end