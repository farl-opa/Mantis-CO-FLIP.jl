module AssemblersTests

using Test

@testset "AssemblersTest" begin include("AssemblersTests.jl") end
@testset "L2ProjectionConvergenceTests" begin include("L2ProjectionConvergenceTests.jl") end
@testset "PoissonZeroFormConvergenceTests" begin include("PoissonZeroFormConvergenceTests.jl") end

end