module AssemblersTests

using Test

@testset "AssemblersTest" begin include("AssemblersTests.jl") end
@testset "L2ProjectionConvergenceTests" begin include("L2ProjectionConvergenceTests.jl") end
@testset "PoissonConvergenceTests" begin include("PoissonConvergenceTests.jl") end

end