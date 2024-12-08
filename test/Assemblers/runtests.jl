module AssemblersTests

using Test

# @testset "AssemblersTest" begin include("AssemblersTests.jl") end
@testset "k-form-TensorProduct-L2ProjectionTests" begin include("k-form-TensorProduct-L2ProjectionTests.jl") end
@testset "0-form-TensorProduct-H1ProjectionTests" begin include("0-form-TensorProduct-H1ProjectionTests.jl") end
# @testset "PolarSplineL2ProjectionTest" begin include("L2ProjectionPolarSplines.jl") end

end