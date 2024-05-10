module HierarchicalFiniteElementSpacesTests

using Test

@testset "HierarchicalBSplineTests" begin include("HierarchicalBSplineTests.jl") end
@testset "TensorProductHBSplineTests" begin include("TensorProductHBSplineTests.jl") end

end