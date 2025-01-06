module HierarchicalFiniteElementSpacesTests

using Test

@testset "HierarchicalBSpline" begin include("HierarchicalBSplineTests.jl") end
@testset "TensorProductHBSpline" begin include("TensorProductHBSplineTests.jl") end
@testset "TensorProductTHBSpline" begin include("TensorProductTHBSplineTests.jl") end

end
