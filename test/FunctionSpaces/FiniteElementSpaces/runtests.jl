module FiniteElementSpacesTests

using Test

@testset "BezierExtraction" begin include("BezierExtractionTests.jl") end
@testset "KnotInsertion" begin include("KnotInsertionTests.jl") end
@testset "UnivariateSplines" begin include("UnivariateSplineTests.jl") end
@testset "TensorProduct" begin include("TensorProductTests.jl") end

@testset verbose=true "HierarchicalSpaces" begin include("Hierarchical/runtests.jl") end

# @testset verbose=true "MultivaluedSpaces" begin include("MultivaluedSpaces/runtests.jl") end

@testset verbose=true "TwoScaleRelations" begin include("TwoScaleRelations/runtests.jl") end

end
