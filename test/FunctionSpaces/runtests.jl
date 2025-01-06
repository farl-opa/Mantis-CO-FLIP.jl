module FunctionSpacesTests

using Test

@testset "CanonicalSpaces" begin include("CanonicalSpaces/runtests.jl") end

@testset "FiniteElementSpaces" begin include("FiniteElementSpaces/runtests.jl") end

end
