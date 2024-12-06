module MantisTests

using Test

@testset verbose=true "Mesh" begin include("Mesh/runtests.jl") end
@testset verbose=true "CanonicalSpaces" begin include("CanonicalSpaces/runtests.jl") end
@testset verbose=true "Quadrature" begin include("Quadrature/runtests.jl") end
@testset verbose=true "FiniteElementSpaces" begin include("FiniteElementSpaces/runtests.jl") end
@testset verbose=true "Geometry" begin include("Geometry/runtests.jl") end
@testset verbose=true "Forms" begin include("Forms/runtests.jl") end
# @testset "Assembly" begin include("Assemblers/runtests.jl") end
# @testset "Plot" begin include("Plot/runtests.jl") end

end; nothing