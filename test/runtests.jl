module MantisTests

using Test

@testset "Mesh" begin include("Mesh/runtests.jl") end
@testset "ElementyLocalBases" begin include("ElementLocalBases/runtests.jl") end
@testset "Quadrature" begin include("Quadrature/runtests.jl") end
@testset "FunctionSpaces" begin include("FunctionSpaces/runtests.jl") end

end; nothing