module MimeticFEMTests

using Test

@time @testset "Polynomials" begin include("Polynomials/runtests.jl") end

end