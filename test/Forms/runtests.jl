import Mantis

module FormTests

using Test

@testset "ExteriorDerivatives" begin include("ExteriorDerivativeTests.jl") end
@testset "FormEvaluations" begin include("FormsTests.jl") end
@testset "Hodges" begin include("HodgeTests.jl") end
@testset "InnerProducts" begin include("InnerProductTests.jl") end
@testset "Wedge" begin include("WedgeTests.jl") end

end
