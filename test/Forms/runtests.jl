import Mantis

module FormTests

using Test

@testset "ExteriorDerivatives" begin include("ExteriorDerivativeTests.jl") end
@testset "FormEvaluations" begin include("FormEvaluationTests.jl") end
@testset "Hodges" begin include("HodgeTests.jl") end
@testset "InnerProducts" begin include("InnerProductTests.jl") end
# TODO: The current way we handle operators is very type unstable so the wedge tests take a
# very long time. Leaving commented out until Forms module is reworked.
#@testset "Wedge" begin include("WedgeTests.jl") end

end
