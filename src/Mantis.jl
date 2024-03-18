module Mantis

# Remember that these include statements have to be in order, so that 
# the include that are listed later can use code from the previous ones, 
# but not the other way around.
include("Mesh/mesh.jl")
include("Polynomials/Polynomials.jl")
include("Quadrature/Quadrature.jl")
include("FunctionSpaces/FunctionSpaces.jl")
include("ExtractionCoefficients/ExtractionCoefficients.jl")

end
