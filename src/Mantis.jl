module Mantis

using InteractiveUtils # For debugging only, includes things like @code_wartype

############################################################################################
#                                         Includes                                         #
############################################################################################
include("Mesh/Mesh.jl")  # Creates Module Mesh
include("Quadrature/Quadrature.jl")  # Creates Module Quadrature
include("FunctionSpaces/FunctionSpaces.jl")  # Creates Module FunctionSpaces
include("Geometry/Geometry.jl")  # Creates Module Geometry
include("Forms/Forms.jl")  # Creates Module Forms
include("Analysis/Analysis.jl")  # Creates Module Analysis
include("Assemblers/Assemblers.jl")  # Creates Module Assemblers
include("Plot/Plot.jl")  # Creates Module Plot

############################################################################################
#                                         Exports                                          #
############################################################################################
export Mesh, Quadrature, FunctionSpaces, Geometry, Forms, Analysis, Assemblers, Plot
include("../exports/Exports.jl")

end
