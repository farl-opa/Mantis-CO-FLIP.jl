module Mantis

using InteractiveUtils # For debugging only, includes things like @code_wartype

# Remember that these include statements have to be in order, so that 
# the include that are listed later can use code from the previous ones, 
# but not the other way around.
include("Mesh/Mesh.jl")  # Creates Module Mesh
include("Quadrature/Quadrature.jl")  # Creates Module Quadrature
include("FunctionSpaces/FunctionSpaces.jl")  # Creates Module FunctionSpaces
include("Fields/Fields.jl") 
include("Geometry/Geometry.jl")  # Creates Module Geometry
include("Assemblers/Assemblers.jl")  # Creates Module Assemblers
include("Analysis/Analysis.jl")  # Creates Module Analysis
include("Plot/Plot.jl")  # Creates Module Plot
# include("HierarchicalFiniteElementSpaces/HierarchicalFiniteElementSpaces.jl")  # Creates Module HierarchicalFiniteElementSpaces

end
