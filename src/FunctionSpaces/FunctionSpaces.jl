module FunctionSpaces

import .. Mesh

abstract type FunctionSpace{n, k} end

include("SplineSpaces.jl")

end