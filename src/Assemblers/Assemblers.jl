"""
    module Assemblers

Contains all assembly-related structs and functions.
"""
module Assemblers

import LinearAlgebra
import SparseArrays; const spa = SparseArrays

import .. Geometry
import .. Forms
import .. Quadrature
import .. Mesh
import .. FunctionSpaces


abstract type AbstractAssemblers end
abstract type AbstractInputs end
abstract type AbstractBoundaryForms end


include("BoundaryForms.jl")
include("WeakFormulations.jl")
include("GlobalAssemblers.jl")
include("ErrorComputations.jl")

include("AssemblerHelpers.jl")


end