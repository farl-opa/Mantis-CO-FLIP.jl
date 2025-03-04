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


abstract type AbstractInputs end


include("WeakFormulations.jl")
include("GlobalAssemblers.jl")
include("AssemblersHelpers.jl")


end
