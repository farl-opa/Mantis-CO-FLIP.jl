"""
    module Assemblers

Contains all assembly-related structs and functions.
"""
module Assemblers

import SparseArrays; const spa = SparseArrays

import .. Mesh
import .. FunctionSpaces



abstract type AbstractAssemblers end
abstract type AbstractBilinearForms end


include("BilinearForms.jl")
include("GlobalAssemblers.jl")


end