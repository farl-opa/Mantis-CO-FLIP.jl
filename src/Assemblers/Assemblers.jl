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
import .. Analysis

abstract type AbstractInputs end

include("WeakFormulations/WeakFormulations.jl")
include("GlobalAssemblers.jl")


end
