"""
    module Assemblers

Contains all assembly-related structs and functions.
"""
module Assemblers

import LinearAlgebra
import SparseArrays; const spa = SparseArrays

using ..Geometry
using ..Forms
using ..Quadrature
using ..Mesh
using ..FunctionSpaces
using ..Analysis

abstract type AbstractInputs end

# include("WeakFormulations.jl")
include("temp_WeakFormInputs.jl")
include("GlobalAssemblers.jl")
include("temp_GlobalAssemblers.jl")
include("AssemblersHelpers.jl")
include("temp_WeakForms.jl")

end
