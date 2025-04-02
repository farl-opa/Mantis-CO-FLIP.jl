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

function get_trace_dofs(space::FunctionSpaces.AbstractFiniteElementSpace{2})
    return [i for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in space.dof_partition[1][j]]
end

function get_trace_dofs(space::FunctionSpaces.AbstractFiniteElementSpace{3})
    return [i for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] for i in space.dof_partition[1][j]]
end

end
