"""
This (sub-)module provides a collection of scalar function spaces.

The exported names are:
"""
module FunctionSpaces

import .. Mesh
import .. Polynomial

"""
    AbstractFunctionSpace

Supertype for all scalar function spaces.
"""
abstract type AbstractFunctionSpace{n} end

include("SplineSpaces.jl")
include("ExtractionCoefficients.jl")

# Getter for the function spaces


# B-Spline getters
#=
function get_space_dim(bspline::BSplineSpace)
    return NTuple{n, Int}(get_space_dim(bspline, d) for d in 1:1:n)
end
=#

"""
    TensorProductSpace{n} <: AbstractFunctionSpace{n} 

`n`-dimensional tensor-product space.

# Fields
- `patch::Patch{n}`: Patch on which the tensor product space is defined.
- `function_spaces::NTuple{n, F} where {F <: AbstractFunctionSpace{1}}`: collection of univariate function spaces per dimension.
"""
struct TensorProductSpace{n} <: AbstractFunctionSpace{n} 
    patch::Mesh.Patch{n}
    function_spaces::NTuple{m, F} where {m, F <: AbstractFunctionSpace}
end


"""
    create_bspline_space(patch::Mesh.Patch{n}, degree::Vector{Int}, regularity::NTuple{n, Vector{Int}}) where {n}

Create a tensor product space made of only univariate b-spline spaces.

# Arguments
- `patch::Patch{n}`: Patch on which the b-spline space should be defined.
- `degree::NTuple{n, Int}`: Polynomial degree per dimension.
- `regularity::NTuple{n, Vector{Int}}`: Regularity per dimension per breakpoint.

# Returns
- `TensorProductSpace{n}`: Tensor product space of univariate b-splines.
"""
function create_bspline_space(patch::Mesh.Patch{n}, degree::Vector{Int}, regularity::NTuple{n, Vector{Int}}) where {n}
    f_spaces = NTuple{n, BSplineSpace}(BSplineSpace(Mesh.get_breakpoints(patch, i), degree[i], regularity[i]) for i in 1:1:n)
    return TensorProductSpace{n}(patch, f_spaces)
end


end