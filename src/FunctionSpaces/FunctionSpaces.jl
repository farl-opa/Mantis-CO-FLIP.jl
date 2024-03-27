"""
This (sub-)module provides a collection of scalar function spaces.

The exported names are:
"""
module FunctionSpaces

import .. Mesh
import .. Polynomials

"""
    AbstractFunctionSpace

Supertype for all scalar function spaces.
"""
abstract type AbstractFunctionSpace{n} end

include("SplineSpaces.jl")
include("ExtractionCoefficients.jl")

# Getters for the function spaces
get_n(f::AbstractFunctionSpace{n}) where {n} = n

# B-Spline getters

"""
    get_space_dim(bspline::BSplineSpace)

Returns the dimension of the univarite function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
# Returns
- `::Int`: The dimension of the B-Spline space.
"""
function get_space_dim(bspline::BSplineSpace)
    return length(bspline.knot_vector.breakpoints-1)*bspline.knot_vector.polynomial_degree - sum(bspline.knot_vector.multiplicity) + 1
end

# TensorProductSpace constructors

"""
    TensorProductSpace{n} <: AbstractFunctionSpace{n} 

`n`-variate tensor-product space.

# Fields
- `patch::Patch{n}`: Patch on which the tensor product space is defined.
- `function_spaces::NTuple{m, F} where {m, F <: AbstractFunctionSpace}`: collection of uni or multivariate function spaces.
"""
struct TensorProductSpace{n} <: AbstractFunctionSpace{n} 
    patch::Mesh.Patch{n}
    function_spaces::NTuple{m, AbstractFunctionSpace} where {m}
    function TensorProductSpace(patch::Mesh.Patch{n}, function_spaces::NTuple{m, AbstractFunctionSpace}) where {n,m}
        if sum([get_n(function_spaces[i]) for i in 1:1:m]) != n
            throw(ArgumentError("The sum of the dimensions of the input spaces does not match the dimension of the patch!"))
        end
        new{n}(patch, function_spaces)
    end
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