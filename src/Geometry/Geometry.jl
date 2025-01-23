"""
    module Geometry

Contains all geometry structure definitions and related methods.
"""
module Geometry

import LinearAlgebra

using .. FunctionSpaces

abstract type AbstractGeometry{manifold_dim} end
abstract type AbstractAnalyticalGeometry{manifold_dim} <: AbstractGeometry{manifold_dim} end
abstract type AbstractFEMGeometry{manifold_dim} <: AbstractGeometry{manifold_dim} end

get_manifold_dim(::AbstractGeometry{manifold_dim}) where {manifold_dim} = manifold_dim

@doc raw"""
    get_element_size(geometry::AbstractGeometry, element_id::Int)

Computes the measure of the element given by 'element_id' in 'geometry'.

# Arguments
- 'geometry::AbstractGeometry': Geometry the element is a part of.
- 'element_id::Int': Index of the element being considered.

# Returns
- '::Float64': The measure of the element.
"""
function get_element_size(geometry::AbstractGeometry, element_id::Int)
    return _get_element_size(geometry, element_id)
end

function get_element_dimensions(geometry::AbstractGeometry, element_id::Int)
    return _get_element_dimensions(geometry, element_id)
end

# core functionality
include("./CartesianGeometry.jl")
include("./FEMGeometry.jl")
include("./MappedGeometry.jl")
include("./TensorProductGeometry.jl")
include("./Metric.jl")

# helper functions for convenience
include("./GeometryHelpers.jl")

end
