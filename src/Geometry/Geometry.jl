"""
    module Geometry

Contains all geometry definitions.
"""
module Geometry

import LinearAlgebra

using .. FunctionSpaces

abstract type AbstractGeometry{n} end
abstract type AbstractAnalyticalGeometry{n} <: AbstractGeometry{n} end
abstract type AbstractFEMGeometry{n} <: AbstractGeometry{n} end

get_manifold_dim(::AbstractGeometry{manifold_dim}) where {manifold_dim} = manifold_dim

@doc raw"""
    get_element_size(geometry::G, element_id::Int) where {G<:AbstractGeometry{n} where {n}}

Computes the measure of the element given by 'element_id' in 'geometry'.

# Arguments
- 'geometry::AbstractGeometry{n}': Geometry the element is a part of.
- 'element_id::Int': Index of the element being considered.

# Returns
- '::Float64': The measure of the element.
"""
function get_element_size(geometry::G, element_id::Int) where {G<:AbstractGeometry{n} where {n}}
    return _get_element_size(geometry, element_id)
end

function get_element_dimensions(geometry::G, element_id::Int) where {G<:AbstractGeometry{n} where {n}}
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
