"""
    module Geometry

Contains all geometry definitions.
"""
module Geometry 

using .. FunctionSpaces

abstract type AbstractGeometry{n} end
abstract type AbstractAnalGeometry{n} <: AbstractGeometry{n} end
abstract type AbstractFEMGeometry{n} <: AbstractGeometry{n} end

@doc raw"""
    get_element_measure(geometry::G, element_id::Int) where {G<:AbstractGeometry{n} where {n}}

Computes the measure of the element given by 'element_id' in 'geometry'.

# Arguments
- 'geometry::AbstractGeometry{n}': Geometry the element is a part of.
- 'element_id::Int': Index of the element being considered.

# Returns
- '::Float64': The measure of the element.
"""
function get_element_measure(geometry::G, element_id::Int) where {G<:AbstractGeometry{n} where {n}}
    return _get_element_measure(geometry, element_id)
end

include("./CartesianGeometry.jl")
include("./FEMGeometry.jl")
include("./MappedGeometry.jl")
include("./TensorProductGeometry.jl")
include("./Metric.jl")

end