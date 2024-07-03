"""
    module Geometry

Contains all geometry definitions.
"""
module Geometry 

using .. FunctionSpaces

abstract type AbstractGeometry{n} end
abstract type AbstractAnalGeometry{n} <: AbstractGeometry{n} end
abstract type AbstractFEMGeometry{n} <: AbstractGeometry{n} end

include("./CartesianGeometry.jl")
include("./FEMGeometry.jl")
include("./MappedGeometry.jl")
include("./TensorProductGeometry.jl")
include("./Metric.jl")

end