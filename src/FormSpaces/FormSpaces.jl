"""
This (sub-)module provides a collection of form spaces.
The exported names are:
"""
module FormSpaces

import .. FunctionSpaces

"""
    AbstractFormSpace

Supertype for all function spaces.
"""
abstract type AbstractFormSpace end

struct FormSpace{n,k,G,F} <: AbstractFormSpace
    geometry::G
    fem_space::Tuple{m,F}

    # 0- and n-form constructor
    function FormSpace(k::Int, geometry::Geometry.AbstractGeometry{n,r}, fem_space::Tuple{F}) where {n,r,F <: FunctionSpaces.AbstractFunctionSpace{n}}
        m = binomial(n,k)
        @assert m == length(fem_space) "Dimension mismatch."
        new{n,k,G,Tuple{F}}(geometry, fem_space)
    end

    # 1-form constructor in 2D
    function FormSpace(k::Int, geometry::Geometry.AbstractGeometry{n,r}, fem_space::Tuple{F1, F2}) where {n,r, F1 <: FunctionSpaces.AbstractFunctionSpace{n}, F2 <: FunctionSpaces.AbstractFunctionSpace{n}}
        m = binomial(n,k)
        @assert m == length(fem_space)
        new{n,k,G,Tuple{F1, F2}}(geometry, fem_space)
    end

    # 1- and 2-form constructor in 3D
    function FormSpace(k::Int, geometry::Geometry.AbstractGeometry{n,r}, fem_space::Tuple{F1, F2, F3}) where {n,r, F1 <: FunctionSpaces.AbstractFunctionSpace{n}, F2 <: FunctionSpaces.AbstractFunctionSpace{n}, F3 <: FunctionSpaces.AbstractFunctionSpace{n}}
        m = binomial(n,k)
        @assert m == length(fem_space)
        new{n,k,G,Tuple{F1, F2, F3}}(geometry, fem_space)
    end
end

end