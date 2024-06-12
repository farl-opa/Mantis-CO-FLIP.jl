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
    fem_space::F

    # 0- and n-form constructor
    function FormSpace(form_rank::Int, geometry::Geometry.AbstractGeometry{domain_dim, codomain_dim}, fem_space::Tuple{F}) where {domain_dim, codomain_dim, F <: FunctionSpaces.AbstractFunctionSpace{domain_dim}}
        form_components = binomial(domain_dim, form_rank)
        @assert form_components == length(fem_space) "Dimension mismatch."
        new{domain_dim, form_rank, G, Tuple{F}}(geometry, fem_space)
    end

    # 1-form constructor in 2D
    function FormSpace(form_rank::Int, geometry::Geometry.AbstractGeometry{domain_dim, codomain_dim}, fem_space::Tuple{F_dξ, F_dη}) where {domain_dim, codomain_dim, F_dξ <: FunctionSpaces.AbstractFunctionSpace{domain_dim}, F_dη <: FunctionSpaces.AbstractFunctionSpace{domain_dim}}
        form_components = binomial(domain_dim, form_rank)
        @assert form_components == length(fem_space)
        new{domain_dim, form_rank, G, Tuple{F_dξ, F_dη}}(geometry, fem_space)
    end

    # 1- and 2-form constructor in 3D
    function FormSpace(form_rank::Int, geometry::Geometry.AbstractGeometry{domain_dim, codomain_dim}, fem_space::Tuple{F_dξ, F_dη, F_dζ}) where {domain_dim, codomain_dim, F_dξ <: FunctionSpaces.AbstractFunctionSpace{domain_dim}, F_dη <: FunctionSpaces.AbstractFunctionSpace{domain_dim}, F_dζ <: FunctionSpaces.AbstractFunctionSpace{domain_dim}}
        form_components = binomial(domain_dim, form_rank)
        @assert form_components == length(fem_space)
        new{domain_dim, form_rank, G, Tuple{F_dξ, F_dη, F_dζ}}(geometry, fem_space)
    end
end

end