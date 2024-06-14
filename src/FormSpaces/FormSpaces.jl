"""
This (sub-)module provides a collection of form spaces.
The exported names are:
"""
module FormSpaces

import .. FunctionSpaces
import .. Geometry

"""
    AbstractFormSpace

Supertype for all function spaces.
"""
abstract type AbstractFormSpace{n,k} end

function get_form_rank(::AbstractFormSpace{n,k})
    return k
end

function get_spatial_dim(::AbstractFormSpace{n,k})
    return n
end

struct FormSpace{n,k,G,F} <: AbstractFormSpace{n,k}
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

struct ExteriorDerivative{n,k,F} <: AbstractFormSpace{n,k}
    form_space::F

    function ExteriorDerivative(form_space::F) where {F <: AbstractFormSpace{n,k} where {n,k}}
        n = get_domain_dim(form_space)
        k = get_form_rank(form_space)
        new{n,k+1,F}(form_space)
    end

end

function evaluate(ext_der::ExteriorDerivative{n,0,F},element_id::Int,quad_rule::Quadrature.QuadratureRule{n}) where {n,F}
    # Compute derivatives of bases
    basis_eval, basis_inds = FunctionSpaces.evaluate(ext_der.form_space.fem_space[1], element_id, quad_rule.xi, 1)
    # construct and return the exterior derivative of the 0-form...
end
