
"""
    AbstractFormSpace

Supertype for all function spaces.
"""
abstract type AbstractFormSpace{n,k} end
# abstract type AbstractFormExpression{n,k} end

# function get_form_rank(::AbstractFormSpace{n,k})
#     return k
# end

# function get_spatial_dim(::AbstractFormSpace{n,k})
#     return n
# end

struct FormSpace{n, k,G, F} <: AbstractFormSpace{n,k}
    geometry::G
    fem_space::F    

    # NOTE: FunctionSpaces.AbstractFunctionSpace does not have a parameter of dimension of manifold,
    # we need to add this, but it implies several changes (I started, but postponed it!)

    # 0- and n-form constructor
    function FormSpace(form_rank::Int, geometry::G, fem_space::F) where {domain_dim, codomain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim}, F <: FunctionSpaces.AbstractFunctionSpace}
        form_components = binomial(domain_dim, form_rank)
        # The dimension match is automatically verified because of the inputs types that are enforced
        new{domain_dim, form_rank, G, F}(geometry, fem_space)
    end

    # 1-form constructor in 2D
    function FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη}) where {domain_dim, codomain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim}, F_dξ <: FunctionSpaces.AbstractFunctionSpace, F_dη <: FunctionSpaces.AbstractFunctionSpace}
        form_components = binomial(domain_dim, form_rank)
        @assert form_components == length(fem_space)
        new{domain_dim, form_rank, G, Tuple{F_dξ, F_dη}}(geometry, fem_space)
    end

    # 1- and 2-form constructor in 3D
    function FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ}) where {domain_dim, codomain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim}, F_dξ <: FunctionSpaces.AbstractFunctionSpace, F_dη <: FunctionSpaces.AbstractFunctionSpace, F_dζ <: FunctionSpaces.AbstractFunctionSpace}
        form_components = binomial(domain_dim, form_rank)
        @assert form_components == length(fem_space)
        new{domain_dim, form_rank, G, Tuple{F_dξ, F_dη, F_dζ}}(geometry, fem_space)
    end
end

# struct ExteriorDerivative{n,k,F} <: AbstractFormExpression{n,k}
#     form_space::F

#     function ExteriorDerivative(form_space::F) where {F <: AbstractFormSpace{n,k} where {n,k}}
#         n = get_form_rank(form_space)
#         k = get_spatial_dim(form_space)
#         new{n,k+1,F}(form_space)
#     end


