"""
    module Forms

Contains all definitions of forms, including form fields, form spaces, and form expressions.
"""
module Forms

import ..FunctionSpaces
import ..Geometry
import ..Quadrature

abstract type AbstractFormExpression{manifold_dim, form_rank, expression_rank, G} end
abstract type AbstractFormField{manifold_dim, form_rank, expression_rank, G} <:
              AbstractFormExpression{manifold_dim, form_rank, expression_rank, G} end

@doc raw"""
    AbstractFormSpace{manifold_dim, form_rank, G} <: AbstractFormField{manifold_dim, form_rank, expression_rank, G}

Supertype for all function spaces representing differential forms.

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0, with one single set of basis forms have rank 1, with two sets of basis forms have rank 2. Higher ranks are not possible.
- `G`: Type of the underlying geometry

# Inner Constructors
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F})`: Constructor for 0-forms and n-forms
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη})`: Constructor for 1-forms in 2D
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ})`: Constructor for 1-forms and 2-forms in 3D
"""
abstract type AbstractFormSpace{manifold_dim, form_rank, G} <:
              AbstractFormField{manifold_dim, form_rank, 1, G} end

# General getters involving forms. These are general enough to work on all subtypes.
@doc raw"""
    get_form_rank(form::FE) where {FE <: AbstractFormExpression{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Returns the form rank for the given `form` (expression).

# Arguments
- `form::AbstractFormExpression`: The form of which to get the rank.
"""
function get_form_rank(::FE) where {
    manifold_dim,
    form_rank,
    expression_rank,
    G <: Geometry.AbstractGeometry,
    FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
}
    return form_rank
end

@doc raw"""
    get_expression_rank(form::FE) where {FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Returns the expression rank for the given `form` (expression).

# Arguments
- `form::AbstractFormExpression`: The form of which to get the expression rank. Expressions without basis forms have rank 0, 
      with one single set of basis forms have rank 1, with two sets of basis forms have rank 2. Higher ranks are not possible.
"""
function get_expression_rank(::FE) where {
    manifold_dim,
    form_rank,
    expression_rank,
    G <: Geometry.AbstractGeometry,
    FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
}
    return expression_rank
end

@doc raw"""
    get_geometry(form::FS) where {FS <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}} where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Returns the geometry associated with the `form_space`.

# Arguments
- `form::AbstractFormExpression`: The Form to get the geometry of.
"""
function get_geometry(form::FE) where {
    manifold_dim,
    form_rank,
    expression_rank,
    G <: Geometry.AbstractGeometry,
    FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
}
    return form.geometry
end

# TODO get_geometry should return the geometry of a form expression, not compare.
#      I agree we must enforce this, but then we should add the enforcement explicitly
#      when needed, not in this function.
@doc raw"""
    get_geometry(form_1::FS1, form_2::FS2) where {FS1 <: AbstractFormExpression{manifold_dim1, form_rank1, expression_rank_1, G1}, FS2 <: AbstractFormExpression{manifold_dim2, form_rank2, expression_rank_2, G2}} where {manifold_dim1, form_rank1, expression_rank_1, G1 <: Geometry.AbstractGeometry{manifold_dim1}, manifold_dim2, form_rank2, , expression_rank_2, G2 <: Geometry.AbstractGeometry{manifold_dim2}}

Returns the geometry associated with `form_1` if it is the same as `form_2`.

# Arguments
- `form_1::AbstractFormExpression`: The frist Form to get the geometry of, if the same as `form_2`.
- `form_2::AbstractFormExpression`: The second Form to get the geometry of, if the same as `form_1`.
"""
function get_geometry(form_1::FS1, form_2::FS2) where {
    FS1 <: AbstractFormExpression{manifold_dim1, form_rank1, expression_rank_1, G1},
    FS2 <: AbstractFormExpression{manifold_dim2, form_rank2, expression_rank_2, G2},
} where {
    manifold_dim1,
    form_rank1,
    expression_rank_1,
    G1 <: Geometry.AbstractGeometry{manifold_dim1},
    manifold_dim2,
    form_rank2,
    expression_rank_2,
    G2 <: Geometry.AbstractGeometry{manifold_dim2},
}
    if form_1.geometry === form_2.geometry
        return form_1.geometry
    else
        geom1 = get_geometry(form_1)
        tpg1 = typeof(geom1)
        geom2 = get_geometry(form_2)
        tpg2 = typeof(geom2)
        msg1 = "The given forms do not have the same geometry. "
        msg2 = "The first has geometry: $geom1 of type $tpg1. "
        msg3 = "The second has geometry: $geom2 of type $tpg2."
        throw(ArgumentError(msg1 * msg2 * msg3))
    end
end

# core functionality
include("./FormSpaces.jl")
include("./FormExpressions.jl") # Requires FormSpaces
include("./FormOperators.jl") # Mind the order, FormOperators requires FormSpaces and FormExpressions
include("./MixedFormSpace.jl")

# helper functions for convenience
include("./FormHelpers.jl")

# 0-form: no basis
# 1-forms: (dx1, dx2, dx3)
# 2-forms: (dx2dx3, dx3dx1, dx1dx2)
# 3-forms: dx1dx2d3

end
