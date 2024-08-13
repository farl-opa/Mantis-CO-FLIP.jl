"""
    module Forms

Contains all definitions of forms, including form fields, form spaces, and form expressions.
"""
module Forms

import .. FunctionSpaces
import .. Geometry
import .. Quadrature

abstract type AbstractFormExpression{manifold_dim, form_rank, G} end
abstract type AbstractFormField{manifold_dim, form_rank, G} <: AbstractFormExpression{manifold_dim, form_rank, G} end
@doc raw"""
    AbstractFormSpace{manifold_dim, form_rank, G} <: AbstractFormField{manifold_dim, form_rank, G}

Supertype for all function spaces representing differential forms.

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the underlying geometry

# Inner Constructors
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F})`: Constructor for 0-forms and n-forms
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη})`: Constructor for 1-forms in 2D
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ})`: Constructor for 1-forms and 2-forms in 3D
"""
abstract type AbstractFormSpace{manifold_dim, form_rank, G} <: AbstractFormField{manifold_dim, form_rank, G} end




# General getters involving forms. These are general enough to work on all subtypes.
@doc raw"""
    get_geometry(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Returns the geometry associated with the `form_space`.

# Arguments
- `form_space::AbstractFormSpace`: The FormSpace to get the geometry of.
"""
function get_geometry(form_space::FS) where {FS <: AbstractFormExpression{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    return form_space.geometry
end

@doc raw"""
    get_geometry(form_space1::FS1, form_space2::FS2) where {FS1 <: AbstractFormSpace{manifold_dim1, form_rank1, G1}, FS2 <: AbstractFormSpace{manifold_dim2, form_rank2, G2}} where {manifold_dim1, form_rank1, G1 <: Geometry.AbstractGeometry{manifold_dim1}, manifold_dim2, form_rank2, G2 <: Geometry.AbstractGeometry{manifold_dim2}}

Returns the geometry associated with `form_space1` if it is the same as `form_space2`.

# Arguments
- `form_space1::AbstractFormSpace`: The frist FormSpace to get the geometry of.
- `form_space2::AbstractFormSpace`: The second FormSpace to get the geometry of.
"""
function get_geometry(form_space1::FS1, form_space2::FS2) where {FS1 <: AbstractFormExpression{manifold_dim1, form_rank1, G1}, FS2 <: AbstractFormExpression{manifold_dim2, form_rank2, G2}} where {manifold_dim1, form_rank1, G1 <: Geometry.AbstractGeometry{manifold_dim1}, manifold_dim2, form_rank2, G2 <: Geometry.AbstractGeometry{manifold_dim2}}
    if form_space1.geometry === form_space2.geometry
        return form_space1.geometry
    else
        geom1 = get_geometry(form_space1)
        tpg1 = typeof(geom1)
        geom2 = get_geometry(form_space2)
        tpg2 = typeof(geom2)
        msg1 = "The given forms do not have the same geometry. "
        msg2 = "The first has geometry: $geom1 of type $tpg1. "
        msg3 = "The second has geometry: $geom2 of type $tpg2."
        throw(ArgumentError(msg1*msg2*msg3))
    end
end

include("./FormSpaces.jl")
include("./FormExpressions.jl") # Requires FormSpaces
include("./FormOperators.jl") # Mind the order, FormOperators requires FormSpaces and FormExpressions

# 0-form: no basis
# 1-forms: (dx1, dx2, dx3)
# 2-forms: (dx2dx3, dx3dx1, dx1dx2)
# 3-forms: dx1dx2d3

end