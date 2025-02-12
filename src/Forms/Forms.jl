"""
    module Forms

Contains all definitions of forms, including form fields, form spaces, and form expressions.
"""
module Forms

import ..FunctionSpaces
import ..Geometry
import ..Quadrature

############################################################################################
#                                      Abstract Types                                      #
############################################################################################
"""
    AbstractFormExpression{manifold_dim, form_rank, G}

Supertype for all form expressions representing differential forms.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0,
    with one single set of basis forms have rank 1, with two sets of basis forms have rank
    2. Higher ranks are not possible.
- `G`: Type of the underlying geometry.
"""
abstract type AbstractFormExpression{manifold_dim, form_rank, expression_rank, G} end

"""
    AbstractFormField{manifold_dim, form_rank, G} <:
    AbstractFormExpression{manifold_dim, form_rank, 0, G}

Supertype for all form fields. 

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the underlying geometry
"""
abstract type AbstractFormField{manifold_dim, form_rank, G} <:
              AbstractFormExpression{manifold_dim, form_rank, 0, G} end

"""
    AbstractFormSpace{manifold_dim, form_rank, G} <:
    AbstractFormExpression{manifold_dim, form_rank, 1, G}

Supertype for all form spaces. These all have expression rank 1, as they have a single set
of basis functions.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `G`: Type of the underlying geometry.
"""
abstract type AbstractFormSpace{manifold_dim, form_rank, G} <:
              AbstractFormExpression{manifold_dim, form_rank, 1, G} end

############################################################################################
#                                     Abstract Methods                                     #
############################################################################################
"""
    get_form_rank(::FE) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry,
        FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    }

Returns the form rank of the given form expression.

# Arguments
- `::FE`: The form expression.

# Returns
- `::Int`: The form rank of the form.

# Notes
This method is used as a fallback if there isn't a more specific method to be used. The
latter should only be implemented explicitly if necessary.
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

"""
    get_expression_rank(::FE) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry,
        FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    }

Returns the `expression_rank`

# Arguments
- `::FE`: The form expression.

# Returns
- `::Int`: The expression rank of the form.

# Notes
This method is used as a fallback if there isn't a more specific method to be used. The
latter should only be implemented explicitly if necessary.
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

"""
    get_geometry(form::FE) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry,
        FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    }

Returns the geometry where the given `form` is defined.

# Arguments
- `form::FE`: The form expression.

# Returns
- `G`: The geometry type.

# Notes
This method is used as a fallback if there isn't a more specific method to be used. The
latter should only be implemented explicitly if necessary.
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
"""
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

############################################################################################
#                                         Includes                                         #
############################################################################################
include("./FormSpaces.jl")
include("./FormExpressions.jl")
include("./FormOperators.jl") 
include("./MixedFormSpace.jl")
include("./FormHelpers.jl")

end
