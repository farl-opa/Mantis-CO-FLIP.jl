"""
    module Forms

Contains all definitions of forms, including form fields, form spaces, and form expressions.
"""
module Forms

using ..FunctionSpaces
using ..Geometry
using ..Quadrature

import LinearAlgebra
import Subscripts

############################################################################################
#                                         Exports                                          #
############################################################################################
include("FormsExports.jl")

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
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the underlying geometry.
"""
abstract type AbstractFormExpression{manifold_dim, form_rank, expression_rank, G} end

"""
    AbstractFormField{manifold_dim, form_rank, G} <:
    AbstractFormExpression{manifold_dim, form_rank, 0, G}

Supertype for all form fields. 

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the underlying geometry
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
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the underlying geometry.
"""
abstract type AbstractFormSpace{manifold_dim, form_rank, G} <:
              AbstractFormExpression{manifold_dim, form_rank, 1, G} end

"""
    AbstractRealValuedOperator{manifold_dim}

Supertype for all real-valued operators defined over a manifold.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
"""
abstract type AbstractRealValuedOperator{manifold_dim} end

############################################################################################
#                                     Abstract Methods                                     #
############################################################################################

"""
    get_manifold_dim(::AbstractFormExpression{manifold_dim}) where {manifold_dim}

Returns the manifold dimension of the given form expression.

# Arguments
- `::AbstractFormExpression{manifold_dim}`: The form expression.

# Returns
- `manifold_dim::Int`: The manifold dimension.
"""
function get_manifold_dim(::AbstractFormExpression{manifold_dim}) where {manifold_dim}
    return manifold_dim
end

"""
    get_geometry(form_expression::AbstractFormExpression)

Returns the geometry of the given form expression.

# Arguments
- `form_expression::AbstractFormExpression`: The form expression.

# Returns
- `<:Geometry.AbstractGeometry`: The geometry of the form expression.
"""
get_geometry(form_expression::AbstractFormExpression) = form_expression.geometry

"""
    get_geometry(
        single_form::AbstractFormExpression, additional_forms::AbstractFormExpression...
    )

If, a single form is given, returns the geometry of that form. If additional forms are
given, checks if all the geometries of the different forms refer to the same object in
memory, and then returns it.

# Arguments
- `single_form::AbstractFormExpression`: The first form.
- `additional_forms::AbstractFormExpression...`: Arbitrary number of additional forms.

# Returns
- `<:Geometry.AbstractGeometry`: The geometry of the given form(s).
"""
function get_geometry(
    single_form::AbstractFormExpression, additional_forms::AbstractFormExpression...
)
    all_forms = tuple(single_form, additional_forms...)

    for i in 1:(length(all_forms) - 1)
        if !(get_geometry(all_forms[i]) == get_geometry(all_forms[i+1]))
            msg1 = "Not all forms share a common geometry. "
            msg2 = "The geometries of form number(i) and form number(i+1) differ."
            throw(ArgumentError(msg1 * msg2))
        end
    end

    return get_geometry(single_form)
end


"""
    get_num_elements(form::AbstractFormExpression)

Returns the number of elements in the geometry of the given form expression.

# Arguments
- `form::AbstractFormExpression`: The form expression.

# Returns
- `Int`: The number of elements in the geometry of the form expression.
"""
function get_num_elements(form::AbstractFormExpression)
    return Geometry.get_num_elements(get_geometry(form))
end

"""
    get_form_rank(::FE) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    }

Returns the form rank of the given form expression.

# Arguments
- `::FE`: The form expression.

# Returns
- `::Int`: The form rank of the form.
"""
function get_form_rank(::FE) where {
    manifold_dim,
    form_rank,
    expression_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
}
    return form_rank
end

"""
    get_expression_rank(::FE) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    }

Returns the `expression_rank`

# Arguments
- `::FE`: The form expression.

# Returns
- `::Int`: The expression rank of the form.
"""
function get_expression_rank(::FE) where {
    manifold_dim,
    form_rank,
    expression_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    FE <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
}
    return expression_rank
end

"""
    get_label(form::AbstractFormExpression)

Returns the label of the form expression.

# Arguments
- `form::AbstractFormExpression`: The form expression.

# Returns
- `String`: The label of the form expression.
"""
get_label(form::AbstractFormExpression) = form.label

############################################################################################
#                                         Includes                                         #
############################################################################################

include("./FormExpressions/FormExpressions.jl")
include("./FormOperators/FormOperators.jl")
include("./FormsHelpers.jl")

end
