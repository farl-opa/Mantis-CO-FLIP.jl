############################################################################################
#                                        Structures                                        #
############################################################################################

"""
    Plus{manifold_dim, form_rank, expression_rank, F1, F2, G} <:
    AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}

Structure representing the sum of two differential forms.

# Fields
- `form_1`: The first form.
- `form_2`: The second form.
- `label`: A string label for the sum.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0,
  with one single set of basis forms have rank 1, with two sets of basis forms have rank
  2. Higher ranks are not possible.
- `F1`: Type of the first form.
- `F2`: Type of the second form.
- `G`: Type of the underlying geometry.

# Inner Constructors
- `Plus(form_1::F1, form_2::F2)`: Creates a new `Plus` instance with the given forms.
"""
struct Plus{manifold_dim, form_rank, expression_rank, F1, F2, G} <:
       AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
    form_1::F1
    form_2::F2
    label::String

    function Plus(
        form_1::F1, form_2::F2
    ) where {
        manifold_dim,
        form_rank,
        G,
        F1 <: AbstractFormField{manifold_dim, form_rank, G},
        F2 <: AbstractFormField{manifold_dim, form_rank, G},
    }
        label = get_label(form_1) * " + " * get_label(form_2)

        return new{manifold_dim, form_rank, 0, F1, F2, G}(form_1, form_2, label)
    end

    function Base.:+(form_1::AbstractFormField, form_2::AbstractFormField)
        return Plus(form_1, form_2)
    end
end

"""
    Minus{manifold_dim, form_rank, expression_rank, F1, F2, G} <:
    AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}

Structure representing the difference of two differential forms.

# Fields
- `form_1`: The first form.
- `form_2`: The second form.
- `label`: A string label for the difference.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0,
  with one single set of basis forms have rank 1, with two sets of basis forms have rank
  2. Higher ranks are not possible.
- `F1`: Type of the first form.
- `F2`: Type of the second form.
- `G`: Type of the underlying geometry.

# Inner Constructors
- `Minus(form_1::F1, form_2::F2)`: Creates a new `Minus` instance with the given forms.
"""
struct Minus{manifold_dim, form_rank, expression_rank, F1, F2, G} <:
       AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
    form_1::F1
    form_2::F2
    label::String

    function Minus(
        form_1::F1, form_2::F2
    ) where {
        manifold_dim,
        form_rank,
        G,
        F1 <: AbstractFormField{manifold_dim, form_rank, G},
        F2 <: AbstractFormField{manifold_dim, form_rank, G},
    }
        label = get_label(form_1) * " - " * get_label(form_2)

        return new{manifold_dim, form_rank, 0, F1, F2, G}(form_1, form_2, label)
    end

    function Base.:-(form_1::AbstractFormField, form_2::AbstractFormField)
        return Minus(form_1, form_2)
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_forms(form_expression::Plus) = form_expression.form_1, form_expression.form_2
get_forms(form_expression::Minus) = form_expression.form_1, form_expression.form_2
get_geometry(form_expression::Plus) = get_geometry(get_forms(form_expression)...)
get_geometry(form_expression::Minus) = get_geometry(get_forms(form_expression)...)

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form_expression::Plus{manifold_dim},
    elem_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    forms = get_forms(form_expression)
    form_1_eval, _ = evaluate(forms[1], elem_id, xi)
    form_2_eval, _ = evaluate(forms[2], elem_id, xi)

    return form_1_eval + form_2_eval, [[1]]
end

function evaluate(
    form_expression::Minus{manifold_dim},
    elem_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    forms = get_forms(form_expression)
    form_1_eval, _ = evaluate(forms[1], elem_id, xi)
    form_2_eval, _ = evaluate(forms[2], elem_id, xi)

    return form_1_eval - form_2_eval, [[1]]
end
