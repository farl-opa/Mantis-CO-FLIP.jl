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
        F1 <: AbstractFormExpression{manifold_dim, form_rank, 0, G},
        F2 <: AbstractFormExpression{manifold_dim, form_rank, 0, G},
    }
        label = "(" * get_label(form_1) * " + " * get_label(form_2) * ")"

        return new{manifold_dim, form_rank, 0, F1, F2, G}(form_1, form_2, label)
    end

    function Base.:+(form_1::AbstractFormExpression, form_2::AbstractFormExpression)
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
        F1 <: AbstractFormExpression{manifold_dim, form_rank, 0, G},
        F2 <: AbstractFormExpression{manifold_dim, form_rank, 0, G},
    }
        label = "(" * get_label(form_1) * " - " * get_label(form_2) * ")"

        return new{manifold_dim, form_rank, 0, F1, F2, G}(form_1, form_2, label)
    end

    function Base.:-(form_1::AbstractFormExpression, form_2::AbstractFormExpression)
        return Minus(form_1, form_2)
    end
end

"""
    AdditiveInverse{manifold_dim, R} <: AbstractRealValuedOperator{manifold_dim}

Structure representing the additive inverse of a given real-valued operator.

# Fields
- `operator`: The operator to be inverted.

# Type parameters
- `manifold_dim`: Dimension of the manifold on which the operator acts.
- `R`: Type of the operator. Must be a subtype of `AbstractRealValuedOperator`.

# Inner Constructors
- `AdditiveInverse(operator::O)`: Creates a new `AdditiveInverse` instance with the given operator.
- `Base.:-(operator::AbstractRealValuedOperator)`: Symbolic wrapper for the inverse
    operator.
"""
struct AdditiveInverse{manifold_dim, O} <: AbstractRealValuedOperator{manifold_dim}
    operator::O

    function AdditiveInverse(
        operator::O
    ) where {manifold_dim, O <: AbstractRealValuedOperator{manifold_dim}}
        return new{manifold_dim, O}(operator)
    end

    function Base.:-(operator::AbstractRealValuedOperator)
        return AdditiveInverse(operator)
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_forms(form_expression::Plus) = form_expression.form_1, form_expression.form_2
get_forms(form_expression::Minus) = form_expression.form_1, form_expression.form_2
get_geometry(form_expression::Plus) = get_geometry(get_forms(form_expression)...)
get_geometry(form_expression::Minus) = get_geometry(get_forms(form_expression)...)
get_operator(add_inverse::AdditiveInverse) = add_inverse.operator

function get_estimated_nnz_per_elem(add_inverse::AdditiveInverse)
    return get_estimated_nnz_per_elem(get_operator(add_inverse))
end

function get_num_evaluation_elements(add_inverse::AdditiveInverse)
    return get_num_evaluation_elements(get_operator(add_inverse))
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

"""
    evaluate(
        form_expression::Plus{manifold_dim},
        element_id::Int,
        xi::NTuple{manifold_dim, Vector{Float64}},
    ) where {manifold_dim}

Evaluates the sum of two differential forms at a given element and canonical points `xi`.

# Arguments
- `form_expression::Plus{manifold_dim}`: The form expression resulting from a sum of form
    expressions to be evaluated.
- `element_id::Int`: The identifier of the element where the evaluation takes place.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: The canonical points where the evaluation is
    performed.

# Returns
- `::Vector{Matrix{Float64}}`: The result of the evaluation of the sum of
    the two forms.
- `[[1]]::Vector{Vector{Int}}`: The indices of the evaluated forms.
"""
function evaluate(
    form_expression::Plus{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    forms = get_forms(form_expression)
    form_1_eval, _ = evaluate(forms[1], element_id, xi)
    form_2_eval, _ = evaluate(forms[2], element_id, xi)

    return form_1_eval + form_2_eval, [[1]]
end

"""
    evaluate(
        form_expression::Minus{manifold_dim},
        element_id::Int,
        xi::NTuple{manifold_dim, Vector{Float64}},
    ) where {manifold_dim}

Evaluates the difference of two differential forms at a given element and canonical points
`xi`.

# Arguments
- `form_expression::Minus{manifold_dim}`: The form expression resulting from a difference
    of form expressions to be evaluated.
- `element_id::Int`: The identifier of the element where the evaluation takes place.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: The canonical points where the evaluation
    is performed.

# Returns
- `::Vector{Matrix{Float64}}`: The result of the evaluation of the difference of
    the two forms.
- `[[1]]::Vector{Vector{Int}}`: The indices of the evaluated forms.
"""
function evaluate(
    form_expression::Minus{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    forms = get_forms(form_expression)
    form_1_eval, _ = evaluate(forms[1], element_id, xi)
    form_2_eval, _ = evaluate(forms[2], element_id, xi)

    return form_1_eval - form_2_eval, [[1]]
end

"""
    evaluate(
        add_inverse::AdditiveInverse{manifold_dim},
        element_id::Int,
    ) where {manifold_dim}

Evaluates the additive inverse of a given operator at a specified element using a quadrature
rule.

# Arguments
- `add_inverse::AdditiveInverse{manifold_dim}`: The inverse operator to evaluate.
- `element_id::Int`: The element over which to evaluate the inverse operator.

# Returns
- `eval::Vector{Float64}`: The additive inverse of the evaluated operator.
- `indices::Vector{Vector{Int}}`: The indices of the evaluated operator. The length of the
    outer vector depends on the `expression_rank` of the form expression.
"""
function evaluate(
    add_inverse::AdditiveInverse{manifold_dim}, element_id::Int
) where {manifold_dim}
    operator = get_operator(add_inverse)
    eval, indices = evaluate(operator, element_id)
    eval .*= -1.0

    return eval, indices
end
