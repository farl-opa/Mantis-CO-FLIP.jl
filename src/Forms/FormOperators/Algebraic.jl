############################################################################################
#                                        Structures                                        #
############################################################################################
"""
  UnitaryOperatorTransformation{manifold_dim, O, T} <: AbstractRealValuedOperator{manifold_dim}

Structure holding the necessary information to evaluate a unitary, algebraic transformation
of a `AbstractRealValuedOperator`.

# Fields
- `operator::O`: The operator to which the transformation is applied.
- `transformation::T`: The transformation to apply to the operator.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the operator is defined.
- `O <: AbstractRealValuedOperator{manifold_dim}`: Type of the original real-valued
  operator.
- `T <: Function`: Function defining the algebraic transformation.

# Inner constructors
- `UnitaryOperatorTransformation(operator::O, transformation::T)`: General constructor.
- `Base.:*(factor::Number, operator::AbstractRealValuedOperator)`: Alias for the
  multiplication of a real-valued operator with a constant factor.
- `Base.:-(operator::AbstractRealValuedOperator)`: Alias for the additive inverse of a
  real-valued operator.
"""
struct UnitaryOperatorTransformation{manifold_dim, O, T} <:
       AbstractRealValuedOperator{manifold_dim}
    operator::O
    transformation::T

    function UnitaryOperatorTransformation(
        operator::O, transformation::T
    ) where {manifold_dim, O <: AbstractRealValuedOperator{manifold_dim}, T <: Function}
        return new{manifold_dim, O, T}(operator, transformation)
    end

    function Base.:*(factor::Number, operator::AbstractRealValuedOperator)
        return UnitaryOperatorTransformation(operator, x -> factor .* x)
    end

    Base.:-(operator::AbstractRealValuedOperator) = -1.0 * operator
end

"""
  BinaryOperatorTransformation{manifold_dim, O1, O2, T} <:
  AbstractRealValuedOperator{manifold_dim}

Structure holding the necessary information to evaluate a binary, algebraic transformation
acting on two real-valued operators.

# Fields
- `operator_1::O1`: The first real-valued operator.
- `operator_2::O2`: The second real-valued operator.
- `transformation::T`: The transformation to apply to the operators.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the operators are defined.
- `O1 <: AbstractRealValuedOperator{manifold_dim}`: The type of the first operator.
- `O2 <: AbstractRealValuedOperator{manifold_dim}`: The type of the second operator.
- `T <: Function`: The type of the algebraic transformation.

# Inner constructors
- `BinaryOperatorTransformation(operator_1::O1, operator_2::O2, transformation::T )`:
  General constructor.
- `Base.:+(operator_1::O1, operator_1::O2)`: Alias for the sum of two operators.
- `Base.:-(operator_1::O1, operator_2::O2)`: Alias for the difference of two operators.
"""
struct BinaryOperatorTransformation{manifold_dim, O1, O2, T} <:
       AbstractRealValuedOperator{manifold_dim}
    operator_1::O1
    operator_2::O2
    transformation::T

    function BinaryOperatorTransformation(
        operator_1::O1, operator_2::O2, transformation::T
    ) where {
        manifold_dim,
        O1 <: AbstractRealValuedOperator{manifold_dim},
        O2 <: AbstractRealValuedOperator{manifold_dim},
        T <: Function,
    }
        return new{manifold_dim, O1, O2, T}(operator_1, operator_2, transformation)
    end

    function Base.:+(
        operator_1::AbstractRealValuedOperator, operator_2::AbstractRealValuedOperator
    )
        return BinaryOperatorTransformation(operator_1, operator_2, (x, y) -> x + y)
    end

    function Base.:-(
        operator_1::AbstractRealValuedOperator, operator_2::AbstractRealValuedOperator
    )
        return BinaryOperatorTransformation(operator_1, operator_2, (x, y) -> x - y)
    end
end

"""
  UnitaryFormTransformation{manifold_dim, form_rank, expression_rank, G, F, T} <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}

Structure holding the necessary information to evaluate a unitary, algebraic transformation
of a `AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}`.

# Fields
- `form::F`: The form to which the transformation is applied.
- `transformation::T`: The transformation to apply to the form.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the operator is defined.
- `form_rant::Int`: The rank of the differential form.
- `expression_rank::Int`: The expression rank of the differential form.
- `G <: AbstractGeometry{manifold_dim}`: The type of the geometry where the form is
  defined.
- `T <: Function`: Function defining the algebraic transformation.

# Inner constructors
- `UnitaryFormTransformation(form::F, transformation::T)`: General constructor.
- `Base.:*(factor::Number, form::AbstractFormExpression)`: Alias for the
  multiplication of a form with a constant factor.
- `Base.:-(operator::AbstractFormExpression)`: Alias for the additive inverse of a
  real-valued form.
"""
struct UnitaryFormTransformation{manifold_dim, form_rank, expression_rank, G, F, T} <:
       AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
    form::F
    transformation::T
    label::String

    function UnitaryFormTransformation(
        form::F, transformation::T, label::String
    ) where {manifold_dim, form_rank, expression_rank, G, F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}, T <: Function}
        return new{manifold_dim, form_rank, expression_rank, G, F, T}(form, transformation, label)
    end

    function Base.:*(factor::Number, form::AbstractFormExpression)
        return UnitaryFormTransformation(form, x -> factor .* x, string(factor)*"×"*get_label(form))
    end

    Base.:-(form::AbstractFormExpression) = -1.0 * form
end

"""
  BinaryFormTransformation{manifold_dim, form_rank, expression_rank, F1, F2, G, T} <:
  AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}

Structure holding the necessary information to evaluate a binary, algebraic transformation
acting on two expression differential forms.

# Fields
- `form_1::F1`: The first expression differential form.
- `form_2::F2`: The second expression differential form.
- `transformation::T`: The transformation to apply to the differential forms.
- `label::String`: The label to associate to the resulting differential form.

# Type parameters
- `manifold_dim::Int`: The dimension of the manifold where the form expressions are defined.
- `form_rank::Int`: The rank of both differential form expressions.
- `expression_rank::Int`: The expression rank of both differential form expressions.
- `F1 <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}`: The type of the first form expression.
- `F2 <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}`: The type of the second form expression.
- `G <: AbstractGeometry{manifold_dim}`: The type of the geometry where both form expressions are
  defined.
- `T <: Function`: The type of the algebraic transformation.

# Inner constructors
- `BinaryFormTransformation(form_1::F1, form_2::F2, transformation::T, label::String)`: General
  constructor.
- `Base.:+(form_1::F1, form_2::F2)`: Alias for the sum of two differential form expressions.
- `Base.:-(form_1::F1, form_2::F2)`: Alias for the difference of two differential form expressions.
"""
struct BinaryFormTransformation{manifold_dim, form_rank, expression_rank, F1, F2, G, T} <:
       AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
    form_1::F1
    form_2::F2
    transformation::T
    label::String

    function BinaryFormTransformation(
        form_1::F1, form_2::F2, transformation::T, label::String
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G,
        F1 <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
        F2 <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
        T <: Function,
    }
        # Check if both forms contain the same forms in their tree
        tree_form_1 = get_form_space_tree(form_1)
        tree_form_2 = get_form_space_tree(form_2)

        if !(tree_form_1 === tree_form_2)
            throw(ArgumentError("Both forms in the binary transformation must contain the same forms in their tree."))
        end

        label = "(" * get_label(form_1) * label * get_label(form_2) * ")"

        return new{manifold_dim, form_rank, expression_rank, F1, F2, G, T}(
            form_1, form_2, transformation, label
        )
    end

    function Base.:+(form_1::AbstractFormExpression, form_2::AbstractFormExpression)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x + y, "+")
    end

    function Base.:-(form_1::AbstractFormExpression, form_2::AbstractFormExpression)
        return BinaryFormTransformation(form_1, form_2, (x, y) -> x - y, "-")
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_forms(bin_trans::BinaryFormTransformation) = bin_trans.form_1, bin_trans.form_2
get_geometry(bin_trans::BinaryFormTransformation) = get_geometry(get_forms(bin_trans)...)
get_form(unit_form::UnitaryFormTransformation) = unit_form.form
get_operator(unit_trans::UnitaryOperatorTransformation) = unit_trans.operator
get_transformation(unit_trans::UnitaryOperatorTransformation) = unit_trans.transformation
get_transformation(unit_form::UnitaryFormTransformation) = unit_form.transformation
get_transformation(bin_trans::BinaryOperatorTransformation) = bin_trans.transformation
get_transformation(bin_trans::BinaryFormTransformation) = bin_trans.transformation

function get_operators(bin_trans::BinaryOperatorTransformation)
    return bin_trans.operator_1, bin_trans.operator_2
end

function get_estimated_nnz_per_elem(unit_trans::UnitaryOperatorTransformation)
    return get_estimated_nnz_per_elem(get_operator(unit_trans))
end

function get_num_evaluation_elements(unit_trans::UnitaryOperatorTransformation)
    return get_num_evaluation_elements(get_operator(unit_trans))
end

function get_num_elements(unit_trans::UnitaryOperatorTransformation)
    return get_num_elements(get_operator(unit_trans))
end

function get_num_elements(bin_trans::BinaryOperatorTransformation)
    return get_num_elements(get_operators(bin_trans)[1])
end

function get_estimated_nnz_per_elem(bin_trans::BinaryOperatorTransformation)
    # WARNING: Here we assume that that both operator are acting on the same form of
    # expression rank 1, meaning they have the same basis, otherwise this would need to
    # become expression_rank=2 to make sense.
    return get_estimated_nnz_per_elem(get_operators(bin_trans)[1])
end

function get_num_evaluation_elements(bin_trans::BinaryOperatorTransformation)
    # WARNING: Here we assume that that both operator are acting on the same form of
    # expression rank 1, meaning they have the same basis, otherwise this would need to
    # become expression_rank=2 to make sense.
    return get_num_evaluation_elements(get_operators(bin_trans)[1])
end

"""
    get_form_space_tree(uni_trans::UnitaryFormTransformation)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the unitary transformation, e.g., for
`c*((α ∧ β) + γ)`, it returns the spaces of `α`, `β`, and `γ`, if all have expression_rank > 1. 
If `α` has expression_rank = 0, it returns only the spaces of `β` and `γ`.

# Arguments
- `uni_trans::UnitaryFormTransformation`: The unitary transformation structure.

# Returns
- `Tuple(<:AbstractFormExpression)`: The list of form spaces present in the tree of the unitary transformation.
"""
function get_form_space_tree(uni_trans::UnitaryFormTransformation)
    return get_form_space_tree(uni_trans.form)
end

"""
    get_form_space_tree(bin_trans::BinaryFormTransformation)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the binary transformation, e.g., for
`(α ∧ β) + γ`, it returns the spaces of `α`, `β`, and `γ`, if all have exprssion_rank > 1. If `α` has expression_rank = 0, 
it returns only the spaces of `β` and `γ`.

# Arguments
- `bin_trans::BinaryFormTransformation`: The binary transformation structure.

# Returns
- `Tuple(<:AbstractFormExpression)`: The list of forms present in the tree of the binary transformation.
"""
function get_form_space_tree(bin_trans::BinaryFormTransformation)
    tree_form_1 = get_form_space_tree(bin_trans.form_1)
    tree_form_2 = get_form_space_tree(bin_trans.form_2)

    if !(tree_form_1 === tree_form_2)
        throw(ArgumentError("Both forms in the binary transformation must contain the same forms in their tree."))
    end

    # We can now safely return the tree of just one of the forms, since the trees are the same.
    return tree_form_1
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(unit_trans::UnitaryOperatorTransformation, element_id::Int)
    operator = get_operator(unit_trans)
    transformation = get_transformation(unit_trans)
    eval, indices = evaluate(operator, element_id)
    eval .= transformation(eval)

    return eval, indices
end

function evaluate(bin_trans::BinaryOperatorTransformation, element_id::Int)
    operators = get_operators(bin_trans)
    transformation = get_transformation(bin_trans)
    # WARNING: Here we assume that that both operator are acting on the same form of
    # expression rank 1, meaning they have the same basis, otherwise this would need to
    # become expression_rank=2 to make sense.
    eval_1, indices = evaluate(operators[1], element_id)
    eval_2, _ = evaluate(operators[2], element_id)
    eval = transformation(eval_1, eval_2)

    return eval, indices
end

function evaluate(
    uni_trans::UnitaryFormTransformation{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    form = get_form(uni_trans)
    transformation = get_transformation(uni_trans)
    form_eval, indices = evaluate(form, element_id, xi)
    eval = transformation(form_eval)

    return eval, indices
end

function evaluate(
    bin_trans::BinaryFormTransformation{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    forms = get_forms(bin_trans)
    transformation = get_transformation(bin_trans)
    # WARNING: Here we assume that that both forms are defined on the same spaces, otherwise
    # This binary operation would not make sense. This means that both indices are the same.
    form_1_eval, indices = evaluate(forms[1], element_id, xi)
    form_2_eval, indices = evaluate(forms[2], element_id, xi)
    eval = transformation(form_1_eval, form_2_eval)

    return eval, indices
end