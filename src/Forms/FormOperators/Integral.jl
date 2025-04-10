############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    Integral{manifold_dim, F} <: AbstractRealValuedOperator{manifold_dim}

Structure representing the integral of a form over a manifold.

# Fields
- `form::F`: The form expression to be integrated.

# Type Parameters
- `manifold_dim::Int`: The dimension of the manifold.
- `F`: The type of the form expression.
# Inner Constructors
- `Integral(form::AbstractFormExpression{manifold_dim})`: General constructor.
# Outer Constructors
- `∫`: Symbolic wrapper for the integral operator.
"""
struct Integral{manifold_dim, F} <: AbstractRealValuedOperator{manifold_dim}
    form::AbstractFormExpression{manifold_dim}
    function Integral(
        form::F
    ) where {manifold_dim, F <: AbstractFormExpression{manifold_dim, manifold_dim}}
        return new{manifold_dim, F}(form)
    end
end

"""
    ∫(form::AbstractFormExpression)

Symbolic wrapper for the integral operator.

# Arguments
- `form::AbstractFormExpression`: The form to be integrated.

# Returns
- `Integral`: The integral operator.
"""
∫(form::AbstractFormExpression) = Integral(form)

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_form(integral::Integral)

Returns the form associated with the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `<: AbstractFormExpression`: The form associated with the integral operator.
"""
get_form(integral::Integral) = integral.form

############################################################################################
#                                        Evaluate                                          #
############################################################################################

"""
    evaluate(
        integral::Integral{manifold_dim, F},
        element_id::Int,
        quad_rule::Quadrature.AbstractQuadratureRule{manifold_dim},
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank},
    }

Evaluates the integral of a form over a given element using a specified quadrature rule.

# Arguments
- `integral::Integral{manifold_dim, F}`: The integral operator to evaluate.
- `element_id::Int`: The element over which to evaluate the integral.
- `quad_rule::Quadrature.AbstractQuadratureRule{manifold_dim}`: The quadrature rule to use
    for computing the integral. 

# Returns
- `integral_eval::Vector{Float64}`: The evaluated integral.
- `integral_indices::Vector{Vector{Int}}`: The indices of the evaluated integral. The length
    of the outer vector depends on the `expression_rank` of the form expression.
"""
function evaluate(
    integral::Integral{manifold_dim, F},
    element_id::Int,
    quad_rule::Quadrature.AbstractQuadratureRule{manifold_dim},
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank},
}
    form = get_form(integral)
    nodes = Quadrature.get_nodes(quad_rule)
    weights = Quadrature.get_weights(quad_rule)
    form_eval, form_indices = evaluate(form, element_id, nodes)
    integral_eval = vec(sum(weights .* form_eval[1]; dims=1))
    integral_indices = [Int[] for _ in 1:expression_rank]
    num_indices = length.(form_indices)
    for j in eachindex(integral_indices)
        append!(integral_indices[j], zeros(Int, prod(num_indices)))
    end

    count = prod(num_indices) - 1
    for id in Iterators.product([1:l for l in num_indices]...)
        for j in eachindex(integral_indices)
            integral_indices[j][end - count] = form_indices[j][id[j]]
        end
        count -= 1
    end

    return integral_eval, integral_indices
end
