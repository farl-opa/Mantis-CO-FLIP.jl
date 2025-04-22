############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    Integral{manifold_dim, F} <: AbstractRealValuedOperator{manifold_dim}

Structure representing the integral of a form over a manifold.

# Fields
- `form::F`: The form expression to be integrated.
- `quad_rule::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}`: The quadrature rule
    used for the integral.

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
    quad_rule::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}
    function Integral(
        form::F, quad_rule::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}
    ) where {manifold_dim, F <: AbstractFormExpression{manifold_dim, manifold_dim}}
        geom = get_geometry(form)
        if Geometry.get_num_elements(geom) != Quadrature.get_num_base_elements(quad_rule)
            throw(
                ArgumentError(
                    """The number of elements in the geometry and quadrature rule must \
                    match. The geometry has $(Geometry.get_num_elements(geom)) elements \
                    and the quadrature rule has \
                    $(Quadrature.get_num_base_elements(quad_rule)) elements."""
                ),
            )
        end

        return new{manifold_dim, F}(form, quad_rule)
    end
end

"""
    ∫(form::AbstractFormExpression)

Symbolic wrapper for the integral operator. The unicode character command is `\\int`.

# Arguments
- `form::AbstractFormExpression`: The form to be integrated.
- `quad_rule::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to be used.

# Returns
- `Integral`: The integral operator.
"""
function ∫(form::AbstractFormExpression, quad_rule::Quadrature.AbstractGlobalQuadratureRule)
    return Integral(form, quad_rule)
end

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

"""
    get_expression_rank(integral::Integral)

Returns the rank of the expression associated with the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `::Int`: The rank of the expression associated with the integral operator.
"""
get_expression_rank(integral::Integral) = get_expression_rank(get_form(integral))

"""
    get_quadrature_rule(integral::Integral)

Returns the quadrature rule associated with the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `<:Quadrature.AbstractGlobalQuadratureRule`: Returns the quadrature rule associated with
    the integral operator.
"""
get_quadrature_rule(integral::Integral) = integral.quad_rule

"""
    get_num_elements(integral::Integral)

Returns the number of elements in the geometry associated with the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `::Int`: The number of elements associated with the integral operator.
"""
function get_num_elements(integral::Integral)
    return Quadrature.get_num_base_elements(get_quadrature_rule(integral))
end

"""
    get_num_evaluation_elements(integral::Integral)

Returns the number of evaluation elements in the quadrature rule associated with the
integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `::Int`: The number of evaluation elements associated with the integral operator.
"""
function get_num_evaluation_elements(integral::Integral)
    return Quadrature.get_num_evaluation_elements(get_quadrature_rule(integral))
end

"""
    get_estimated_nnz_per_elem(integral::Integral)

Returns the estimated number of non-zero entries per element for the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `::Int`: The estimated number of non-zero entries per element associated with the integral
    operator.
"""
function get_estimated_nnz_per_elem(integral::Integral)
    return get_estimated_nnz_per_elem(get_form(integral))
end
############################################################################################
#                                        Evaluate                                          #
############################################################################################

"""
    evaluate(
        integral::Integral{manifold_dim, F},
        global_element_id::Int,
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank},
    }

Evaluates the integral of a form over a given global element using a specified quadrature
rule.

# Arguments
- `integral::Integral{manifold_dim, F}`: The integral operator to evaluate.
- `global_element_id::Int`: The global element over which to evaluate the integral.

# Returns
- `integral_eval::Vector{Float64}`: The evaluated integral.
- `integral_indices::Vector{Vector{Int}}`: The indices of the evaluated integral. The length
    of the outer vector depends on the `expression_rank` of the form expression.
"""
function evaluate(
    integral::Integral{manifold_dim, F}, global_element_id::Int
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank},
}
    quad_rule = get_quadrature_rule(integral)
    quadrature_elements = Quadrature.get_element_idxs(quad_rule, global_element_id)
    if isempty(quadrature_elements)
        return Float64[], Vector{Int}[]
    end

    integral_ids = [Int[] for _ in 1:expression_rank]
    integral_vals = Float64[]
    form = get_form(integral)
    for quad_element_id in quadrature_elements
        element_quad_rule = Quadrature.get_element_quadrature_rule(
            quad_rule, quad_element_id
        )
        element_val, element_ids = evaluate(
            form, global_element_id, Quadrature.get_nodes(element_quad_rule)
        )
        append!(
            integral_vals,
            vec(sum(Quadrature.get_weights(element_quad_rule) .* element_val[1]; dims=1)),
        )
        num_indices = length.(element_ids)
        for j in eachindex(integral_ids)
            append!(integral_ids[j], zeros(Int, prod(num_indices)))
        end

        count = prod(num_indices) - 1
        for id in Iterators.product([1:l for l in num_indices]...)
            for j in eachindex(integral_ids)
                integral_ids[j][end - count] = element_ids[j][id[j]]
            end

            count -= 1
        end
    end

    return integral_vals, integral_ids
end
