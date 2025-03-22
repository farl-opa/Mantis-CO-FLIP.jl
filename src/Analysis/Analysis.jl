"""
    module Analysis

Contains all analysis-related structs and functions.
"""
module Analysis

import .. Geometry
import .. Forms
import .. Quadrature

"""
    integrate(masked_qrule, element_idx_base, form_expression)

Integrate a volume-form expression over a masked quadrature rule.

# Arguments
- `masked_qrule::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}`: The masked quadrature rule.
- `element_idx_base::Int`: The base element index.
- `form_expression::Forms.AbstractFormExpression{manifold_dim, manifold_dim, expression_rank, G}`: The form expression to integrate.

# Returns
- `val::Float64`: The integral value.
- `idxs::Vector{Vector{Int}}`: The indices of the integral value.
"""
function integrate(
    masked_qrule::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    element_idx_base::Int,
    form_expression::Forms.AbstractFormExpression{manifold_dim, manifold_dim, expression_rank, G}
) where {manifold_dim, expression_rank, G}

    # get all evaluation/quadrature elements for the given base element
    element_idxs = Quadrature.get_element_idxs(
        masked_qrule, element_idx_base
    )

    if isempty(element_idxs)
        return Float64[], Vector{Int}[] # val, idxs
    else
        # store indices and integral values here
        idxs = [Int[] for _ in 1:expression_rank]
        val = Float64[]
        for element_idx in element_idxs
            # get element quadrature rule
            element_qrule = Quadrature.get_element_quadrature_rule(masked_qrule, element_idx)

            # evaluate the form expression
            vv, ii = Forms.evaluate(form_expression, element_idx_base, Quadrature.get_nodes(element_qrule))

            # evaluate the integral by reducing over the quadrature nodes by multiplying with the weights
            append!(val, vec(sum(Quadrature.get_weights(element_qrule) .* vv[1]; dims=1)))

            # update index list
            n_ii = length.(ii)
            for j in eachindex(idxs)
                append!(idxs[j], zeros(Int, prod(n_ii)))
            end
            count = prod(n_ii)-1
            for idx in Iterators.product([1:l for l in n_ii]...)
                for j in eachindex(idxs)
                    idxs[j][end-count] = ii[j][idx[j]]
                end
                count -= 1
            end
        end

        return val, idxs
    end
end

function integrate(
    masked_qrule::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    element_idx_base::Int,
    form_expression::Forms.AbstractFormExpression{manifold_dim, manifold_dim, 0, G}
) where {manifold_dim, G}

    # get all evaluation elements for the given base element
    element_idxs = Quadrature.get_element_idxs(
        masked_qrule, element_idx_base
    )

    if isempty(element_idxs)
        return 0.0, [[1]]
    else
        # store the integral value here
        val = 0.0
        for element_idx in element_idxs
            # get element quadrature rule
            element_qrule = Quadrature.get_element_quadrature_rule(masked_qrule, element_idx)

            # evaluate the form expression
            vv, _ = Forms.evaluate(form_expression, element_idx_base, Quadrature.get_nodes(element_qrule))

            # evaluate the integral by reducing over the quadrature nodes
            val += sum(Quadrature.get_weights(element_qrule) .* vv[1])

        end

        return val, [[1]]
    end
end

include("ErrorComputations.jl")

end
