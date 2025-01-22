@doc raw"""
    tensor_product_rule(p::NTuple{manifold_dim, Int}, quad_rule::F, rule_args_1d...) where {
        manifold_dim, F <: Function
    }

Returns a tensor product quadrature rule of given degree and rule type.

# Arguments
- `p::NTuple{manifold_dim, Int}`: Degree of the quadrature rule per dimension.
- `quad_rule::F`: The function that returns a `QuadratureRule{1}` given an integer degree.
    May take additional arguments.
- `rule_args_1d...`: Additional arguments for the 1D quadrature rule. Optional.

# Returns
- `::QuadratureRule{manifold_dim}`: QuadratureRule of the new dimension.
"""
function tensor_product_rule(
    p::NTuple{manifold_dim, Int}, quad_rule::F, rule_args_1d...
) where {manifold_dim, F <: Function}
    qrules = NTuple{manifold_dim, QuadratureRule{1}}(
        quad_rule(p[k], rule_args_1d...) for k = 1:manifold_dim
    )
    points = NTuple{manifold_dim, Vector{Float64}}(
        get_quadrature_nodes(qrules[k])[1] for k = 1:manifold_dim
    )
    weights_1d = NTuple{manifold_dim, Vector{Float64}}(
        get_quadrature_weights(qrules[k]) for k = 1:manifold_dim
    )

    weights = _compute_tensor_product(weights_1d)

    # If only one rule is used, the label is the same as the label of the 1D rule. This
    # ensures that the tensor product of a 1d rule is the same as the 1d rule.
    if manifold_dim == 1
        rule_label = get_quadrature_rule_label(qrules[1])
    else
        qrule_label = get_quadrature_rule_label(qrules[1])
        rule_label = "Tensor-product of $manifold_dim $(qrule_label) rules"
    end

    return QuadratureRule{manifold_dim}(points, weights, rule_label)
end

@doc raw"""
    tensor_product_rule(qrules_1d::NTuple{manifold_dim, QuadratureRule{1}}) where {
        manifold_dim
    }

Returns a tensor product quadrature rule from the given rules.

# Arguments
- `qrules_1d::NTuple{manifold_dim, QuadratureRule{1}}`: Quadrature rules per dimension.

# Returns
- `::QuadratureRule{manifold_dim}`: QuadratureRule of the new dimension.
"""
function tensor_product_rule(qrules_1d::NTuple{manifold_dim, QuadratureRule{1}}) where {
    manifold_dim
}
    points = NTuple{manifold_dim, Vector{Float64}}(
        get_quadrature_nodes(qrules_1d[k])[1] for k = 1:manifold_dim
    )
    weights_1d = NTuple{manifold_dim, Vector{Float64}}(
        get_quadrature_weights(qrules_1d[k]) for k = 1:manifold_dim
    )

    weights = _compute_tensor_product(weights_1d)

    # If only one rule is used, the label is the same as the label of the 1D rule. This
    # ensures that the tensor product of a 1d rule is the same as the 1d rule.
    if manifold_dim == 1
        rule_label = get_quadrature_rule_label(qrules_1d[1])
    else
        rule_labels = join(
            [get_quadrature_rule_label(qrules_1d[i]) for i = 1:manifold_dim], ", "
        )
        rule_label = "Tensor-product of ($rule_labels) rules"
    end

    return QuadratureRule{manifold_dim}(points, weights, rule_label)
end

@doc raw"""
    _compute_tensor_product(weights_1d::NTuple{manifold_dim, Vector{T}}) where {
        manifold_dim, T <: Number
    }

Compute the tensor product of the given 1D quadrature weights.

# Arguments
- `weights_1d::NTuple{manifold_dim, Vector{T}}`: Quadrature weights per dimension.

# Returns
- `::Vector{T}`: Tensor product of the quadrature weights.
"""
function _compute_tensor_product(weights_1d::NTuple{manifold_dim, Vector{T}}) where {
    manifold_dim, T <: Number
}
    weights = Vector{T}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end

    return weights
end
