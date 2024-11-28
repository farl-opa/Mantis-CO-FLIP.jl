@doc raw"""
    tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F, rule_args_1d...) where {domain_dim, F <: Function}

Returns a tensor product quadrature rule of given degree and rule type.

# Arguments
- `p::NTuple{domain_dim, Int}`: Degree of the quadrature rule per dimension.
- `quad_rule::F`: The function that returns a `QuadratureRule{1}` given 
                  an integer degree. May take additional arguments.
- `rule_args_1d...`: Additional arguments for the 1D quadrature rule. Optional.

# Returns
- `::QuadratureRule{domain_dim}`: QuadratureRule of the new dimension.
"""
function tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F, rule_args_1d...) where {domain_dim, F <: Function}
    # Compute the nodes and weights per dimensions for given rule type 
    # and degree.
    qrules = NTuple{domain_dim, QuadratureRule{1}}(quad_rule(p[k], rule_args_1d...) for k = 1:domain_dim)
    points = NTuple{domain_dim, Vector{Float64}}(get_quadrature_nodes(qrules[k])[1] for k = 1:domain_dim)
    weights_1d = NTuple{domain_dim, Vector{Float64}}(get_quadrature_weights(qrules[k]) for k = 1:domain_dim)
    
    # Compute the tensor product of the quadrature weights.
    weights = _compute_tensor_product_weights(weights_1d)
    
    # Create the label. If only one rule is used, the label is the same 
    # as the rule type of the 1D rule.
    if domain_dim == 1
        rule_type = get_quadrature_rule_type(qrules[1])
    else
        rule_type = "Tensor-product of $domain_dim $(get_quadrature_rule_type(qrules[1])) rules"
    end

    return QuadratureRule{domain_dim}(points, weights, rule_type)
end

@doc raw"""
    tensor_product_rule(qrules_1d::NTuple{domain_dim, QuadratureRule{1}}) where {domain_dim}

Returns a tensor product quadrature rule from the given rules.

# Arguments
- `qrules_1d::NTuple{domain_dim, QuadratureRule{1}}`: Quadrature rules 
                                                      per dimension.

# Returns
- `::QuadratureRule{domain_dim}`: QuadratureRule of the new dimension.
"""
function tensor_product_rule(qrules_1d::NTuple{domain_dim, QuadratureRule{1}}) where {domain_dim}
    # Compute the nodes and weights per dimensions for given rule type 
    # and degree.
    points = NTuple{domain_dim, Vector{Float64}}(get_quadrature_nodes(qrules_1d[k])[1] for k = 1:domain_dim)
    weights_1d = NTuple{domain_dim, Vector{Float64}}(get_quadrature_weights(qrules_1d[k]) for k = 1:domain_dim)
    
    # Compute the tensor product of the quadrature weights.
    weights = _compute_tensor_product_weights(weights_1d)
    
    # Create the label. If only one rule is used, the label is the same 
    # as the rule type of the 1D rule.
    if domain_dim == 1
        rule_type = get_quadrature_rule_type(qrules_1d[1])
    else
        rule_types = join([get_quadrature_rule_type(qrules_1d[i]) for i = 1:domain_dim], ", ")
        rule_type = "Tensor-product of ($rule_types) rules"
    end
    
    return QuadratureRule{domain_dim}(points, weights, rule_type)
end

@doc raw"""
    _compute_tensor_product_weights(weights_1d::NTuple{domain_dim, Vector{T}}) where {domain_dim, T <: Number}

Compute the tensor product of the given 1D quadrature weights.

# Arguments
- `weights_1d::NTuple{domain_dim, Vector{T}}`: Quadrature weights per dimension.

# Returns
- `::Vector{T}`: Tensor product of the quadrature weights.
"""
function _compute_tensor_product_weights(weights_1d::NTuple{domain_dim, Vector{T}}) where {domain_dim, T <: Number}
    weights = Vector{T}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end
    return weights
end