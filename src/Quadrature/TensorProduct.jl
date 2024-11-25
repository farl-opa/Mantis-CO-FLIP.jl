@doc raw"""
    tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F) where {domain_dim, F <: Function}

Returns a tensor product quadrature rule of given degree and rule type.

# Arguments
- `p::NTuple{domain_dim, Int}`: Degree of the quadrature rule per dimension.
- `quad_rule::F`: The function that returns a `QuadratureRule{1}` given 
                  an integer degree.

# Returns
- `::QuadratureRule{domain_dim}`: QuadratureRule of the new dimension.
"""
function tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F) where {domain_dim, F <: Function}
    # Compute the nodes and weights per dimensions for given rule type 
    # and degree.
    qrules = NTuple{domain_dim, QuadratureRule{1}}(quad_rule(p[k]) for k = 1:domain_dim)
    points = NTuple{domain_dim, Vector{Float64}}(get_quadrature_nodes(qrules[k])[1] for k = 1:domain_dim)
    weights_1d = NTuple{domain_dim, Vector{Float64}}(get_quadrature_weights(qrules[k]) for k = 1:domain_dim)
    
    # Compute the tensor product of the quadrature weights.
    weights = Vector{Float64}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end
    
    return QuadratureRule{domain_dim}(points, weights, "Tensor-product of $domain_dim $(get_quadrature_rule_type(qrules[1])) rules")
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
    weights = Vector{Float64}(undef, prod(size.(weights_1d, 1)))
    for (linear_idx, weights_all) in enumerate(Iterators.product(weights_1d...))
        weights[linear_idx] = prod(weights_all)
    end
    
    rule_type = join([get_quadrature_rule_type(qrules_1d[i]) for i = 1:domain_dim], ", ")
    return QuadratureRule{domain_dim}(points, weights, "Tensor-product of ($rule_type) rules")
end