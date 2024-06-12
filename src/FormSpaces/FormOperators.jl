# Priority 1: Inner products for abstract form fields

# 0-forms
function inner_product(f1::FormSpace{domain_dim, 0, G, Tuple{F1}}, f2::FormSpace{domain_dim, 0, G, Tuple{F2}}, element_id::Int, quad_rule::QuadratureRule{domain_dim}) where {domain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim} where {codomain_dim}, F1 <: FunctionSpaces.AbstractFunctionSpace{domain_dim}, F2 <: FunctionSpaces.AbstractFunctionSpace{domain_dim}}

    # Compute bases
    basis_1_eval, basis_1_inds = FunctionSpaces.evaluate(f1.fem_space[1], element_id, xi, 0)
    basis_2_eval, basis_2_inds = FunctionSpaces.evaluate(f2.fem_space[1], element_id, xi, 0)
end

# n-forms
function inner_product(f1::FormSpace{domain_dim, domain_dim, G, Tuple{F1}}, f2::FormSpace{domain_dim, domain_dim, G, Tuple{F2}}, element_id::Int, quad_rule::QuadratureRule{domain_dim}) where {domain_dim, G <: Geometry.AbstractGeometry{domain_dim, codomain_dim} where {codomain_dim}, F1 <: FunctionSpaces.AbstractFunctionSpace{domain_dim}, F2 <: FunctionSpaces.AbstractFunctionSpace{domain_dim}}

    # Compute bases and their derivatives.
    basis_eval_1, basis_inds_1 = FunctionSpaces.evaluate(f1.fem_space[1], element_id, quad_rule.xi, 0)
    basis_eval_2, basis_inds_2 = FunctionSpaces.evaluate(f2.fem_space[1], element_id, quad_rule.xi, 0)

    # Compute the quantities related to the geometry.
    _, _, sqrt_g = Geometry.inv_metric(f.geometry, element_id, quad_rule.xi, nderivatives)

    # Count the number of supported basis on this element.
    n_1 = length(basis_inds_1)
    n_2 = length(basis_inds_2)
    n_total = n_1 * n_2

    # Pre-allocate the matrices (their row, colum, and value vectors).
    M_row = Vector{Int}(undef, n_total)
    M_col = Vector{Int}(undef, n_total)
    M_val = Vector{Float64}(undef, n_total)

    for j in 1:n_2
        for i in 1:n_1
            idx = i + (j-1) * n_1

            M_row[idx] = basis_inds_1[i]
            M_col[idx] = basis_inds_2[j]

            Mij = 0.0
            for k in eachindex(quad_rule.weights)
                Mij += (quad_rule.weights[k]/sqrt_g[k]) * basis_eval_1[k,i] * basis_eval_2[k,j]
            end

            M_val[idx] = Mij
        end
    end
end

# 1-forms in 2D
function inner_product(f1::FormSpace{2,1,G,Tuple{F1x,F1y}}, f2::FormSpace{2,1,G,Tuple{F2x,F2y}}, element_id::Int, quad_rule::QuadratureRule{2}) where {G<:Geometry.AbstractGeometry{2,m} where {m}, F1x <: FunctionSpaces.AbstractFunctionSpace{2}, F1y <: FunctionSpaces.AbstractFunctionSpace{2}, F2x <: FunctionSpaces.AbstractFunctionSpace{2}, F2y <: FunctionSpaces.AbstractFunctionSpace{2}}



end

# 1-forms in 3D
function inner_product(f1::FormSpace{3,1,G,Tuple{F1x,F1y,F1z}}, f2::FormSpace{3,1,G,Tuple{F2x,F2y,F2z}}, element_id::Int, quad_rule::QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3,m} where {m}, F1x <: FunctionSpaces.AbstractFunctionSpace{3}, F1y <: FunctionSpaces.AbstractFunctionSpace{3}, F1z <: FunctionSpaces.AbstractFunctionSpace{3}, F2x <: FunctionSpaces.AbstractFunctionSpace{3}, F2y <: FunctionSpaces.AbstractFunctionSpace{3}, F2z <: FunctionSpaces.AbstractFunctionSpace{3}}



end

# 2-forms in 3D
function inner_product(f1::FormSpace{3,2,G,Tuple{F1x,F1y,F1z}}, f2::FormSpace{3,2,G,Tuple{F2x,F2y,F2z}}, element_id::Int, quad_rule::QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3,m} where {m}, F1x <: FunctionSpaces.AbstractFunctionSpace{3}, F1y <: FunctionSpaces.AbstractFunctionSpace{3}, F1z <: FunctionSpaces.AbstractFunctionSpace{3}, F2x <: FunctionSpaces.AbstractFunctionSpace{3}, F2y <: FunctionSpaces.AbstractFunctionSpace{3}, F2z <: FunctionSpaces.AbstractFunctionSpace{3}}



end

# Priority 2: Wedge products

# Priority 3: Triple products