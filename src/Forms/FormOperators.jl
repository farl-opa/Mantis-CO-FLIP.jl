# Priority 1: Inner products for abstract form fields

# 0-forms in any dimension
function inner_product(f1::AbstractFormExpression{domain_dim, 0, G}, f2::AbstractFormExpression{domain_dim, 0, G}, element_idx::Int, quad_rule::Quadrature.QuadratureRule{domain_dim}) where {domain_dim, G <: Geometry.AbstractGeometry{domain_dim}}
    # Quadrature information.
    xi = Quadrature.get_quadrature_nodes(quad_rule)  # Tuple of quad nodes.
    weights = Quadrature.get_quadrature_weights(quad_rule)  # Vector of tensor product weights.

    # Forms evaluated at quadrature nodes.
    f1_eval, f1_indices = evaluate(f1, element_idx, xi)
    f2_eval, f2_indices = evaluate(f2, element_idx, xi)

    # Count the number of active basis functions.
    n_basis_1 = size(f1_eval[1], 2)  # number of basis functions in f1, if field only 1, if space then the number of basis in this component
    n_basis_2 = size(f2_eval[1], 2)  # the same as above 
    n_basis_total = n_basis_1 * n_basis_2

    # Pre-allocate the inner product data: row indices, colum indices, and values.
    A_row_idx = Vector{Int}(undef, n_basis_total)
    A_column_idx = Vector{Int}(undef, n_basis_total)
    A_values = Vector{Float64}(undef, n_basis_total)

    # Get geometry information, immediately checking if f1 and f2 have the same geometry.
    _, det_g = Geometry.metric(get_geometry(f1, f2), element_idx, xi)
    for (f1_linear, f1_basis_idx) in enumerate(f1_indices)
        for (f2_linear, f2_basis_idx) in enumerate(f2_indices)
            inner_prod_idx = f2_linear + (f1_linear - 1) * n_basis_1  # get the linear index of the inner product (f1_basis_idx, f2_basis_idx)

            A_row_idx[inner_prod_idx] = f1_indices[1][f1_basis_idx]
            A_column_idx[inner_prod_idx] = f2_indices[1][f2_basis_idx]

            # Compute the value of the inner product. Note that 
            # `A_values` is initialised as undef, so we cannot add the 
            # values directly.
            inner_product_value = 0.0
            for quad_node_idx in eachindex(weights)
                inner_product_value += det_g[quad_node_idx] * weights[quad_node_idx] * f1_eval[1][quad_node_idx, f1_basis_idx] * f2_eval[1][quad_node_idx, f2_basis_idx]
            end
            A_values[inner_prod_idx] = inner_product_value
            
        end
    end

    return (A_row_idx, A_column_idx, A_values)
end

# 1-forms in 2D
function inner_product(f1::AbstractFormExpression{2, 1, G}, f2::AbstractFormExpression{2, 1, G}, element_idx::Int, quad_rule::Quadrature.QuadratureRule{2}) where {G <: Geometry.AbstractGeometry{2}}
    # Quadrature information.
    xi = Quadrature.get_quadrature_nodes(quad_rule)  # Tuple of quad nodes.
    weights = Quadrature.get_quadrature_weights(quad_rule)  # Vector of tensor product weights.

    # Forms evaluated at quadrature nodes.
    f1_eval, f1_indices = evaluate(f1, element_idx, xi)
    f2_eval, f2_indices = evaluate(f2, element_idx, xi)

    # We need to recheck this, we need the number of active basis
    # Count the number of active basis functions.
    n_basis_1 = size(f1_eval[1], 2)  # number of basis functions in f1, if field only 1, if space then the number of basis in this component
    n_basis_2 = size(f2_eval[1], 2)  # the same as above 
    n_basis_total = n_basis_1 * n_basis_2

    # Pre-allocate the inner product data: row indices, colum indices, and values.
    A_row_idx = Vector{Int}(undef, n_basis_total)
    A_column_idx = Vector{Int}(undef, n_basis_total)
    A_values = Vector{Float64}(undef, n_basis_total)

    A_values = zeros(n_basis_1, n_basis_2, 2, 2)
    A_row_idx = zeros(n_basis_1, n_basis_2, 2, 2)
    A_column_idx = zeros(n_basis_1, n_basis_2, 2 ,2)

    # Get geometry information, immediately checking if f1 and f2 have the same geometry.
    g, det_g = Geometry.metric(get_geometry(f1, f2), element_idx, xi)

    for f1_basis_idx = 1:n_basis_1
        for f2_basis_idx = 1:n_basis_2
            # α¹ = α¹₁dξ¹ + α¹₂dξ²
            # β¹ = β¹₁dξ¹ + β¹₂dξ²
            # <α¹, β²> = <α¹₁dξ¹, β¹₁dξ¹> + <α¹₁dξ¹, β¹₂dξ²> + ...

            # <α¹₁dξ¹, β¹₁dξ¹>
            A_values[f1_basis_idx, f2_basis_idx, 1, 1] += sum(f1_eval[1][:, f1_basis_idx] .* g[:, 1, 1] .* f2_eval[1][:, f2_basis_idx])
            A_row_idx[f1_basis_idx, f2_basis_idx, 1, 1] = f1_indices[1][f1_basis_idx]
            A_column_idx[f1_basis_idx, f2_basis_idx, 1, 1] = f2_indices[1][f2_basis_idx]

            
            # <α¹₁dξ¹, β¹₂dξ²>
            A_values[f1_basis_idx, f2_basis_idx, 1, 2] += sum(f1_eval[1][:, f1_basis_idx] .* g[:, 1, 2] .* f2_eval[2][:, f2_basis_idx])
            A_row_idx[f1_basis_idx, f2_basis_idx, 1, 2] = f1_indices[1][f1_basis_idx]
            A_column_idx[f1_basis_idx, f2_basis_idx, 1, 2] = f2_indices[2][f2_basis_idx]

            # the other terms (...)
        end
    end

    return 
end

# n-forms in any dimension (dimension larger than 0, because the method 
# signature will be identical to the 0-form case when the dimension is 
# 0). Note that this is not prevented at the moment.
function inner_product(f1::AbstractFormExpression{domain_dim, domain_dim, G}, f2::AbstractFormExpression{domain_dim, domain_dim, G}, element_idx::Int, quad_rule::Quadrature.QuadratureRule{domain_dim}) where {domain_dim, G <: Geometry.AbstractGeometry{domain_dim}}
    # Quadrature information.
    xi = Quadrature.get_quadrature_nodes(quad_rule)  # Tuple of quad nodes.
    weights = Quadrature.get_quadrature_weights(quad_rule)  # Vector of tensor product weights.

    # Forms evaluated at quadrature nodes.
    f1_eval, f1_indices = evaluate(f1, element_idx, xi)
    f2_eval, f2_indices = evaluate(f2, element_idx, xi)

    # Count the number of active basis functions.
    n_basis_1 = size(f1_eval[1], 2)  # number of basis functions in f1, if field only 1, if space then the number of basis in this component
    n_basis_2 = size(f2_eval[1], 2)  # the same as above 
    n_basis_total = n_basis_1 * n_basis_2

    # Pre-allocate the inner product data: row indices, colum indices, and values.
    A_row_idx = Vector{Int}(undef, n_basis_total)
    A_column_idx = Vector{Int}(undef, n_basis_total)
    A_values = Vector{Float64}(undef, n_basis_total)

    # Get geometry information, immediately checking if f1 and f2 have the same geometry.
    _, det_g = Geometry.metric(get_geometry(f1, f2), element_idx, xi)
    for (f1_linear, f1_basis_idx) in enumerate(f1_indices)
        for (f2_linear, f2_basis_idx) in enumerate(f2_indices)
            inner_prod_idx = f2_linear + (f1_linear - 1) * n_basis_1  # get the linear index of the inner product (f1_basis_idx, f2_basis_idx)

            A_row_idx[inner_prod_idx] = f1_indices[1][f1_basis_idx]
            A_column_idx[inner_prod_idx] = f2_indices[1][f2_basis_idx]

            # Compute the value of the inner product. Note that 
            # `A_values` is initialised as undef, so we cannot add the 
            # values directly.
            inner_product_value = 0.0
            for quad_node_idx in eachindex(weights)
                inner_product_value += (1.0/det_g[quad_node_idx]) * weights[quad_node_idx] * f1_eval[1][quad_node_idx, f1_basis_idx] * f2_eval[1][quad_node_idx, f2_basis_idx]
            end
            A_values[inner_prod_idx] = inner_product_value
            
        end
    end

    return (A_row_idx, A_column_idx, A_values)
end

# (0-forms, 0-forms)
function inner_product(f1::FormSpace{domain_dim, 0, G, Tuple{F1}}, f2::FormSpace{domain_dim, 0, G, Tuple{F2}}, element_id::Int, quad_rule::Quadrature.QuadratureRule{domain_dim}) where {domain_dim, G <: Geometry.AbstractGeometry{domain_dim}, F1 <: FunctionSpaces.AbstractFunctionSpace, F2 <: FunctionSpaces.AbstractFunctionSpace}

    # Compute bases
    basis_1_eval, basis_1_inds = FunctionSpaces.evaluate(f1.fem_space[1], element_id, quad_rule.xi)
    basis_2_eval, basis_2_inds = FunctionSpaces.evaluate(f2.fem_space[1], element_id, quad_rule.xi)

    # Compute the quantities related to the geometry.
    # The geometries of f1 and f2 are assumed equal, so we use one. 
    # We need to add a check for the equality of both geometries
    _, _, sqrt_g = Geometry.inv_metric(f1.geometry, element_id, quad_rule.xi, nderivatives)

    # Count the number of supported basis on this element.
    n_basis_1 = length(basis_1_inds)
    n_basis_2 = length(basis_2_inds)
    n_basis_total = n_basis_1 * n_basis_2

    # Pre-allocate the inner product data: row indices, colum indices, and values).
    M_row = Vector{Int}(undef, n_basis_total)
    M_col = Vector{Int}(undef, n_basis_total)
    M_val = Vector{Float64}(undef, n_basis_total)

    # Compute the inner products
    # Inner product M_{i,j} = (f^{1}_{i}, f^{2}_{j})
    # I change the order of the indices with respect to the one for n-forms.
    # The reason derives from the definition of the inner product above: 
    # f^{1} is on the left and f^{2} on the right, therefore the index for 
    # the basis of f^{1} is associated to the rows and the index for the basis 
    # of f^{2} is associated to the columns.
    for basis_2_idx in 1:n_basis_2
        for basis_1_idx in 1:n_basis_1
            inner_prod_idx = basis_1_idx + (basis_2_idx - 1) * n_basis_2  # get the linear index of the inner product (basis_1_idx, basis_2_idx)

            M_row[inner_prod_idx] = basis_1_inds[basis_1_idx]
            M_col[inner_prod_idx] = basis_2_inds[basis_2_idx]

            M_val[inner_prod_idx] = 0.0
            for quad_node_idx in eachindex(quad_rule.weights)
                M_val[inner_prod_idx] += (quad_rule.weights[quad_node_idx]/sqrt_g[quad_node_idx]) * basis_1_eval[quad_node_idx, basis_1_idx] * basis_2_eval[quad_node_idx, basis_2_idx]
            end
        end
    end

    return M_row, M_col, M_val
end

# (n-forms, n-forms)
function inner_product(f1::FormSpace{domain_dim, domain_dim, G, Tuple{F1}}, f2::FormSpace{domain_dim, domain_dim, G, Tuple{F2}}, element_id::Int, quad_rule::Quadrature.QuadratureRule{domain_dim}) where {domain_dim, G <: Geometry.AbstractGeometry{domain_dim}, F1 <: FunctionSpaces.AbstractFunctionSpace, F2 <: FunctionSpaces.AbstractFunctionSpace}

    # Compute bases and their derivatives.
    basis_1_eval, basis_1_inds = FunctionSpaces.evaluate(f1.fem_space[1], element_id, quad_rule.xi)
    basis_2_eval, basis_2_inds = FunctionSpaces.evaluate(f2.fem_space[1], element_id, quad_rule.xi)

    # Compute the quantities related to the geometry.
    # The geometries of f1 and f2 are assumed equal, so we use one. 
    # We need to add a check for the equality of both geometries
    _, _, sqrt_g = Geometry.inv_metric(f1.geometry, element_id, quad_rule.xi, nderivatives)

    # Count the number of supported basis on this element.
    n_basis_1 = length(basis_1_inds)
    n_basis_2 = length(basis_2_inds)
    n_total = n_basis_1 * n_basis_2

    # Pre-allocate the matrices (their row, colum, and value vectors).
    M_row = Vector{Int}(undef, n_total)
    M_col = Vector{Int}(undef, n_total)
    M_val = Vector{Float64}(undef, n_total)

    for j in 1:n_basis_2
        for i in 1:n_basis_1
            inner_prod_idx = i + (j-1) * n_basis_1

            M_row[inner_prod_idx] = basis_1_inds[i]
            M_col[inner_prod_idx] = basis_2_inds[j]

            Mij = 0.0
            for quad_node_idx in eachindex(quad_rule.weights)
                Mij += (quad_rule.weights[quad_node_idx]/sqrt_g[quad_node_idx]) * basis_1_eval[quad_node_idx,i] * basis_2_eval[quad_node_idx,j]
            end

            M_val[inner_prod_idx] = Mij
        end
    end

    return M_row, M_col, M_val
end

# (1-forms, 1-forms) in 2D
function inner_product(f1::FormSpace{2,1,G,Tuple{F1x,F1y}}, f2::FormSpace{2,1,G,Tuple{F2x,F2y}}, element_id::Int, quad_rule::Quadrature.QuadratureRule{2}) where {G<:Geometry.AbstractGeometry{2}, F1x <: FunctionSpaces.AbstractFunctionSpace, F1y <: FunctionSpaces.AbstractFunctionSpace, F2x <: FunctionSpaces.AbstractFunctionSpace, F2y <: FunctionSpaces.AbstractFunctionSpace}



end

# (1-forms, 1-forms) in 2D with expressions
function inner_product(f1::AbstractFormSpace{2,1,G}, f2::AbstractFormSpace{2,1,G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{2}) where {G<:Geometry.AbstractGeometry{2}}

    # 1. evaluate f1 and f2 using evaluate methods for AbstractFormSpaces
    # 2. compute inner product

end

# 1-forms in 3D
function inner_product(f1::FormSpace{3,1,G,Tuple{F1x,F1y,F1z}}, f2::FormSpace{3,1,G,Tuple{F2x,F2y,F2z}}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3}, F1x <: FunctionSpaces.AbstractFunctionSpace, F1y <: FunctionSpaces.AbstractFunctionSpace, F1z <: FunctionSpaces.AbstractFunctionSpace, F2x <: FunctionSpaces.AbstractFunctionSpace, F2y <: FunctionSpaces.AbstractFunctionSpace, F2z <: FunctionSpaces.AbstractFunctionSpace}



end

# 2-forms in 3D
function inner_product(f1::FormSpace{3,2,G,Tuple{F1x,F1y,F1z}}, f2::FormSpace{3,2,G,Tuple{F2x,F2y,F2z}}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3}, F1x <: FunctionSpaces.AbstractFunctionSpace, F1y <: FunctionSpaces.AbstractFunctionSpace, F1z <: FunctionSpaces.AbstractFunctionSpace, F2x <: FunctionSpaces.AbstractFunctionSpace, F2y <: FunctionSpaces.AbstractFunctionSpace, F2z <: FunctionSpaces.AbstractFunctionSpace}



end

# Priority 2: Wedge products

# Priority 3: Triple products