# Priority 1: Inner products for abstract form fields
#=
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

            A_row_idx[inner_prod_idx] = f1_basis_idx
            A_column_idx[inner_prod_idx] = f2_basis_idx

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
# This would be covered by the computation below if we could do type parameter arithmetic.
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
    # A_row_idx = Vector{Int}(undef, n_basis_total)
    # A_column_idx = Vector{Int}(undef, n_basis_total)
    # A_values = Vector{Float64}(undef, n_basis_total)

    A_values = zeros(n_basis_1, n_basis_2, 2, 2)
    A_row_idx = zeros(n_basis_1, n_basis_2, 2, 2)
    A_column_idx = zeros(n_basis_1, n_basis_2, 2, 2)

    # Get geometry information, immediately checking if f1 and f2 have the same geometry.
    g, det_g = Geometry.metric(get_geometry(f1, f2), element_idx, xi)
    for f1_basis_idx = 1:n_basis_1
        for f2_basis_idx = 1:n_basis_2
            # α¹ = α¹₁dξ¹ + α¹₂dξ²
            # β¹ = β¹₁dξ¹ + β¹₂dξ²
            # <α¹, β²> = <α¹₁dξ¹, β¹₁dξ¹> + <α¹₁dξ¹, β¹₂dξ²> + <α¹₂dξ¹, β¹₁dξ²> + <α¹₂dξ¹, β¹₂dξ²>

            # <α¹₁dξ¹, β¹₁dξ¹>
            A_row_idx[f1_basis_idx, f2_basis_idx, 1, 1] = f1_indices[1][f1_basis_idx]
            A_column_idx[f1_basis_idx, f2_basis_idx, 1, 1] = f2_indices[1][f2_basis_idx]
            for quad_node_idx in eachindex(weights)
                A_values[f1_basis_idx, f2_basis_idx, 1, 1] += (1.0/det_g[quad_node_idx]) * weights[quad_node_idx] * f1_eval[1][quad_node_idx, f1_basis_idx] * g[quad_node_idx, 1, 1] * f2_eval[1][quad_node_idx, f2_basis_idx]
            end
            
            # <α¹₁dξ¹, β¹₂dξ²>
            A_row_idx[f1_basis_idx, f2_basis_idx, 1, 2] = f1_indices[1][f1_basis_idx]
            A_column_idx[f1_basis_idx, f2_basis_idx, 1, 2] = f2_indices[2][f2_basis_idx]
            for quad_node_idx in eachindex(weights)
                A_values[f1_basis_idx, f2_basis_idx, 1, 2] += (1.0/det_g[quad_node_idx]) * weights[quad_node_idx] * f1_eval[1][quad_node_idx, f1_basis_idx] * g[quad_node_idx, 1, 2] * f2_eval[2][quad_node_idx, f2_basis_idx]
            end

            # <α¹₂dξ¹, β¹₁dξ²>
            A_row_idx[f1_basis_idx, f2_basis_idx, 2, 1] = f1_indices[1][f1_basis_idx]
            A_column_idx[f1_basis_idx, f2_basis_idx, 2, 1] = f2_indices[2][f2_basis_idx]
            for quad_node_idx in eachindex(weights)
                A_values[f1_basis_idx, f2_basis_idx, 2, 1] += (1.0/det_g[quad_node_idx]) * weights[quad_node_idx] * f1_eval[2][quad_node_idx, f1_basis_idx] * g[quad_node_idx, 2, 1] * f2_eval[1][quad_node_idx, f2_basis_idx]
            end

            # <α¹₂dξ¹, β¹₂dξ²>
            A_row_idx[f1_basis_idx, f2_basis_idx, 2, 2] = f1_indices[2][f1_basis_idx]
            A_column_idx[f1_basis_idx, f2_basis_idx, 2, 2] = f2_indices[2][f2_basis_idx]
            for quad_node_idx in eachindex(weights)
                A_values[f1_basis_idx, f2_basis_idx, 2, 2] += (1.0/det_g[quad_node_idx]) * weights[quad_node_idx] * f1_eval[2][quad_node_idx, f1_basis_idx] * g[quad_node_idx, 2, 2] * f2_eval[2][quad_node_idx, f2_basis_idx]
            end
        end
    end
    
    return (A_row_idx, A_column_idx, A_values)
end

# 1-forms in 3D
# This is essentially an inner product of (n-2)-forms in n dimensions. 
# Unfortunately, we cannot (easily) write domain_dim-2 as type parameter. 
# Something along the lines of `ComputedFieldTypes.jl` might be 
# interesting, though it is focused on structs. We might also find a 
# work around.
function inner_product(f1::AbstractFormExpression{3, 1, G}, f2::AbstractFormExpression{3, 1, G}, element_idx::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G <: Geometry.AbstractGeometry{3}}
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
    # A_row_idx = Vector{Int}(undef, n_basis_total)
    # A_column_idx = Vector{Int}(undef, n_basis_total)
    # A_values = Vector{Float64}(undef, n_basis_total)

    A_values = zeros(n_basis_1, n_basis_2, 3, 3) # manifold_dim x manifold_dim
    A_row_idx = zeros(n_basis_1, n_basis_2, 3, 3)
    A_column_idx = zeros(n_basis_1, n_basis_2, 3, 3)

    # Get geometry information, immediately checking if f1 and f2 have the same geometry.
    g_inv, _, det_g = Geometry.inv_metric(get_geometry(f1, f2), element_idx, xi)
    for f1_basis_idx = 1:n_basis_1
        for f2_basis_idx = 1:n_basis_2
            
            for component_1 in 1:1:3 # manifold_dim
                for component_2 in 1:1:3 # manifold_dim

                    A_row_idx[f1_basis_idx, f2_basis_idx, component_1, component_2] = f1_indices[component_1][f1_basis_idx]
                    A_column_idx[f1_basis_idx, f2_basis_idx, component_1, component_2] = f2_indices[component_2][f2_basis_idx]
                    for quad_node_idx in eachindex(weights)
                        A_values[f1_basis_idx, f2_basis_idx, component_1, component_2] += det_g[quad_node_idx] * weights[quad_node_idx] * f1_eval[component_1][quad_node_idx, f1_basis_idx] * g_inv[quad_node_idx, component_1, component_2] * f2_eval[component_2][quad_node_idx, f2_basis_idx]
                    end
                end
            end
            
        end
    end
    
    return (A_row_idx, A_column_idx, A_values)
end

# 2-forms in 3D
# This is essentially an inner product of (n-1)-forms in n dimensions. 
# Unfortunately, we cannot (easily) write domain_dim-1 as type parameter. 
# Something along the lines of `ComputedFieldTypes.jl` might be 
# interesting, though it is focused on structs. We might also find a 
# work around.
function inner_product(f1::AbstractFormExpression{3, 2, G}, f2::AbstractFormExpression{3, 2, G}, element_idx::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G <: Geometry.AbstractGeometry{3}}
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
    # A_row_idx = Vector{Int}(undef, n_basis_total)
    # A_column_idx = Vector{Int}(undef, n_basis_total)
    # A_values = Vector{Float64}(undef, n_basis_total)

    A_values = zeros(n_basis_1, n_basis_2, 3, 3) # manifold_dim x manifold_dim
    A_row_idx = zeros(n_basis_1, n_basis_2, 3, 3)
    A_column_idx = zeros(n_basis_1, n_basis_2, 3, 3)

    # Get geometry information, immediately checking if f1 and f2 have the same geometry.
    g, det_g = Geometry.metric(get_geometry(f1, f2), element_idx, xi)
    for f1_basis_idx = 1:n_basis_1
        for f2_basis_idx = 1:n_basis_2
            
            for component_1 in 1:1:3 # manifold_dim
                for component_2 in 1:1:3 # manifold_dim

                    A_row_idx[f1_basis_idx, f2_basis_idx, component_1, component_2] = f1_indices[component_1][f1_basis_idx]
                    A_column_idx[f1_basis_idx, f2_basis_idx, component_1, component_2] = f2_indices[component_2][f2_basis_idx]
                    for quad_node_idx in eachindex(weights)
                        A_values[f1_basis_idx, f2_basis_idx, component_1, component_2] += (1.0/det_g[quad_node_idx]) * weights[quad_node_idx] * f1_eval[component_1][quad_node_idx, f1_basis_idx] * g[quad_node_idx, component_1, component_2] * f2_eval[component_2][quad_node_idx, f2_basis_idx]
                    end
                end
            end
            
        end
    end
    
    return (A_row_idx, A_column_idx, A_values)
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

            A_row_idx[inner_prod_idx] = f1_basis_idx
            A_column_idx[inner_prod_idx] = f2_basis_idx

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

# # (0-forms, 0-forms)
# function inner_product(f1::FormSpace{domain_dim, 0, G, Tuple{F1}}, f2::FormSpace{domain_dim, 0, G, Tuple{F2}}, element_id::Int, quad_rule::Quadrature.QuadratureRule{domain_dim}) where {domain_dim, G <: Geometry.AbstractGeometry{domain_dim}, F1 <: FunctionSpaces.AbstractFunctionSpace, F2 <: FunctionSpaces.AbstractFunctionSpace}

#     # Compute bases
#     basis_1_eval, basis_1_inds = FunctionSpaces.evaluate(f1.fem_space[1], element_id, quad_rule.nodes)
#     basis_2_eval, basis_2_inds = FunctionSpaces.evaluate(f2.fem_space[1], element_id, quad_rule.nodes)

#     # Compute the quantities related to the geometry.
#     # The geometries of f1 and f2 are assumed equal, so we use one. 
#     # We need to add a check for the equality of both geometries
#     _, _, sqrt_g = Geometry.inv_metric(f1.geometry, element_id, quad_rule.nodes, nderivatives)

#     # Count the number of supported basis on this element.
#     n_basis_1 = length(basis_1_inds)
#     n_basis_2 = length(basis_2_inds)
#     n_basis_total = n_basis_1 * n_basis_2

#     # Pre-allocate the inner product data: row indices, colum indices, and values).
#     M_row = Vector{Int}(undef, n_basis_total)
#     M_col = Vector{Int}(undef, n_basis_total)
#     M_val = Vector{Float64}(undef, n_basis_total)

#     # Compute the inner products
#     # Inner product M_{i,j} = (f^{1}_{i}, f^{2}_{j})
#     # I change the order of the indices with respect to the one for n-forms.
#     # The reason derives from the definition of the inner product above: 
#     # f^{1} is on the left and f^{2} on the right, therefore the index for 
#     # the basis of f^{1} is associated to the rows and the index for the basis 
#     # of f^{2} is associated to the columns.
#     for basis_2_idx in 1:n_basis_2
#         for basis_1_idx in 1:n_basis_1
#             inner_prod_idx = basis_1_idx + (basis_2_idx - 1) * n_basis_2  # get the linear index of the inner product (basis_1_idx, basis_2_idx)

#             M_row[inner_prod_idx] = basis_1_inds[basis_1_idx]
#             M_col[inner_prod_idx] = basis_2_inds[basis_2_idx]

#             M_val[inner_prod_idx] = 0.0
#             for quad_node_idx in eachindex(quad_rule.weights)
#                 M_val[inner_prod_idx] += (quad_rule.weights[quad_node_idx]/sqrt_g[quad_node_idx]) * basis_1_eval[quad_node_idx, basis_1_idx] * basis_2_eval[quad_node_idx, basis_2_idx]
#             end
#         end
#     end

#     return M_row, M_col, M_val
# end

# # (n-forms, n-forms)
# function inner_product(f1::FormSpace{domain_dim, domain_dim, G, Tuple{F1}}, f2::FormSpace{domain_dim, domain_dim, G, Tuple{F2}}, element_id::Int, quad_rule::Quadrature.QuadratureRule{domain_dim}) where {domain_dim, G <: Geometry.AbstractGeometry{domain_dim}, F1 <: FunctionSpaces.AbstractFunctionSpace, F2 <: FunctionSpaces.AbstractFunctionSpace}

#     # Compute bases and their derivatives.
#     basis_1_eval, basis_1_inds = FunctionSpaces.evaluate(f1.fem_space[1], element_id, quad_rule.nodes)
#     basis_2_eval, basis_2_inds = FunctionSpaces.evaluate(f2.fem_space[1], element_id, quad_rule.nodes)

#     # Compute the quantities related to the geometry.
#     # The geometries of f1 and f2 are assumed equal, so we use one. 
#     # We need to add a check for the equality of both geometries
#     _, _, sqrt_g = Geometry.inv_metric(f1.geometry, element_id, quad_rule.nodes, nderivatives)

#     # Count the number of supported basis on this element.
#     n_basis_1 = length(basis_1_inds)
#     n_basis_2 = length(basis_2_inds)
#     n_total = n_basis_1 * n_basis_2

#     # Pre-allocate the matrices (their row, colum, and value vectors).
#     M_row = Vector{Int}(undef, n_total)
#     M_col = Vector{Int}(undef, n_total)
#     M_val = Vector{Float64}(undef, n_total)

#     for j in 1:n_basis_2
#         for i in 1:n_basis_1
#             inner_prod_idx = i + (j-1) * n_basis_1

#             M_row[inner_prod_idx] = basis_1_inds[i]
#             M_col[inner_prod_idx] = basis_2_inds[j]

#             Mij = 0.0
#             for quad_node_idx in eachindex(quad_rule.weights)
#                 Mij += (quad_rule.weights[quad_node_idx]/sqrt_g[quad_node_idx]) * basis_1_eval[quad_node_idx,i] * basis_2_eval[quad_node_idx,j]
#             end

#             M_val[inner_prod_idx] = Mij
#         end
#     end

#     return M_row, M_col, M_val
# end

=#

# (0-forms, 0-forms) 
function inner_product(form_space1::AbstractFormSpace{manifold_dim, 0, G}, form_space2::AbstractFormSpace{manifold_dim, 0, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(form_space1.geometry, element_id, quad_rule.nodes)

    form1_basis_eval, form1_basis_indices = evaluate(form_space1, element_id, quad_rule.nodes)
    form2_basis_eval, form2_basis_indices = evaluate(form_space2, element_id, quad_rule.nodes)
    
    n_basis_1 = length(form1_basis_indices[1])
    n_basis_2 = length(form2_basis_indices[1])

    # Form space 1: α⁰ = α⁰₁
    # Form space 2: β⁰ = β⁰₁
    # ⟨α¹, β¹⟩ = ∫α⁰₁β⁰₁ⱼ√det(g)dξ¹∧…∧dξⁿ
    
    prod_form_rows = [Vector{Int}(undef, n_basis_1*n_basis_2)]
    prod_form_cols = [Vector{Int}(undef, n_basis_1*n_basis_2)]
    prod_form_basis_eval = [Vector{Float64}(undef, n_basis_1*n_basis_2)]

    prod_form_rows, prod_form_cols, prod_form_basis_eval = inner_product_0_form_component!(
        prod_form_rows, prod_form_cols, prod_form_basis_eval, quad_rule, sqrt_g, 
        form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, 
        n_basis_1, n_basis_2
    ) # Evaluates α¹₀β¹₀

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end

# (n-forms, n-forms) 
function inner_product(form_space1::AbstractFormSpace{manifold_dim, manifold_dim, G}, form_space2::AbstractFormSpace{manifold_dim, manifold_dim, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(form_space1.geometry, element_id, quad_rule.nodes)

    form1_basis_eval, form1_basis_indices = evaluate(form_space1, element_id, quad_rule.nodes)
    form2_basis_eval, form2_basis_indices = evaluate(form_space2, element_id, quad_rule.nodes)
    
    n_basis_1 = length(form1_basis_indices[1])
    n_basis_2 = length(form2_basis_indices[1])

    # Form space 1: αⁿ = αⁿ₁dξ¹∧…∧dξⁿ
    # Form space 2: βⁿ = βⁿ₁dξ¹∧…∧dξⁿ
    # ⟨αⁿ, βⁿ⟩ = ∫αⁿ₁βⁿ₁ⱼ√det(g)⁻¹dξ¹∧…∧dξⁿ
    
    prod_form_rows = [Vector{Int}(undef, n_basis_1*n_basis_2)]
    prod_form_cols = [Vector{Int}(undef, n_basis_1*n_basis_2)]
    prod_form_basis_eval = [Vector{Float64}(undef, n_basis_1*n_basis_2)]

    prod_form_rows, prod_form_cols, prod_form_basis_eval = inner_product_n_form_component!(
        prod_form_rows, prod_form_cols, prod_form_basis_eval, quad_rule, sqrt_g,
        form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, 
        n_basis_1, n_basis_2
    ) # Evaluates αⁿ₁βⁿ₁ⱼ√det(g)⁻¹

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end 

# (1-forms, 1-forms) 2D
function inner_product(form_space1::AbstractFormSpace{2, 1, G}, form_space2::AbstractFormSpace{2, 1, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{2}) where {G<:Geometry.AbstractGeometry{2}}
    g, sqrt_g = Geometry.metric(form_space1.geometry, element_id, quad_rule.nodes)
    inv_g = mapslices(inv, g, dims=(2,3))

    form1_basis_eval, form1_basis_indices = evaluate(form_space1, element_id, quad_rule.nodes)
    form2_basis_eval, form2_basis_indices = evaluate(form_space2, element_id, quad_rule.nodes)
    
    n_basis_1 = map(basis_indices -> length(basis_indices), form1_basis_indices)
    n_basis_2 = map(basis_indices -> length(basis_indices), form2_basis_indices)

    # Form space 1: α¹ = α¹₁dξ¹ +  α¹₂dξ²
    # Form space 2: β¹ = β¹₁dξ¹ +  β¹₂dξ²
    # ⟨α¹, β¹⟩ = ∫α¹ᵢβ¹ⱼgⁱʲ√det(g)dξ¹∧dξ²
    
    prod_form_rows = vec([Vector{Int}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:2, j ∈ 1:2])
    prod_form_cols = vec([Vector{Int}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:2, j ∈ 1:2])
    prod_form_basis_eval = vec([Vector{Float64}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:2, j ∈ 1:2])

    for i ∈1:2
        for j ∈1:2
            prod_form_rows, prod_form_cols, prod_form_basis_eval = inner_product_1_form_component!(
                prod_form_rows, prod_form_cols, prod_form_basis_eval, quad_rule, 
                inv_g, sqrt_g, form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, 
                n_basis_1, n_basis_2, j + (i-1)*2, (i,j)
            ) # Evaluates α¹ᵢβ¹ⱼgⁱʲ
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end 

# (1-forms, 1-forms) 3D
function inner_product(form_space1::AbstractFormSpace{3, 1, G}, form_space2::AbstractFormSpace{3, 1, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3}}
    g, sqrt_g = Geometry.metric(form_space1.geometry, element_id, quad_rule.nodes)
    inv_g = mapslices(inv, g, dims=(2,3))

    form1_basis_eval, form1_basis_indices = evaluate(form_space1, element_id, quad_rule.nodes)
    form2_basis_eval, form2_basis_indices = evaluate(form_space2, element_id, quad_rule.nodes)
    
    n_basis_1 = map(basis_indices -> length(basis_indices), form1_basis_indices)
    n_basis_2 = map(basis_indices -> length(basis_indices), form2_basis_indices)

    # Form space 1: α¹ = α¹₁dξ¹ +  α¹₂dξ² + α¹₃dξ³
    # Form space 2: β¹ = β¹₁dξ¹ +  β¹₂dξ² + β¹₃dξ³
    # ⟨α¹, β¹⟩ = ∫α¹ᵢβ¹ⱼgⁱʲ√det(g)dξ¹∧dξ²∧dξ³
    
    prod_form_rows = vec([Vector{Int}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:3, j ∈ 1:3])
    prod_form_cols = vec([Vector{Int}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:3, j ∈ 1:3])
    prod_form_basis_eval = vec([Vector{Float64}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:3, j ∈ 1:3])

    for i ∈1:3
        for j ∈1:3
            prod_form_rows, prod_form_cols, prod_form_basis_eval = inner_product_1_form_component!(
                prod_form_rows, prod_form_cols, prod_form_basis_eval, quad_rule, 
                inv_g, sqrt_g, form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, 
                n_basis_1, n_basis_2, j + (i-1)*3, (i,j)
            ) # Evaluates α¹ᵢβ¹ⱼgⁱʲ
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end

# (2-forms, 2-forms) 3D
function inner_product(form_space1::AbstractFormSpace{3, 2, G}, form_space2::AbstractFormSpace{3, 2, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3}}
    g, sqrt_g = Geometry.metric(form_space1.geometry, element_id, quad_rule.nodes)
    inv_g = mapslices(inv, g, dims=(2,3))

    form1_basis_eval, form1_basis_indices = evaluate(form_space1, element_id, quad_rule.nodes)
    form2_basis_eval, form2_basis_indices = evaluate(form_space2, element_id, quad_rule.nodes)
    
    n_basis_1 = map(basis_indices -> length(basis_indices), form1_basis_indices)
    n_basis_2 = map(basis_indices -> length(basis_indices), form2_basis_indices)

    # Form space 1: α² = α₁²dξ₂∧dξ₃ + α₂²dξ₃∧dξ₁ + α₃²dξ₁∧dξ₂
    # Form space 2: β² = β₁²dξ₂∧dξ₃ + β₂²dξ₃∧dξ₁ + β₃²dξ₁∧dξ₂
    # ⟨α², β²⟩ = ∫α¹ᵢβ¹ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)dξ¹∧dξ²∧dξ³
    
    prod_form_rows = vec([Vector{Int}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:3, j ∈ 1:3])
    prod_form_cols = vec([Vector{Int}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:3, j ∈ 1:3])
    prod_form_basis_eval = vec([Vector{Float64}(undef, n_basis_1[i]*n_basis_2[j]) for i ∈ 1:3, j ∈ 1:3])

    for i ∈1:3
        for j ∈1:3
            prod_form_rows, prod_form_cols, prod_form_basis_eval = inner_product_2_form_component!(
                prod_form_rows, prod_form_cols, prod_form_basis_eval, quad_rule, 
                inv_g, sqrt_g, form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, 
                n_basis_1, n_basis_2, j + (i-1)*3, (i,j)
            ) # Evaluates α¹ᵢβ¹ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end

# Helpers for inner product

function inner_product_0_form_component!(prod_form_rows::Vector{Vector{Int}}, prod_form_cols::Vector{Vector{Int}}, prod_form_basis_eval::Vector{Vector{Float64}}, quad_rule, sqrt_g, form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, n_basis_1, n_basis_2)

    for j ∈ 1:n_basis_2
        for i ∈ 1:n_basis_1
            linear_idx = i + (j-1) * n_basis_1

            prod_form_rows[1][linear_idx] = form1_basis_indices[1][i]
            prod_form_cols[1][linear_idx] = form2_basis_indices[1][j]

            prod_form_basis_eval[1][linear_idx] = @views (quad_rule.weights .* form1_basis_eval[1][:,i])' * (form2_basis_eval[1][:,j] .* sqrt_g)    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end

function inner_product_n_form_component!(prod_form_rows::Vector{Vector{Int}}, prod_form_cols::Vector{Vector{Int}}, prod_form_basis_eval::Vector{Vector{Float64}}, quad_rule, sqrt_g, form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, n_basis_1, n_basis_2)

    for j ∈ 1:n_basis_2
        for i ∈ 1:n_basis_1
            linear_idx = i + (j-1) * n_basis_1

            prod_form_rows[1][linear_idx] = form1_basis_indices[1][i]
            prod_form_cols[1][linear_idx] = form2_basis_indices[1][j]

            prod_form_basis_eval[1][linear_idx] = @views (quad_rule.weights .* form1_basis_eval[1][:,i])' * (form2_basis_eval[1][:,j] .*(sqrt_g.^(-1)))    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end

function inner_product_1_form_component!(prod_form_rows::Vector{Vector{Int}}, prod_form_cols::Vector{Vector{Int}}, prod_form_basis_eval::Vector{Vector{Float64}}, quad_rule, inv_g, sqrt_g, form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, n_basis_1, n_basis_2, component_idx::Int, form_idxs::NTuple{2, Int})

    for j ∈ 1:n_basis_2[form_idxs[2]]
        for i ∈ 1:n_basis_1[form_idxs[1]]
            linear_idx = i + (j-1) * n_basis_1[form_idxs[1]]

            prod_form_rows[component_idx][linear_idx] = form1_basis_indices[form_idxs[1]][i]
            prod_form_cols[component_idx][linear_idx] = form2_basis_indices[form_idxs[2]][j]

            prod_form_basis_eval[component_idx][linear_idx] = @views (quad_rule.weights .* form1_basis_eval[form_idxs[1]][:,i])' * (form2_basis_eval[form_idxs[2]][:,j] .* inv_g[:,form_idxs[1],form_idxs[2]] .* sqrt_g)    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end

function inner_product_2_form_component!(prod_form_rows::Vector{Vector{Int}}, prod_form_cols::Vector{Vector{Int}}, prod_form_basis_eval::Vector{Vector{Float64}}, quad_rule, inv_g, sqrt_g, form1_basis_eval, form1_basis_indices, form2_basis_eval, form2_basis_indices, n_basis_1, n_basis_2, component_idx::Int, form_idxs::NTuple{2, Int})

    inv_indices_1 = (mod(form_idxs[1], 3) + 1, mod(form_idxs[1]+1, 3) + 1)
    inv_indices_2 = (mod(form_idxs[2], 3) + 1, mod(form_idxs[2]+1, 3) + 1)

    inv_g_factor = inv_g[:,inv_indices_1[1],inv_indices_2[1]].*inv_g[:,inv_indices_1[2],inv_indices_2[2]] .- inv_g[:,inv_indices_1[1],inv_indices_2[2]].*inv_g[:,inv_indices_1[2],inv_indices_2[1]]

    for j ∈ 1:n_basis_2[form_idxs[2]]
        for i ∈ 1:n_basis_1[form_idxs[1]]
            linear_idx = i + (j-1) * n_basis_1[form_idxs[1]]

            prod_form_rows[component_idx][linear_idx] = form1_basis_indices[form_idxs[1]][i]
            prod_form_cols[component_idx][linear_idx] = form2_basis_indices[form_idxs[2]][j]

            prod_form_basis_eval[component_idx][linear_idx] = @views (quad_rule.weights .* form1_basis_eval[form_idxs[1]][:,i])' * (form2_basis_eval[form_idxs[2]][:,j] .* inv_g_factor .* sqrt_g)    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_basis_eval
end

# Priority 2: Wedge products

# Priority 3: Triple products