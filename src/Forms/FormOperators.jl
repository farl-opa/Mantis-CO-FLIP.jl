# Priority 1: Inner products for abstract form fields

# (0-forms, 0-forms)
@doc raw"""
    evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, 0, G}, 
                           form_expression2::AbstractFormExpression{manifold_dim, 0, G}, 
                           element_id::Int, 
                           quad_rule::Quadrature.QuadratureRule{manifold_dim}) 
                           where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}

Compute the inner product of two differential 0-forms over a specified element of a manifold.

# Arguments
- `form_expression1::AbstractFormExpression{manifold_dim, 0, G}`: The first differential form expression. It represents a form on a manifold of dimension `manifold_dim`.
- `form_expression2::AbstractFormExpression{manifold_dim, 0, G}`: The second differential form expression. It should have the same dimension and geometry as `form_expression1`.
- `element_id::Int`: The identifier of the element on which the inner product is to be evaluated. This is typically an index into a mesh or other discretized representation of the manifold.
- `quad_rule::Quadrature.QuadratureRule{manifold_dim}`: The quadrature rule used to approximate the integral of the inner product over the specified element. The quadrature rule should be compatible with the dimension of the manifold.

# Returns
- `prod_form_rows::Vector{Vector{Int}}`: Indices of the first form expression in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_cols::Vector{Vector{Int}}`: Indices of the second form expresson in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_eval::Vector{Vector{Float64}}`: Evaluation of the inner product in each component m. Hence, prod_form_eval[m][l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. If the form expressions given have coefficients for the evaluate method, the return format is [x] where x is the inner product value.
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, 0, G}, form_expression2::AbstractFormExpression{manifold_dim, 0, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    # Form 1: α⁰ = α⁰₁
    # Form 2: β⁰ = β⁰₁
    # ⟨α⁰, β⁰⟩ = ∫α⁰₁β⁰₁ⱼ√det(g)dξ¹∧…∧dξⁿ
    
    prod_form_rows = [Vector{Int}(undef, n_indices_1*n_indices_2)]
    prod_form_cols = [Vector{Int}(undef, n_indices_1*n_indices_2)]
    prod_form_eval = [Vector{Float64}(undef, n_indices_1*n_indices_2)]

    prod_form_rows, prod_form_cols, prod_form_eval = inner_product_0_form_component!(
        prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, sqrt_g, 
        form1_eval, form1_indices, form2_eval, form2_indices, 
        n_indices_1, n_indices_2
    ) # Evaluates α⁰₁β⁰₁√det(g)

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# (n-forms, n-forms)
@doc raw"""
    evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, manifold_dim, G}, 
                           form_expression2::AbstractFormExpression{manifold_dim, manifold_dim, G}, 
                           element_id::Int, 
                           quad_rule::Quadrature.QuadratureRule{manifold_dim}) 
                           where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}

Compute the inner product of two differential n-forms over a specified element of a manifold.

# Arguments
- `form_expression1::AbstractFormExpression{manifold_dim, manifold_dim, G}`: The first differential form expression. It represents a form on a manifold of dimension `manifold_dim`.
- `form_expression2::AbstractFormExpression{manifold_dim, manifold_dim, G}`: The second differential form expression. It should have the same dimension and geometry as `form_expression1`.
- `element_id::Int`: The identifier of the element on which the inner product is to be evaluated. This is typically an index into a mesh or other discretized representation of the manifold.
- `quad_rule::Quadrature.QuadratureRule{manifold_dim}`: The quadrature rule used to approximate the integral of the inner product over the specified element. The quadrature rule should be compatible with the dimension of the manifold.

# Returns
- `prod_form_rows::Vector{Vector{Int}}`: Indices of the first form expression in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_cols::Vector{Vector{Int}}`: Indices of the second form expresson in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_eval::Vector{Vector{Float64}}`: Evaluation of the inner product in each component m. Hence, prod_form_eval[m][l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. If the form expressions given have coefficients for the evaluate method, the return format is [x] where x is the inner product value.
""" 
function evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, manifold_dim, G}, form_expression2::AbstractFormExpression{manifold_dim, manifold_dim, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    # Form 1: αⁿ = αⁿ₁dξ¹∧…∧dξⁿ
    # Form 2: βⁿ = βⁿ₁dξ¹∧…∧dξⁿ
    # ⟨αⁿ, βⁿ⟩ = ∫αⁿ₁βⁿ₁ⱼ√det(g)⁻¹dξ¹∧…∧dξⁿ
    
    prod_form_rows = [Vector{Int}(undef, n_indices_1*n_indices_2)]
    prod_form_cols = [Vector{Int}(undef, n_indices_1*n_indices_2)]
    prod_form_eval = [Vector{Float64}(undef, n_indices_1*n_indices_2)]

    prod_form_rows, prod_form_cols, prod_form_eval = inner_product_n_form_component!(
        prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, sqrt_g,
        form1_eval, form1_indices, form2_eval, form2_indices, 
        n_indices_1, n_indices_2
    ) # Evaluates αⁿ₁βⁿ₁ⱼ√det(g)⁻¹

    return prod_form_rows, prod_form_cols, prod_form_eval
end 

# (1-forms, 1-forms) 2D
@doc raw"""
    evaluate_inner_product(form_expression1::AbstractFormExpression{2, 1, G}, 
                           form_expression2::AbstractFormExpression{2, 1, G}, 
                           element_id::Int, 
                           quad_rule::Quadrature.QuadratureRule{2}) 
                           where {G<:Geometry.AbstractGeometry{2}}

Compute the inner product of two differential 1-forms over a specified element of a 2-dimensional manifold.

# Arguments
- `form_expression1::AbstractFormExpression{2, 1, G}`: The first differential form expression. It represents a form on a manifold of dimension 2.
- `form_expression2::AbstractFormExpression{2, 1, G}`: The second differential form expression. It should have the same dimension and geometry as `form_expression1`.
- `element_id::Int`: The identifier of the element on which the inner product is to be evaluated. This is typically an index into a mesh or other discretized representation of the manifold.
- `quad_rule::Quadrature.QuadratureRule{2}`: The quadrature rule used to approximate the integral of the inner product over the specified element. The quadrature rule should be compatible with the dimension of the manifold.

# Returns
- `prod_form_rows::Vector{Vector{Int}}`: Indices of the first form expression in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_cols::Vector{Vector{Int}}`: Indices of the second form expresson in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_eval::Vector{Vector{Float64}}`: Evaluation of the inner product in each component m. Hence, prod_form_eval[m][l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. If the form expressions given have coefficients for the evaluate method, the return format is [x] where x is the inner product value.
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{2, 1, G}, form_expression2::AbstractFormExpression{2, 1, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{2}) where {G<:Geometry.AbstractGeometry{2}}
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    n_indices_1 = map(indices -> length(indices), form1_indices)
    n_indices_2 = map(indices -> length(indices), form2_indices)

    # Form 1: α¹ = α¹₁dξ¹ +  α¹₂dξ²
    # Form 2: β¹ = β¹₁dξ¹ +  β¹₂dξ²
    # ⟨α¹, β¹⟩ = ∫α¹ᵢβ¹ⱼgⁱʲ√det(g)dξ¹∧dξ²
    
    prod_form_rows = vec([Vector{Int}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:2, j ∈ 1:2])
    prod_form_cols = vec([Vector{Int}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:2, j ∈ 1:2])
    prod_form_eval = vec([Vector{Float64}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:2, j ∈ 1:2])

    for i ∈1:2
        for j ∈1:2
            prod_form_rows, prod_form_cols, prod_form_eval = inner_product_1_form_component!(
                prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, 
                inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, 
                n_indices_1, n_indices_2, j + (i-1)*2, (i,j)
            ) # Evaluates α¹ᵢβ¹ⱼgⁱʲ√det(g)
        end
    end

    if all(n -> n==1, n_indices_1) && all(n -> n==1, n_indices_2) # This output makes more sense in the case of FormFields.
        return [[1]], [[1]], [sum(prod_form_eval)]
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end 

# (1-forms, 1-forms) 3D
@doc raw"""
    evaluate_inner_product(form_expression1::AbstractFormExpression{3, 1, G}, 
                           form_expression2::AbstractFormExpression{3, 1, G}, 
                           element_id::Int, 
                           quad_rule::Quadrature.QuadratureRule{3}) 
                           where {G<:Geometry.AbstractGeometry{3}}

Compute the inner product of two differential 1-forms over a specified element of a 3-dimensional manifold.

# Arguments
- `form_expression1::AbstractFormExpression{3, 1, G}`: The first differential form expression. It represents a form on a manifold of dimension 3.
- `form_expression2::AbstractFormExpression{3, 1, G}`: The second differential form expression. It should have the same dimension and geometry as `form_expression1`.
- `element_id::Int`: The identifier of the element on which the inner product is to be evaluated. This is typically an index into a mesh or other discretized representation of the manifold.
- `quad_rule::Quadrature.QuadratureRule{3}`: The quadrature rule used to approximate the integral of the inner product over the specified element. The quadrature rule should be compatible with the dimension of the manifold.

# Returns
- `prod_form_rows::Vector{Vector{Int}}`: Indices of the first form expression in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_cols::Vector{Vector{Int}}`: Indices of the second form expresson in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_eval::Vector{Vector{Float64}}`: Evaluation of the inner product in each component m. Hence, prod_form_eval[m][l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. If the form expressions given have coefficients for the evaluate method, the return format is [x] where x is the inner product value.
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{3, 1, G}, form_expression2::AbstractFormExpression{3, 1, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3}}
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    n_indices_1 = map(basis_indices -> length(basis_indices), form1_indices)
    n_indices_2 = map(basis_indices -> length(basis_indices), form2_indices)

    # Form 1: α¹ = α¹₁dξ¹ +  α¹₂dξ² + α¹₃dξ³
    # Form 2: β¹ = β¹₁dξ¹ +  β¹₂dξ² + β¹₃dξ³
    # ⟨α¹, β¹⟩ = ∫α¹ᵢβ¹ⱼgⁱʲ√det(g)dξ¹∧dξ²∧dξ³
    
    prod_form_rows = vec([Vector{Int}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:3, j ∈ 1:3])
    prod_form_cols = vec([Vector{Int}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:3, j ∈ 1:3])
    prod_form_eval = vec([Vector{Float64}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:3, j ∈ 1:3])

    for i ∈1:3
        for j ∈1:3
            prod_form_rows, prod_form_cols, prod_form_eval = inner_product_1_form_component!(
                prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, 
                inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, 
                n_indices_1, n_indices_2, j + (i-1)*3, (i,j)
            ) # Evaluates α¹ᵢβ¹ⱼgⁱʲ
        end
    end

    if all(n -> n==1, n_indices_1) && all(n -> n==1, n_indices_2) # This output makes more sense in the case of FormFields.
        return [[1]], [[1]], [sum(prod_form_eval)]
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# (2-forms, 2-forms) 3D
@doc raw"""
    evaluate_inner_product(form_expression1::AbstractFormExpression{3, 2, G}, 
                           form_expression2::AbstractFormExpression{3, 2, G}, 
                           element_id::Int, 
                           quad_rule::Quadrature.QuadratureRule{3}) 
                           where {G<:Geometry.AbstractGeometry{3}}

Compute the inner product of two differential 2-forms over a specified element of a 3-dimensional manifold.

# Arguments
- `form_expression1::AbstractFormExpression{3, 2, G}`: The first differential form expression. It represents a form on a manifold of dimension 3.
- `form_expression2::AbstractFormExpression{3, 2, G}`: The second differential form expression. It should have the same dimension and geometry as `form_expression1`.
- `element_id::Int`: The identifier of the element on which the inner product is to be evaluated. This is typically an index into a mesh or other discretized representation of the manifold.
- `quad_rule::Quadrature.QuadratureRule{3}`: The quadrature rule used to approximate the integral of the inner product over the specified element. The quadrature rule should be compatible with the dimension of the manifold.

# Returns
- `prod_form_rows::Vector{Vector{Int}}`: Indices of the first form expression in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_cols::Vector{Vector{Int}}`: Indices of the second form expresson in each component the inner product. If the form expressions given have coefficients for the evaluate method, the return format is [1] to indicate a single "basis" function.
- `prod_form_eval::Vector{Vector{Float64}}`: Evaluation of the inner product in each component m. Hence, prod_form_eval[m][l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. If the form expressions given have coefficients for the evaluate method, the return format is [x] where x is the inner product value.
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{3, 2, G}, form_expression2::AbstractFormExpression{3, 2, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3}}
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    n_indices_1 = map(basis_indices -> length(basis_indices), form1_indices)
    n_indices_2 = map(basis_indices -> length(basis_indices), form2_indices)

    # Form space 1: α² = α²₁dξ₂∧dξ₃ + α²₂dξ₃∧dξ₁ + α²₃dξ₁∧dξ₂
    # Form space 2: β² = β²₁dξ₂∧dξ₃ + β²₂dξ₃∧dξ₁ + β²₃dξ₁∧dξ₂
    # ⟨α², β²⟩ = ∫α²ᵢβ²ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)dξ¹∧dξ²∧dξ³
    
    prod_form_rows = vec([Vector{Int}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:3, j ∈ 1:3])
    prod_form_cols = vec([Vector{Int}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:3, j ∈ 1:3])
    prod_form_eval = vec([Vector{Float64}(undef, n_indices_1[i]*n_indices_2[j]) for i ∈ 1:3, j ∈ 1:3])

    for i ∈1:3
        for j ∈1:3
            prod_form_rows, prod_form_cols, prod_form_eval = inner_product_2_form_component!(
                prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, 
                inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, 
                n_indices_1, n_indices_2, j + (i-1)*3, (i,j)
            ) # Evaluates α¹ᵢβ¹ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)
        end
    end

    if all(n -> n==1, n_indices_1) && all(n -> n==1, n_indices_2) # This output makes more sense in the case of FormFields.
        return [[1]], [[1]], [sum(prod_form_eval)]
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# Helpers for inner product

function inner_product_0_form_component!(prod_form_rows::Vector{Vector{Int}}, prod_form_cols::Vector{Vector{Int}}, prod_form_eval::Vector{Vector{Float64}}, quad_rule, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2)

    for j ∈ 1:n_indices_2
        for i ∈ 1:n_indices_1
            linear_idx = i + (j-1) * n_indices_1

            prod_form_rows[1][linear_idx] = form1_indices[1][i]
            prod_form_cols[1][linear_idx] = form2_indices[1][j]

            prod_form_eval[1][linear_idx] = @views (quad_rule.weights .* form1_eval[1][:,i])' * (form2_eval[1][:,j] .* sqrt_g)    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function inner_product_n_form_component!(prod_form_rows::Vector{Vector{Int}}, prod_form_cols::Vector{Vector{Int}}, prod_form_eval::Vector{Vector{Float64}}, quad_rule, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2)

    for j ∈ 1:n_indices_2
        for i ∈ 1:n_indices_1
            linear_idx = i + (j-1) * n_indices_1

            prod_form_rows[1][linear_idx] = form1_indices[1][i]
            prod_form_cols[1][linear_idx] = form2_indices[1][j]

            prod_form_eval[1][linear_idx] = @views (quad_rule.weights .* form1_eval[1][:,i])' * (form2_eval[1][:,j] .*(sqrt_g.^(-1)))    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function inner_product_1_form_component!(prod_form_rows::Vector{Vector{Int}}, prod_form_cols::Vector{Vector{Int}}, prod_form_eval::Vector{Vector{Float64}}, quad_rule, inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2, component_idx::Int, form_idxs::NTuple{2, Int})

    for j ∈ 1:n_indices_2[form_idxs[2]]
        for i ∈ 1:n_indices_1[form_idxs[1]]
            linear_idx = i + (j-1) * n_indices_1[form_idxs[1]]

            prod_form_rows[component_idx][linear_idx] = form1_indices[form_idxs[1]][i]
            prod_form_cols[component_idx][linear_idx] = form2_indices[form_idxs[2]][j]

            prod_form_eval[component_idx][linear_idx] = @views (quad_rule.weights .* form1_eval[form_idxs[1]][:,i])' * (form2_eval[form_idxs[2]][:,j] .* inv_g[:,form_idxs[1],form_idxs[2]] .* sqrt_g)    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function inner_product_2_form_component!(prod_form_rows::Vector{Vector{Int}}, prod_form_cols::Vector{Vector{Int}}, prod_form_eval::Vector{Vector{Float64}}, quad_rule, inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2, component_idx::Int, form_idxs::NTuple{2, Int})

    inv_indices_1 = (mod(form_idxs[1], 3) + 1, mod(form_idxs[1]+1, 3) + 1)
    inv_indices_2 = (mod(form_idxs[2], 3) + 1, mod(form_idxs[2]+1, 3) + 1)

    inv_g_factor = inv_g[:,inv_indices_1[1],inv_indices_2[1]].*inv_g[:,inv_indices_1[2],inv_indices_2[2]] .- inv_g[:,inv_indices_1[1],inv_indices_2[2]].*inv_g[:,inv_indices_1[2],inv_indices_2[1]]

    for j ∈ 1:n_indices_2[form_idxs[2]]
        for i ∈ 1:n_indices_1[form_idxs[1]]
            linear_idx = i + (j-1) * n_indices_1[form_idxs[1]]

            prod_form_rows[component_idx][linear_idx] = form1_indices[form_idxs[1]][i]
            prod_form_cols[component_idx][linear_idx] = form2_indices[form_idxs[2]][j]

            prod_form_eval[component_idx][linear_idx] = @views (quad_rule.weights .* form1_eval[form_idxs[1]][:,i])' * (form2_eval[form_idxs[2]][:,j] .* inv_g_factor .* sqrt_g)    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# Priority 2: Wedge products

# Priority 3: Triple products

# Hodge star Operator (⋆) evaluation methods

# ωⁿ = √det(gᵢⱼ) dx¹∧…∧dxⁿ denotes the volume form of a Riemannian manifold.
# αⱼᵏ denotes the j-th component of a k-form αᵏ. 

# 0-forms (n dimensions)
@doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{manifold_dim, 0, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}})

Evaluates the Hodge star operator ⋆ of a 0-form in n dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{manifold_dim, 0, G}`: The differential 0-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate,
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `form_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{manifold_dim, 0, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    n_indices = length(form_indices[1])

    hodge_eval = Vector{Matrix{Float64}}(undef, 1)

    # ⋆α₁⁰ = α₁⁰√det(gᵢⱼ) dξ₁∧…∧dξₙ. 
    hodge_eval[1] = reshape(form_eval[1] .* sqrt_g, (:, n_indices))
 
    return hodge_eval, form_indices
end

# n-forms (n dimensions)
 @doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{manifold_dim, manifold_dim, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}

Evaluates the Hodge star operator ⋆ of a n-form in n dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{manifold_dim, manifold_dim, G}`: The differential n-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `form_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{manifold_dim, manifold_dim, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    n_indices = length(form_indices[1])

    hodge_eval = Vector{Matrix{Float64}}(undef, 1)

    # ⋆α₁ⁿdξ₁∧…∧dξₙ = α₁ⁿ(√det(gᵢⱼ))⁻¹. 
    hodge_eval[1] = reshape(form_eval[1] .* (sqrt_g.^(-1)), (:, n_indices))
 
    return hodge_eval, form_indices
end

# 1-forms (2 dimensions)
@doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{2, 1, G}, element_id::Int, xi::NTuple{2, Vector{Float64}}) where {G <: Geometry.AbstractGeometry{2}}

Evaluates the Hodge star operator ⋆ of a 1-form in 2 dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{2, 1, G}`: The differential 1-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{2, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval::Vector{Matrix{Float64}}`: Vector of arrays containing evaluated hodge star of the basis functions.
- `hodge_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{2, 1, G}, element_id::Int, xi::NTuple{2, Vector{Float64}}) where {G <: Geometry.AbstractGeometry{2}}
    inv_g, _, sqrt_g = Geometry.inv_metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    
    hodge_eval = Vector{Matrix{Float64}}(undef, 2)

    # ⋆(α₁¹dξ₁+α₂¹dξ₂) = [-(α₁¹g²¹+α₂¹g²²)dξ₁ + (α₁¹g¹¹+α₂¹g¹²)dξ₂]√det(gᵢⱼ). 

    # First: -(α₁¹g²¹+α₂¹g²²)dξ₁
    hodge_eval[1] = @views hcat([-form_eval[i] .* inv_g[:, 2, i] for i in 1:2]...) .* sqrt_g

    # Second: (α₁¹g¹¹+α₂¹g¹²)dξ₂
    hodge_eval[2] = @views hcat([form_eval[i] .* inv_g[:, 1, i] for i in 1:2]...) .* sqrt_g

    if all(indices -> length(indices)==1, form_indices) # This output makes more sense in the case of FormFields.
        hodge_eval[1] = sum(hodge_eval[1], dims=2)
        hodge_eval[2] = sum(hodge_eval[2], dims=2)

        return  hodge_eval, [[1], [1]]
    end

    hodge_indices = repeat([vcat(form_indices...)], 2)

    return hodge_eval, hodge_indices
end

# 1-forms (3 dimensions)
@doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{3, 1, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {G <: Geometry.AbstractGeometry{3}}

Evaluates the Hodge star operator ⋆ of a 1-form in 3 dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{3, 1, G}`: The differential 1-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `hodge_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{3, 1, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {G <: Geometry.AbstractGeometry{3}}
    inv_g, _, sqrt_g = Geometry.inv_metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)

    hodge_eval = Vector{Matrix{Float64}}(undef, 3)

    # ⋆(α₁¹dξ₁+α₂¹dξ₂+α₃¹dξ₃) = [(α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃ + (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁ + (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂]√det(gᵢⱼ). 
    # First: (α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃
    hodge_eval[1] = @views hcat([form_eval[i] .* inv_g[:, 1, i] for i in 1:3]...).*sqrt_g

    # Second: (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁
    hodge_eval[2] = @views hcat([form_eval[i] .* inv_g[:, 2, i] for i in 1:3]...).*sqrt_g

    # Third: (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂
    hodge_eval[3] = @views hcat([form_eval[i] .* inv_g[:, 3, i] for i in 1:3]...).*sqrt_g

    if all(indices -> length(indices)==1, form_indices) # This output makes more sense in the case of FormFields.
        hodge_eval[1] = sum(hodge_eval[1], dims=2)
        hodge_eval[2] = sum(hodge_eval[2], dims=2)
        hodge_eval[3] = sum(hodge_eval[3], dims=2)

        return  hodge_eval, [[1], [1]]
    end

    hodge_indices = repeat([vcat(form_indices...)], 3)

    return hodge_eval, hodge_indices
end

# 2-forms (3 dimensions)
@doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{3, 2, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {G <: Geometry.AbstractGeometry{3}}

Evaluates the Hodge star operator ⋆ of a 2-form in 3 dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{3, 2, G}`: The differential 2-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `hodge_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{3, 2, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {G <: Geometry.AbstractGeometry{3}}
    inv_g, _, sqrt_g = Geometry.inv_metric(form_expression.geometry, element_id, xi)
    
    form_eval, form_indices = evaluate(form_expression, element_id, xi)

    hodge_eval = Vector{Matrix{Float64}}(undef, 3)

    # ⋆(α₁²dξ₂∧dξ₃ + α₂²dξ₃∧dξ₁ + α₃²dξ₁∧dξ₂) = [(α₁²(g²²g³³-g²³g³²) + α₂²(g²³g³¹-g²¹g³³) + α₃²(g²¹g³²-g²²g³¹))dξ¹ +
    #                                            (α₁²(g³²g¹³-g³³g¹²) + α₂²(g³³g¹¹-g³¹g¹³) + α₃²(g³¹g¹²-g³²g¹¹))dξ² +
    #                                            (α₁²(g¹²g²³-g¹³g²²) + α₂²(g¹³g²¹-g¹¹g²³) + α₃²(g¹¹g²²-g¹²g²¹))dξ³ ]√det(gᵢⱼ). 
    # First: (α₁²(g²²g³³-g²³g³²) + α₂²(g²³g³¹-g²¹g³³) + α₃²(g²¹g³²-g²²g³¹))dξ¹
    hodge_eval[1] = @views hcat(form_eval[1].*(inv_g[:,2,2].*inv_g[:,3,3]-inv_g[:,2,3].*inv_g[:,3,2]), form_eval[2].*(inv_g[:,2,3].*inv_g[:,3,1]-inv_g[:,2,1].*inv_g[:,3,3]), form_eval[3].*(inv_g[:,2,1].*inv_g[:,3,2]-inv_g[:,2,2].*inv_g[:,3,1])).*sqrt_g

    # Second: (α₁²(g³²g¹³-g³³g¹²) + α₂²(g³³g¹¹-g³¹g¹³) + α₃²(g³¹g¹²-g³²g¹¹))dξ²
    hodge_eval[2] = @views hcat(form_eval[1].*(inv_g[:,3,2].*inv_g[:,1,3]-inv_g[:,3,3].*inv_g[:,1,2]), form_eval[2].*(inv_g[:,3,3].*inv_g[:,1,1]-inv_g[:,3,1].*inv_g[:,1,3]), form_eval[3].*(inv_g[:,3,1].*inv_g[:,1,2]-inv_g[:,3,2].*inv_g[:,1,1])).*sqrt_g

    # Third: (α₁²(g¹²g²³-g¹³g²²) + α₂²(g¹³g²¹-g¹¹g²³) + α₃²(g¹¹g²²-g¹²g²¹))dξ³
    hodge_eval[3] = @views hcat(form_eval[1].*(inv_g[:,1,2].*inv_g[:,2,3]-inv_g[:,1,3].*inv_g[:,2,2]), form_eval[2].*(inv_g[:,1,3].*inv_g[:,2,1]-inv_g[:,1,1].*inv_g[:,2,3]), form_eval[3].*(inv_g[:,1,1].*inv_g[:,2,2]-inv_g[:,1,2].*inv_g[:,2,1])).*sqrt_g

    if all(indices -> length(indices)==1, form_indices) # This output makes more sense in the case of FormFields.
        hodge_eval[1] = sum(hodge_eval[1], dims=2)
        hodge_eval[2] = sum(hodge_eval[2], dims=2)
        hodge_eval[3] = sum(hodge_eval[3], dims=2)

        return  hodge_eval, [[1], [1]]
    end

    hodge_indices = repeat([vcat(form_indices...)], 3)

    return hodge_eval, hodge_indices
end
