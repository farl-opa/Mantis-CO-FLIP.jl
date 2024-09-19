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
- `prod_form_rows::Vector{Int}`: Indices of the first form expression. 
- `prod_form_cols::Vector{Int}`: Indices of the second form expression. 
- `prod_form_eval::Vector{Float64}`: Evaluation of the inner product. Hence, prod_form_eval[l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. 
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, 0, G}, form_expression2::AbstractFormExpression{manifold_dim, 0, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    # evaluate both the form expressions; it is assumed that the output is of the following form:
    # form_eval::Vector{Matrix{Float64}} of length 1, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # get dimension of each form expression
    n_indices_1 = length(form1_indices)
    n_indices_2 = length(form2_indices)

    # Form 1: α⁰ = α⁰₁
    # Form 2: β⁰ = β⁰₁
    # ⟨α⁰, β⁰⟩ = ∫α⁰₁β⁰₁ⱼ√det(g)dξ¹∧…∧dξⁿ
    
    prod_form_rows = Vector{Int}(undef, n_indices_1*n_indices_2)
    prod_form_cols = Vector{Int}(undef, n_indices_1*n_indices_2)
    prod_form_eval = Vector{Float64}(undef, n_indices_1*n_indices_2)

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
- `prod_form_rows::Vector{Int}`: Indices of the first form expression. 
- `prod_form_cols::Vector{Int}`: Indices of the second form expression. 
- `prod_form_eval::Vector{Float64}`: Evaluation of the inner product. Hence, prod_form_eval[l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. 
""" 
function evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, manifold_dim, G}, form_expression2::AbstractFormExpression{manifold_dim, manifold_dim, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    # form_eval::Vector{Matrix{Float64}} of length 1, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # get dimension of each form expression
    n_indices_1 = length(form1_indices)
    n_indices_2 = length(form2_indices)

    # Form 1: αⁿ = αⁿ₁dξ¹∧…∧dξⁿ
    # Form 2: βⁿ = βⁿ₁dξ¹∧…∧dξⁿ
    # ⟨αⁿ, βⁿ⟩ = ∫αⁿ₁βⁿ₁ⱼ√det(g)⁻¹dξ¹∧…∧dξⁿ
    
    prod_form_rows = Vector{Int}(undef, n_indices_1*n_indices_2)
    prod_form_cols = Vector{Int}(undef, n_indices_1*n_indices_2)
    prod_form_eval = Vector{Float64}(undef, n_indices_1*n_indices_2)

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
- `prod_form_rows::Vector{Int}`: Indices of the first form expression. 
- `prod_form_cols::Vector{Int}`: Indices of the second form expression. 
- `prod_form_eval::Vector{Float64}`: Evaluation of the inner product. Hence, prod_form_eval[l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. 
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{2, 1, G}, form_expression2::AbstractFormExpression{2, 1, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{2}) where {G<:Geometry.AbstractGeometry{2}}
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    # form_eval::Vector{Matrix{Float64}} of length 2, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # get dimension of each form expression
    n_indices_1 = length(form1_indices)
    n_indices_2 = length(form2_indices)

    n_prod_indices = n_indices_1*n_indices_2
    
    # Form 1: α¹ = α¹₁dξ¹ +  α¹₂dξ²
    # Form 2: β¹ = β¹₁dξ¹ +  β¹₂dξ²
    # ⟨α¹, β¹⟩ = ∫α¹ᵢβ¹ⱼgⁱʲ√det(g)dξ¹∧dξ²
    
    prod_form_rows = Vector{Int}(undef, n_prod_indices)
    prod_form_cols = Vector{Int}(undef, n_prod_indices)
    prod_form_eval = Vector{Float64}(undef, n_prod_indices)

    for i ∈ 1:2
        for j ∈ 1:2
            prod_form_rows, prod_form_cols, prod_form_eval = inner_product_1_form_component!(
                prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, 
                inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, 
                n_indices_1, n_indices_2, (i,j)) # Evaluates α¹ᵢβ¹ⱼgⁱʲ√det(g)
        end
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
- `prod_form_rows::Vector{Int}`: Indices of the first form expression. 
- `prod_form_cols::Vector{Int}`: Indices of the second form expression. 
- `prod_form_eval::Vector{Float64}`: Evaluation of the inner product. Hence, prod_form_eval[l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. 
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{3, 1, G}, form_expression2::AbstractFormExpression{3, 1, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3}}
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    # form_eval::Vector{Matrix{Float64}} of length 3, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # get dimension of each form expression
    n_indices_1 = length(form1_indices)
    n_indices_2 = length(form2_indices)

    n_prod_indices = n_indices_1*n_indices_2

    # Form 1: α¹ = α¹₁dξ¹ +  α¹₂dξ² + α¹₃dξ³
    # Form 2: β¹ = β¹₁dξ¹ +  β¹₂dξ² + β¹₃dξ³
    # ⟨α¹, β¹⟩ = ∫α¹ᵢβ¹ⱼgⁱʲ√det(g)dξ¹∧dξ²∧dξ³
    
    prod_form_rows = Vector{Int}(undef, n_prod_indices)
    prod_form_cols = Vector{Int}(undef, n_prod_indices)
    prod_form_eval = Vector{Float64}(undef, n_prod_indices)

    for i ∈1:3
        for j ∈1:3
            prod_form_rows, prod_form_cols, prod_form_eval = inner_product_1_form_component!(
                prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, 
                inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, 
                n_indices_1, n_indices_2, (i,j)) # Evaluates α¹ᵢβ¹ⱼgⁱʲ√det(g)
        end
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
- `prod_form_rows::Vector{Int}`: Indices of the first form expression. 
- `prod_form_cols::Vector{Int}`: Indices of the seconWd form expression. 
- `prod_form_eval::Vector{Float64}`: Evaluation of the inner product. Hence, prod_form_eval[l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. 
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{3, 2, G}, form_expression2::AbstractFormExpression{3, 2, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {G<:Geometry.AbstractGeometry{3}}
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    # assumes that the output is of the following form:
    # form_eval::Vector{Matrix{Float64}} of length 3, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # get dimension of each form expression
    n_indices_1 = length(form1_indices)
    n_indices_2 = length(form2_indices)

    n_prod_indices = n_indices_1*n_indices_2

    # Form space 1: α² = α²₁dξ₂∧dξ₃ + α²₂dξ₃∧dξ₁ + α²₃dξ₁∧dξ₂
    # Form space 2: β² = β²₁dξ₂∧dξ₃ + β²₂dξ₃∧dξ₁ + β²₃dξ₁∧dξ₂
    # ⟨α², β²⟩ = ∫α²ᵢβ²ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)dξ¹∧dξ²∧dξ³
    
    prod_form_rows = Vector{Int}(undef, n_total_indices)
    prod_form_cols = Vector{Int}(undef, n_total_indices)
    prod_form_eval = Vector{Float64}(undef, n_total_indices)

    for i ∈1:3
        for j ∈1:3
            prod_form_rows, prod_form_cols, prod_form_eval = inner_product_2_form_component!(
                prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, 
                inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, 
                n_indices_1, n_indices_2, (i,j)
            ) # Evaluates α¹ᵢβ¹ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# Helpers for inner product

function inner_product_0_form_component!(prod_form_rows::Vector{Int}, prod_form_cols::Vector{Int}, prod_form_eval::Vector{Float64}, quad_rule, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2)

    for j ∈ 1:n_indices_2
        for i ∈ 1:n_indices_1
            linear_idx = i + (j-1) * n_indices_1

            prod_form_rows[linear_idx] = form1_indices[i]
            prod_form_cols[linear_idx] = form2_indices[j]

            prod_form_eval[linear_idx] = @views (quad_rule.weights .* form1_eval[1][:,i])' * (form2_eval[1][:,j] .* sqrt_g)
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function inner_product_n_form_component!(prod_form_rows::Vector{Int}, prod_form_cols::Vector{Int}, prod_form_eval::Vector{Float64}, quad_rule, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2)

    for j ∈ 1:n_indices_2
        for i ∈ 1:n_indices_1
            linear_idx = i + (j-1) * n_indices_1

            prod_form_rows[linear_idx] = form1_indices[i]
            prod_form_cols[linear_idx] = form2_indices[j]

            prod_form_eval[linear_idx] = @views (quad_rule.weights .* form1_eval[1][:,i])' * (form2_eval[1][:,j] .*(sqrt_g.^(-1)))    
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function inner_product_1_form_component!(prod_form_rows::Vector{Int}, prod_form_cols::Vector{Int}, prod_form_eval::Vector{Float64}, quad_rule, inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2, form_idxs::NTuple{2, Int})
    for j ∈ 1:n_indices_2
        for i ∈ 1:n_indices_1
            linear_idx = i + (j-1) * n_indices_1

            prod_form_rows[linear_idx] = form1_indices[i]
            prod_form_cols[linear_idx] = form2_indices[j]

            prod_form_eval[linear_idx] += @views (quad_rule.weights .* form1_eval[form_idxs[1]][:,i])' * (form2_eval[form_idxs[2]][:,j] .* inv_g[:,form_idxs[1],form_idxs[2]] .* sqrt_g)
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function inner_product_2_form_component!(prod_form_rows::Vector{Int}, prod_form_cols::Vector{Int}, prod_form_eval::Vector{Float64}, quad_rule, inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2, component_idx::Int, form_idxs::NTuple{2, Int}, indices_offset)

    inv_indices_1 = (mod(form_idxs[1], 3) + 1, mod(form_idxs[1]+1, 3) + 1)
    inv_indices_2 = (mod(form_idxs[2], 3) + 1, mod(form_idxs[2]+1, 3) + 1)

    inv_g_factor = inv_g[:,inv_indices_1[1],inv_indices_2[1]].*inv_g[:,inv_indices_1[2],inv_indices_2[2]] .- inv_g[:,inv_indices_1[1],inv_indices_2[2]].*inv_g[:,inv_indices_1[2],inv_indices_2[1]]

    for j ∈ 1:n_indices_2
        for i ∈ 1:n_indices_1
            linear_idx = i + (j-1) * n_indices_1

            prod_form_rows[linear_idx] = form1_indices[i]
            prod_form_cols[linear_idx] = form2_indices[j]

            prod_form_eval[linear_idx] += @views (quad_rule.weights .* form1_eval[form_idxs[1]][:,i])' * (form2_eval[form_idxs[2]][:,j] .* inv_g_factor .* sqrt_g)
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# Priority 2: Wedge products

# Priority 3: Triple products

# Useful proxy vector fields associated to forms

@doc raw"""
    evaluate_sharp(form_expression::AbstractFormExpression{manifold_dim, 1, G}, 
                   element_id::Int, 
                   xi::NTuple{manifold_dim, Vector{Float64}}) 
                   where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}

Compute the sharp of a differential 1-form over a specified element of a manifold, converting the form into a vector field. Note that both the 1-form and the vector-field are defined in reference, curvilinear coordinates.

# Arguments
- `form_expression::AbstractFormExpression{manifold_dim, 1, G}`: An expression representing the 1-form on the manifold.
- `element_id::Int`: The identifier of the element on which the sharp is to be evaluated.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the 1-form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# Returns
- `sharp_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
- `sharp_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the indices of the evaluated basis functions.
"""
function evaluate_sharp(form_expression::AbstractFormExpression{manifold_dim, 1, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
    manifold_dim >= 2 || throw(ArgumentError("Manifold dimension should be 2 or 3 for 1-forms. Dimension $manifold_dim was given."))

    inv_g, _, _ = Geometry.inv_metric(form_expression.geometry, element_id, xi)

    num_form_components = manifold_dim # = binomial(manifold_dim, 1)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)

    sharp_eval = Vector{Matrix{Float64}}(undef, num_form_components)

    # ♯: dξⁱ ↦ ♯(dξⁱ) = gⁱʲ∂ⱼ
    for component ∈ 1:num_form_components
        sharp_eval[component] = @views hcat([form_eval[i] .* inv_g[:, i, component] for i in 1:num_form_components]...)
    end

    sharp_indices = repeat([vcat(form_indices...)], num_form_components)

    return sharp_eval, sharp_indices
end

@doc raw"""
    evaluate_pushforward(vfield::Vector{Matrix{Float64}}, 
                         jacobian::Array{Float64,3})

Evaluate the pushforward of the vector field at the discrete points where it has been evaluated. The pushforward is the action of the Jacobian of the field on the field itself.

# Arguments
- `vfield::Vector{Matrix{Float64}}`: The pointwise evaluated vector field to evaluate the pushforward for.
- `jacobian::Array{Float64,3}`: The Jacobian of the vector field evaluated at the discrete points.
- `manifold_dim::Int`: The dimension of the embedding manifold.

# Returns
- `::Vector{Matrix{Float64}}`: The evaluated pushforward of the vector field at the discrete points.
"""
function evaluate_pushforward(vfield::Vector{Matrix{Float64}}, jacobian::Array{Float64,3}, manifold_dim::Int)
    image_dim = size(jacobian,2)
    
    # Gᵢ: v ↦ Gᵢ(v) = Jᵢⱼvʲ
    evaluated_pushforward = Vector{Matrix{Float64}}(undef, image_dim)
    for component ∈ 1:image_dim
        evaluated_pushforward[component] = @views reduce(+,[vfield[i] .* jacobian[:, component, i] for i in 1:manifold_dim])
    end

    return evaluated_pushforward
end

@doc raw"""
    evaluate_sharp_pushforward(form_expression::AbstractFormExpression{manifold_dim, 1, G}, 
                                element_id::Int, 
                                xi::NTuple{manifold_dim, Vector{Float64}}) 
                                where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}

Compute the pushforward of the sharp of a differential 1-form over a specified element of a manifold, converting the form into a vector field. Note that the output vector-field is defined in physical coordinates.

# Arguments
- `form_expression::AbstractFormExpression{manifold_dim, 1, G}`: An expression representing the 1-form on the manifold.
- `element_id::Int`: The identifier of the element on which the sharp is to be evaluated.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the 1-form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# Returns
- `sharp_eval::Vector{Matrix{Float64}}`: Each component of the pushed-forward vector, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
- `sharp_indices::Vector{Vector{Int}}`: Each component of the vector, stores the indices of the evaluated basis functions.
"""
function evaluate_sharp_pushforward(form_expression::AbstractFormExpression{manifold_dim, 1, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
    sharp_eval, sharp_indices = evaluate_sharp(form_expression, element_id, xi)
    
    # dξⁱ ↦ G∘♯ (dξⁱ)
    jacobian = Geometry.jacobian(form_expression.geometry, element_id, xi)
    evaluated_pushforward = evaluate_pushforward(sharp_eval, jacobian, manifold_dim)
    
    return evaluated_pushforward, sharp_indices
end

@doc raw"""
    evaluate_rotated_proxy_vector_field(form_expression::AbstractFormExpression{manifold_dim, form_rank, G}, 
                                        element_id::Int, 
                                        xi::NTuple{manifold_dim, Vector{Float64}}) 
                                        where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Compute the rotated proxy vector-field associated to a differential (n-1)-form over a specified element of an n-dimensional manifold, converting the form into a vector field. Note that both the form and the vector-field are defined in reference, curvilinear coordinates.

# Arguments
- `form_expression::AbstractFormExpression{manifold_dim, form_rank, G}`: An expression representing the form on the manifold.
- `element_id::Int`: The identifier of the element on which the sharp is to be evaluated.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# Returns
- `sharp_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
- `hodge_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the indices of the evaluated basis functions. Uses the indices of the hodge evaluation because they are the same.

# Throws an error if the manifold dimension is less than 2 or the form rank is not one less than the manifold dimension.
"""
function evaluate_rotated_proxy_vector_field(form_expression::AbstractFormExpression{manifold_dim, form_rank, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    manifold_dim >= 2 && form_rank == manifold_dim-1 || throw(ArgumentError("Manifold dimension should be 2 or higher and form rank should be 1 less than its value. Dimension $manifold_dim and form rank $form_rank were given."))

    # Examples...
    # 2D: dξⁱ ↦ ♯∘★ (dξⁱ)
    # 3D: dξⁱ∧dξʲ ↦ ♯∘★ (dξⁱ∧dξʲ)
    # ... and so on.
    hodge_form = hodge(form_expression)
    return evaluate_sharp(hodge_form, element_id, xi)

    # OLD IMPLEMENTATION BELOW:
    # hodge_eval, hodge_indices = evaluate(hodge(form_expression), element_id, xi)
    # sharp_eval = Vector{Matrix{Float64}}(undef, num_form_components)

    # for component ∈ 1:num_form_components
    #     sharp_eval[component] = @views reduce(+,[hodge_eval[i] .* inv_g[:, i, component] for i in 1:num_form_components])
    # end

    # return sharp_eval, hodge_indices
end

@doc raw"""
    evaluate_rotated_proxy_vector_fiel_pushforward(form_expression::AbstractFormExpression{manifold_dim, form_rank, G}, 
                                        element_id::Int, 
                                        xi::NTuple{manifold_dim, Vector{Float64}}) 
                                        where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Compute the pushforward of the rotated proxy vector-field associated to a differential (n-1)-form over a specified element of an n-dimensional manifold, converting the form into a vector field. Note that both the vector-field is defined in physical coordinates.

# Arguments
- `form_expression::AbstractFormExpression{manifold_dim, form_rank, G}`: An expression representing the form on the manifold.
- `element_id::Int`: The identifier of the element on which the sharp is to be evaluated.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# Returns
- `sharp_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
- `hodge_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each ∂ᵢ, stores the indices of the evaluated basis functions. Uses the indices of the hodge evaluation because they are the same.

# Throws an error if the manifold dimension is less than 2 or the form rank is not one less than the manifold dimension.
"""
function evaluate_rotated_proxy_vector_field_pushforward(form_expression::AbstractFormExpression{manifold_dim, form_rank, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    manifold_dim >= 2 && form_rank == manifold_dim-1 || throw(ArgumentError("Manifold dimension should be 2 or higher and form rank should be 1 less than its value. Dimension $manifold_dim and form rank $form_rank were given."))

    # Examples...
    # 2D: dξⁱ ↦ G∘♯∘★(dξⁱ)
    # 3D: dξⁱ∧dξʲ ↦ G∘♯∘★(dξⁱ∧dξʲ)
    # ... and so on.
    hodge_form = hodge(form_expression)
    return evaluate_sharp_pushforward(hodge_form, element_id, xi)

end

# @doc raw"""
#     evaluate_flat(form_expression::AbstractFormExpression{manifold_dim, 1, G}, 
#                    element_id::Int, 
#                    xi::NTuple{manifold_dim, Vector{Float64}}) 
#                    where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}

# Compute the flat of a vector field over a specified element of a manifold, converting the vector field into a 1-form. Note that both the 1-form and the vector-field are defined in reference, curvilinear coordinates.

# # Arguments
# - `form_expression::AbstractFormExpression{manifold_dim, 1, G}`: An expression representing the vector field on the manifold.
# - `element_id::Int`: The identifier of the element on which the flat is to be evaluated.
# - `xi::NTuple{manifold_dim, Vector{Float64}}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the 1-form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# # Returns
# - `flat_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each dξⁱ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
# - `flat_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each dξⁱ, stores the indices of the evaluated basis functions.
# """
# function evaluate_flat(form_expression::AbstractFormExpression{manifold_dim, 1, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
#     manifold_dim >= 2 || throw(ArgumentError("Manifold dimension should be 2 or 3 for 1-forms. Dimension $manifold_dim was given."))

#     g, _ = Geometry.metric(form_expression.geometry, element_id, xi)

#     num_form_components = manifold_dim # = binomial(manifold_dim, 1)

#     form_eval, form_indices = evaluate(form_expression, element_id, xi)

#     flat_eval = Vector{Matrix{Float64}}(undef, num_form_components)

#     for component ∈ 1:num_form_components
#         flat_eval[component] = @views hcat([form_eval[i] .* g[:, i, component] for i in 1:num_form_components]...)
#     end

#     flat_indices = repeat([vcat(form_indices...)], num_form_components)

#     return flat_eval, flat_indices
# end

# @doc raw"""
#     evaluate_flat(form_expression::AbstractFormExpression{manifold_dim, 2, G}, 
#                    element_id::Int, 
#                    xi::NTuple{manifold_dim, Vector{Float64}}) 
#                    where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}

# Compute the flat of a vector field over a specified element of a manifold, converting the vector field into a form.

# # Arguments
# - `form_expression::AbstractFormExpression{manifold_dim, 2, G}`: An expression representing the vector field on the manifold.
# - `element_id::Int`: The identifier of the element on which the flat is to be evaluated.
# - `xi::NTuple{manifold_dim, Vector{Float64}}`: A tuple containing vectors of floating-point numbers representing the coordinates at which the 2-form is evaluated. Each vector within the tuple corresponds to a dimension of the manifold.

# # Returns
# - `flat_eval::Vector{Matrix{Float64}}`: Each component of the vector, corresponding to each dξⁱ, stores the sharp evaluation. The size of each matrix is (number of evaluation points)x(number of basis functions).
# - `hodge_indices::Vector{Vector{Int}}`: Each component of the vector, corresponding to each dξⁱ, stores the indices of the evaluated basis functions. Uses the indices of the hodge evaluation because they are the same.
# """
# function evaluate_flat(form_expression::AbstractFormExpression{manifold_dim, 2, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
#     manifold_dim == 3 || throw(ArgumentError("Manifold dimension should be 3 for 2-forms. Dimension $manifold_dim was given."))

#     g, _ = Geometry.metric(form_expression.geometry, element_id, xi)

#     num_form_components = manifold_dim #binomial(manifold_dim, 2)

#     hodge_eval, hodge_indices = evaluate(hodge(form_expression), element_id, xi)

#     flat_eval = Vector{Matrix{Float64}}(undef, num_form_components)

#     # ♯: dξⁱ∧dξʲ ↦ ♯★dξⁱ∧dξʲ
#     for component ∈ 1:num_form_components
#         flat_eval[component] = @views reduce(+, [hodge_eval[i] .* g[:, i, component] for i in 1:num_form_components])
#     end

#     return flat_eval, hodge_indices
# end

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

    hodge_indices = repeat([vcat(form_indices...)], 3)

    return hodge_eval, hodge_indices
end
