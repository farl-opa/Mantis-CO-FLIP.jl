# Priority 1: Inner products for abstract form fields

# (0-forms, 0-forms)
@doc raw"""
    evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, 0, expression_rank_1, G}, 
                           form_expression2::AbstractFormExpression{manifold_dim, 0, expression_rank_2, G}, 
                           element_id::Int, 
                           quad_rule::Quadrature.QuadratureRule{manifold_dim}) 
                           where {manifold_dim, G<:Geometry.AbstractGeometry{manifold_dim}}

Compute the inner product of two differential 0-forms over a specified element of a manifold.

# Arguments
- `form_expression1::AbstractFormExpression{manifold_dim, 0, expression_rank_1, G}`: The first differential form expression. It represents a form on a manifold of dimension `manifold_dim`.
- `form_expression2::AbstractFormExpression{manifold_dim, 0, expression_rank_2, G}`: The second differential form expression. It should have the same dimension and geometry as `form_expression1`.
- `element_id::Int`: The identifier of the element on which the inner product is to be evaluated. This is typically an index into a mesh or other discretized representation of the manifold.
- `quad_rule::Quadrature.QuadratureRule{manifold_dim}`: The quadrature rule used to approximate the integral of the inner product over the specified element. The quadrature rule should be compatible with the dimension of the manifold.

# Returns
- `prod_form_rows::Vector{Int}`: Indices of the first form expression. 
- `prod_form_cols::Vector{Int}`: Indices of the second form expression. 
- `prod_form_eval::Vector{Float64}`: Evaluation of the inner product. Hence, prod_form_eval[l(i,j)] stores the inner product of the i-indexed function of the first form expression with the j-indexed function from the second form expression, where l(i,j) is a linear indexing. 
"""
function evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, 0, expression_rank_1, G}, form_expression2::AbstractFormExpression{manifold_dim, 0, expression_rank_2, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, expression_rank_1, expression_rank_2, G<:Geometry.AbstractGeometry{manifold_dim}}
    if expression_rank_1 > 1
        throw(ArgumentError("Inner product only valid between expressions with expression rank < 2, got: $expression_rank_1"))
    end
    if expression_rank_2 > 1
        throw(ArgumentError("Inner product only valid between expressions with expression rank < 2, got: $expression_rank_2"))
    end

    _, sqrt_g = Geometry.metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)
    
    # evaluate both the form expressions; it is assumed that the output is of the following form:
    # form_eval::Vector{Matrix{Float64}} of length 1, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # Get dimension of each form expression
    # Since we restrict to expresions with expression rank 0 or 1, we only have one set of indices
    # If we wish to have inner product between expressions with higher ranks, this needs to be changed
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    # Form 1: α⁰ = α⁰₁
    # Form 2: β⁰ = β⁰₁
    # ⟨α⁰, β⁰⟩ = ∫α⁰₁β⁰₁ⱼ√det(g)dξ¹∧…∧dξⁿ
    
    prod_form_rows = Vector{Int}(undef, n_indices_1*n_indices_2)
    prod_form_cols = Vector{Int}(undef, n_indices_1*n_indices_2)
    prod_form_eval = Vector{Float64}(undef, n_indices_1*n_indices_2)

    prod_form_rows, prod_form_cols, prod_form_eval = inner_product_0_form_component!(
        prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, sqrt_g, 
        form1_eval, form1_indices[1], form2_eval, form2_indices[1], 
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
function evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, manifold_dim, expression_rank_1, G}, form_expression2::AbstractFormExpression{manifold_dim, manifold_dim, expression_rank_2, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, expression_rank_1, expression_rank_2, G<:Geometry.AbstractGeometry{manifold_dim}}
    if expression_rank_1 > 1
        throw(ArgumentError("Inner product only valid between expressions with expression rank < 2, got: $expression_rank_1"))
    end
    if expression_rank_2 > 1
        throw(ArgumentError("Inner product only valid between expressions with expression rank < 2, got: $expression_rank_2"))
    end

    _, sqrt_g = Geometry.metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    # form_eval::Vector{Matrix{Float64}} of length 1, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # Get dimension of each form expression
    # Since we restrict to expresions with expression rank 0 or 1, we only have one set of indices
    # If we wish to have inner product between expressions with higher ranks, this needs to be changed
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    # Form 1: αⁿ = αⁿ₁dξ¹∧…∧dξⁿ
    # Form 2: βⁿ = βⁿ₁dξ¹∧…∧dξⁿ
    # ⟨αⁿ, βⁿ⟩ = ∫αⁿ₁βⁿ₁ⱼ√det(g)⁻¹dξ¹∧…∧dξⁿ
    
    prod_form_rows = Vector{Int}(undef, n_indices_1*n_indices_2)
    prod_form_cols = Vector{Int}(undef, n_indices_1*n_indices_2)
    prod_form_eval = Vector{Float64}(undef, n_indices_1*n_indices_2)

    prod_form_rows, prod_form_cols, prod_form_eval = inner_product_n_form_component!(
        prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, sqrt_g,
        form1_eval, form1_indices[1], form2_eval, form2_indices[1], 
        n_indices_1, n_indices_2
    ) # Evaluates αⁿ₁βⁿ₁ⱼ√det(g)⁻¹

    return prod_form_rows, prod_form_cols, prod_form_eval
end 

# (1-forms, 1-forms)
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
function evaluate_inner_product(form_expression1::AbstractFormExpression{manifold_dim, 1, expression_rank_1, G}, form_expression2::AbstractFormExpression{manifold_dim, 1, expression_rank_2, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, expression_rank_1, expression_rank_2, G<:Geometry.AbstractGeometry{manifold_dim}}
    if expression_rank_1 > 1
        throw(ArgumentError("Inner product only valid between expressions with expression rank < 2, got: $expression_rank_1"))
    end
    if expression_rank_2 > 1
        throw(ArgumentError("Inner product only valid between expressions with expression rank < 2, got: $expression_rank_2"))
    end

    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    # form_eval::Vector{Matrix{Float64}} of length manifold_dim, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # Get dimension of each form expression
    # Since we restrict to expresions with expression rank 0 or 1, we only have one set of indices
    # If we wish to have inner product between expressions with higher ranks, this needs to be changed
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    n_prod_indices = n_indices_1*n_indices_2
    
    ## 2D case:
    # Form 1: α¹ = α¹₁dξ¹ +  α¹₂dξ²
    # Form 2: β¹ = β¹₁dξ¹ +  β¹₂dξ²
    # ⟨α¹, β¹⟩ = ∫α¹ᵢβ¹ⱼgⁱʲ√det(g)dξ¹∧dξ²
    ## 3D case:
    # Form 1: α¹ = α¹₁dξ¹ +  α¹₂dξ² + α¹₃dξ³
    # Form 2: β¹ = β¹₁dξ¹ +  β¹₂dξ² + β¹₃dξ³
    # ⟨α¹, β¹⟩ = ∫α¹ᵢβ¹ⱼgⁱʲ√det(g)dξ¹∧dξ²∧dξ³
    
    prod_form_rows = zeros(Int, n_prod_indices)
    prod_form_cols = zeros(Int, n_prod_indices)
    prod_form_eval = zeros(Float64, n_prod_indices)

    for i ∈ 1:manifold_dim
        for j ∈ 1:manifold_dim
            prod_form_rows, prod_form_cols, prod_form_eval = inner_product_1_form_component!(
                prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, 
                inv_g, sqrt_g, form1_eval, form1_indices[1], form2_eval, form2_indices[1], 
                n_indices_1, n_indices_2, (i,j)) # Evaluates α¹ᵢβ¹ⱼgⁱʲ√det(g)
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# (2-forms, 2-forms) in 3D
@doc raw"""
    evaluate_inner_product(form_expression1::AbstractFormExpression{3, 2, expression_rank_1, G}, 
                           form_expression2::AbstractFormExpression{3, 2, expression_rank_2, G}, 
                           element_id::Int, 
                           quad_rule::Quadrature.QuadratureRule{3}) 
                           where {expression_rank_1, expression_rank_2, G<:Geometry.AbstractGeometry{3}}

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
function evaluate_inner_product(form_expression1::AbstractFormExpression{3, 2, expression_rank_1, G}, form_expression2::AbstractFormExpression{3, 2, expression_rank_2, G}, element_id::Int, quad_rule::Quadrature.QuadratureRule{3}) where {expression_rank_1, expression_rank_2, G<:Geometry.AbstractGeometry{3}}
    if expression_rank_1 > 1
        throw(ArgumentError("Inner product only valid between expressions with expression rank < 2, got: $expression_rank_1"))
    end
    if expression_rank_2 > 1
        throw(ArgumentError("Inner product only valid between expressions with expression rank < 2, got: $expression_rank_2"))
    end
    
    inv_g, _, sqrt_g = Geometry.inv_metric(get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes)

    # assumes that the output is of the following form:
    # form_eval::Vector{Matrix{Float64}} of length 3, where each matrix is of size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)
    
    # Get dimension of each form expression
    # Since we restrict to expresions with expression rank 0 or 1, we only have one set of indices
    # If we wish to have inner product between expressions with higher ranks, this needs to be changed
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    n_prod_indices = n_indices_1*n_indices_2

    # Form space 1: α² = α²₁dξ₂∧dξ₃ + α²₂dξ₃∧dξ₁ + α²₃dξ₁∧dξ₂
    # Form space 2: β² = β²₁dξ₂∧dξ₃ + β²₂dξ₃∧dξ₁ + β²₃dξ₁∧dξ₂
    # ⟨α², β²⟩ = ∫α²ᵢβ²ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)dξ¹∧dξ²∧dξ³
    
    prod_form_rows = zeros(Int, n_prod_indices)
    prod_form_cols = zeros(Int, n_prod_indices)
    prod_form_eval = zeros(Float64, n_prod_indices)

    for i ∈1:3
        for j ∈1:3
            prod_form_rows, prod_form_cols, prod_form_eval = inner_product_2_form_component!(
                prod_form_rows, prod_form_cols, prod_form_eval, quad_rule, 
                inv_g, sqrt_g, form1_eval, form1_indices[1], form2_eval, form2_indices[1], 
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

function inner_product_2_form_component!(prod_form_rows::Vector{Int}, prod_form_cols::Vector{Int}, prod_form_eval::Vector{Float64}, quad_rule, inv_g, sqrt_g, form1_eval, form1_indices, form2_eval, form2_indices, n_indices_1, n_indices_2, form_idxs::NTuple{2, Int})

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

# TODO Is the sharp supposed to work for expressions with expression_rank higher than 0? If not, then enforce it.
@doc raw"""
    evaluate_sharp(form_expression::AbstractFormExpression{manifold_dim, 1, expression_rank, G}, 
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
function evaluate_sharp(form_expression::AbstractFormExpression{manifold_dim, 1, expression_rank, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    manifold_dim >= 2 || throw(ArgumentError("Manifold dimension should be 2 or 3 for 1-forms. Dimension $manifold_dim was given."))

    inv_g, _, _ = Geometry.inv_metric(form_expression.geometry, element_id, xi)

    num_form_components = manifold_dim # = binomial(manifold_dim, 1)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)

    sharp_eval = Vector{Matrix{Float64}}(undef, num_form_components)

    # ♯: dξⁱ ↦ ♯(dξⁱ) = gⁱʲ∂ⱼ
    for component ∈ 1:num_form_components
        sharp_eval[component] = @views hcat([form_eval[i] .* inv_g[:, i, component] for i in 1:num_form_components]...)
    end

    return sharp_eval, form_indices
end

# TODO Is the pushforward supposed to work for expressions with expression_rank higher than 0? If not, then enforce it.
#      There should be a vector expression and a vector field.
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

# TODO Similar thing here. This is a concrete expression. If we make general expressions and combine them, everything becomes simpler.
""" 
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
    evaluate_rotated_proxy_vector_field_pushforward(form_expression::AbstractFormExpression{manifold_dim, form_rank, G}, 
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
    evaluate_hodge_star(form_expression::AbstractFormExpression{manifold_dim, 0, expression_rank, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}})

Evaluates the Hodge star operator ⋆ of a 0-form in n dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{manifold_dim, 0, expression_rank, G}`: The differential 0-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate,
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `form_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{manifold_dim, 0, expression_rank, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    if expression_rank > 1
        throw(ArgumentError("Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank"))
    end
    
    _, sqrt_g = Geometry.metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # We restrict the evaluation of hodge-star to expression of rank 1 and lower, therefore we can 
    # select just the first index of the vector of indices, for higher rank expression we will 
    # need to change this
    n_indices = length(form_indices[1])

    hodge_eval = Vector{Matrix{Float64}}(undef, 1)

    # ⋆α₁⁰ = α₁⁰√det(gᵢⱼ) dξ₁∧…∧dξₙ. 
    hodge_eval[1] = reshape(form_eval[1] .* sqrt_g, (:, n_indices)) # TODO: is this reshape needed?
 
    return hodge_eval, form_indices
end

# n-forms (n dimensions)
 @doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{manifold_dim, manifold_dim, expression_rank, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Evaluates the Hodge star operator ⋆ of a n-form in n dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{manifold_dim, manifold_dim, expression_rank, G}`: The differential n-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval`: Vector of arrays containing evaluated hodge star of the basis functions.
- `form_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{manifold_dim, manifold_dim, expression_rank, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    if expression_rank > 1
        throw(ArgumentError("Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank"))
    end
    
    _, sqrt_g = Geometry.metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # Because we restrict ourselves to expression with rank 0 or 1, we can 
    # extract the first set of indices (only set), for higher ranks, we need 
    # to change this
    n_indices = length(form_indices[1])

    hodge_eval = Vector{Matrix{Float64}}(undef, 1)

    # ⋆α₁ⁿdξ₁∧…∧dξₙ = α₁ⁿ(√det(gᵢⱼ))⁻¹. 
    hodge_eval[1] = reshape(form_eval[1] .* (sqrt_g.^(-1)), (:, n_indices)) # TODO: is this reshape needed?
 
    return hodge_eval, form_indices
end

# 1-forms (2 dimensions)
@doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{2, 1, expression_rank, G}, element_id::Int, xi::NTuple{2, Vector{Float64}}) where {expression_rank, G <: Geometry.AbstractGeometry{2}}

Evaluates the Hodge star operator ⋆ of a 1-form in 2 dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{2, 1, expression_rank, G}`: The differential 1-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{2, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval::Vector{Matrix{Float64}}`: Vector of arrays containing evaluated hodge star of the basis functions.
- `hodge_indices`: Vector of vectors containing indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{2, 1, expression_rank, G}, element_id::Int, xi::NTuple{2, Vector{Float64}}) where {expression_rank,G <: Geometry.AbstractGeometry{2}}
    if expression_rank > 1
        throw(ArgumentError("Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank"))
    end

    inv_g, _, sqrt_g = Geometry.inv_metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    
    n_eval_points = length(sqrt_g)
    # Because we restrict ourselves to expression with rank 0 or 1, we can 
    # extract the first set of indices (only set), for higher ranks, we need 
    # to change this
    n_indices = length(form_indices[1])
    hodge_eval = [zeros(n_eval_points, n_indices) for _ in 1:2]
    
    # ⋆(α₁¹dξ₁+α₂¹dξ₂) = [-(α₁¹g²¹+α₂¹g²²)dξ₁ + (α₁¹g¹¹+α₂¹g¹²)dξ₂]√det(gᵢⱼ). 
    # First: -(α₁¹g²¹+α₂¹g²²)dξ₁
    for i in 1:2
        hodge_eval[1] .+= @views -form_eval[i] .* inv_g[:, 2, i]
    end
    hodge_eval[1] .*= sqrt_g

    # Second: (α₁¹g¹¹+α₂¹g¹²)dξ₂
    for i in 1:2
        hodge_eval[2] .+= @views form_eval[i] .* inv_g[:, 1, i]
    end
    hodge_eval[2] .*= sqrt_g

    return hodge_eval, form_indices
end

# 1-forms (3 dimensions)
@doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{3, 1, expression_rank, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {expression_rank, G <: Geometry.AbstractGeometry{3}}

Evaluates the Hodge star operator ⋆ of a 1-form in 3 dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{3, 1, expression_rank, G}`: The differential 1-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval`: Vector of arrays containing evaluated hodge star of the basis functions for each of the components.
- `hodge_indices`: Vector of containing the indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{3, 1, expression_rank, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {expression_rank, G <: Geometry.AbstractGeometry{3}}
    if expression_rank > 1
        throw(ArgumentError("Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank"))
    end
    
    # Set the number of components of the original expression and the Hodge-⋆ (3 in this case)
    n_expression_components = 3
    n_hodge_components = 3

    # Compute the metric terms
    inv_g, _, sqrt_g = Geometry.inv_metric(form_expression.geometry, element_id, xi)

    # Evaluate the form expression to which we wish to apply the Hodge-⋆
    form_eval, form_indices = evaluate(form_expression, element_id, xi)

    # Preallocate memory for the Hodge evaluation matrix
    n_eval_points = prod(size.(xi, 1))  # number of points where the forms are evaluated (tensor product)
    # Because we restrict ourselves to expression with rank 0 or 1, we can 
    # extract the first set of indices (only set), for higher ranks, we need 
    # to change this
    n_active_basis = length(form_indices[1])
    hodge_eval = [zeros(n_eval_points, n_active_basis) for _ in 1:n_hodge_components]

    # Compute the Hodge-⋆ following the analytical expression
    # ⋆(α₁¹dξ₁+α₂¹dξ₂+α₃¹dξ₃) = [(α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃ + (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁ + (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂]√det(gᵢⱼ). 
    
    # First: (α₁¹g¹¹+α₂¹g¹²+α₃¹g¹³)dξ₂∧dξ₃  --> component 1
    # Second: (α₁¹g²¹+α₂¹g²²+α₃¹g²³)dξ₃∧dξ₁ --> component 2
    # Third: (α₁¹g³¹+α₂¹g³²+α₃¹g³³)dξ₁∧dξ₂  --> component 3
    # Done in a double for loop
    for k_hodge_component in 1:n_hodge_components
        # (α₁¹gᵏ¹+α₂gᵏ²+α₃gᵏ³)  --> to be added to the component k of the Hodge 
        for k_expression_component in 1:n_expression_components
            hodge_eval[k_hodge_component] .+= @views form_eval[k_expression_component] .* inv_g[:, k_hodge_component, k_expression_component]
        end

        # Multiply by √det(gᵢⱼ) to get (α₁¹gᵏ¹+α₂gᵏ²+α₃gᵏ³) √det(gᵢⱼ)
        hodge_eval[k_hodge_component] .*= sqrt_g
    end

    return hodge_eval, form_indices
end

# 2-forms (3 dimensions)
@doc raw"""
    evaluate_hodge_star(form_expression::AbstractFormExpression{3, 2, expression_rank, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {expression_rank, G <: Geometry.AbstractGeometry{3}}

Evaluates the Hodge star operator ⋆ of a 2-form in 3 dimensions.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{3, 2, G}`: The differential 2-form expression. It represents a form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `hodge_eval`: Vector of arrays containing evaluated hodge star of the basis functions for each of the components.
- `hodge_indices`: Vector of containing the indices of the basis functions.
"""
function evaluate_hodge_star(form_expression::AbstractFormExpression{3, 2, expression_rank, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {expression_rank, G <: Geometry.AbstractGeometry{3}}
    if expression_rank > 1
        throw(ArgumentError("Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank"))
    end
    
    # Set the number of components of the original expression and the Hodge-⋆ (3 in this case)
    n_expression_components = 3
    n_hodge_components = 3

    # The Hodge-⋆ of a 2-form in 3D is the inverse of the Hodge-⋆ of a 1-form in 3D 
    # Therefore we can use the metric tensor instead of the inverse metric tensor 
    # and use the same expression as for the Hodge-⋆ of 1-forms but now using the 
    # metric tensor instead of the inverse of the metric tensor.

    # Compute the metric terms
    g, sqrt_g = Geometry.metric(form_expression.geometry, element_id, xi)

    # Evaluate the form expression to which we wish to apply the Hodge-⋆
    form_eval, form_indices = evaluate(form_expression, element_id, xi)

    # Preallocate memory for the Hodge evaluation matrix
    n_eval_points = prod(size.(xi, 1))  # number of points where the forms are evaluated (tensor product)
    # Because we restrict ourselves to expression with rank 0 or 1, we can 
    # extract the first set of indices (only set), for higher ranks, we need 
    # to change this
    n_active_basis = length(form_indices[1])
    hodge_eval = [zeros(n_eval_points, n_active_basis) for _ in 1:n_hodge_components]

    # Compute the Hodge-⋆ following the analytical expression
    # ⋆(α₁dξ²∧dξ³+α₂dξ³∧dξ¹+α₃dξ¹∧dξ²) = [(α₁g₁₁+α₂¹g₁₂+α₃¹g₁₃)dξ₁ + (α₁g₂₁+α₂g₂₂+α₃g₂₃)dξ₂ + (α₁g₃₁+α₂g₃₂+α₃g₃₃)dξ₃]/√det(gᵢⱼ). 
    
    # First: (α₁g₁₁+α₂¹g₁₂+α₃¹g₁₃)dξ₁  --> component 1
    # Second: (α₁g₂₁+α₂g₂₂+α₃g₂₃)dξ₂   --> component 2
    # Third: (α₁g₃₁+α₂g₃₂+α₃g₃₃)dξ₃    --> component 3
    # Done in a double for loop
    for k_hodge_component in 1:n_hodge_components
        # (α₁gₖ₁+α₂gₖ₂+α₃gₖ₃)  --> to be added to the component k of the Hodge 
        for k_expression_component in 1:n_expression_components
            hodge_eval[k_hodge_component] .+= @views form_eval[k_expression_component] .* g[:, k_hodge_component, k_expression_component]
        end

        # Divide by √det(gᵢⱼ) to get (α₁gₖ₁+α₂gₖ₂+α₃gₖ₃) / √det(gᵢⱼ)
        hodge_eval[k_hodge_component] ./= sqrt_g
    end

    return hodge_eval, form_indices
end


@doc raw"""
    evaluate_wedge(form_expression_1::AbstractFormExpression{manifold_dim, form_rank_1, expression_rank_1, G}, form_expression_2::AbstractFormExpression{manifold_dim, form_rank_2, expression_rank_2, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank_1, expression_rank_1, form_rank_2, expression_rank_2, G <: Geometry.AbstractGeometry{manifold_dim}}

Evaluates the wedge operator ∧ between a k-form and an l-form, i.e.,

```math
    \alpha^{k} \wedge \beta^{l}
```

Both ```\alpha^{k}``` and ```\beta^{l}``` can have any expression rank, i.e., can be a `FormField` or `AnalyticalFormField` (rank 0),
a FormSpace (rank 1), or higher rank expressions like the tensor product of FormSpaces (rank n),
as obtained, for example, when computing the wedge between two FormSpaces.

Two cases are relevant to distinguish:

1. ```\alpha^{k}``` or ```\beta^{l}``` is an expression of rank 0 (fields)
2. ```\alpha^{k}``` and ```\beta^{l}``` are expressions of rank > 0 (e.g., `FormSpace`)

1.
In this case, for example, ``\alpha^{k}`` is an expression of rank 0 (the other case is identical), then 
```\alpha^{k} \wedge \beta^{l}``` will have the same expression rank as ```\beta^{l}```. 
If ```\beta^{l}``` has rank ```n``` then ```\beta^{l}``` is an expression with ```n``` indices 

```math
    \beta^{l} =
    \left[ 
        \begin{array}{c}
            \beta^{l}_{i_{1}, \dots, i_{n}}
        \end{array}
    \right]
```

and ```\alpha^{k}``` will have rank 0, therefore no indices

```math
    \alpha^{k} = \alpha^{k} \qquad (\text{no indices})
```

The wedge will then be 

```math
    \left(\alpha^{k} \wedge \beta^{l}\right)_{i_{1}, \dots, i_{n}} = \alpha^{k} \wedge \beta^{l}_{i_{1}, \dots, i_{n}}
```

Note that, because ``\alpha^{k}`` has rank 0, the rank of the wedge will be equal 
to the rank of ``\beta^{l}`` and therefore the same number of indices will be present 
in the wedge product.

2.
In this case, for example, ``\alpha^{k}`` is an expression of rank ``m``, and ``\beta^{l}`` 
is an expression of rank ``n``, then ```\alpha^{k} \wedge \beta^{l}``` will have expression rank 
equal to ``(m + n)``. Again, following what was introduced above, we have that

```math
    \alpha^{k} =
    \left[ 
        \begin{array}{c}
            \alpha^{k}_{s_{1}, \dots, s_{n}}
        \end{array}
    \right]
```

and 

```math
    \beta^{l} =
    \left[ 
        \begin{array}{c}
            \beta^{l}_{i_{1}, \dots, i_{n}}
        \end{array}
    \right]
```

The wedge will then be 

```math
    \left(\alpha^{k} \wedge \beta^{l}\right)_{s_{1}, \dots, s_{m}, i_{1}, \dots, i_{n}} = \alpha^{k}_{s_{1}, \dots, s_{n}} \wedge \beta^{l}_{i_{1}, \dots, i_{n}}
```

Note again that the rank of the wedge will be equal to the sum of the (expression) ranks of each 
of the expressions. The same is true for the (form) ranks, but that is standard for the wedge product.

# Arguments
- `form_expression::form_expression::AbstractFormExpression{manifold_dim, form_rank_1, expression_rank_1, G}`: The differential `form_rank_1`-form expression. It represents a `form_rank_1`-form on a manifold.
- `form_expression::form_expression::AbstractFormExpression{manifold_dim, form_rank_2, expression_rank_2, G}`: The differential `form_rank_2`-form expression. It represents a `form_rank_2`-form on a manifold.
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
    Points are the tensor product of the coordinates per dimension, therefore there will be
    ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `wedge_eval`: Vector of arrays containing evaluated wedge operator of the wedfge expression over each of the components.
        `wedge_eval[j][r, s₁, …, sₘ, i₁, …, iₙ]` corresponds to the evaluation of component `j` at point `r` of 
        the wedge product between the term `[s₁, …, sₘ]` of `form_expression_1` and the term 
        `[i₁, …, iₘ]` of `form_expression_2`. 
        For the two cases discussed above (1 and 2) we will have 

        1.
        ```math 
            \texttt{wedge_eval[j][r,i₁, …, iₙ]} = \left\{\left(\alpha^{k} \wedge \beta^{l}\right)(\boldsymbol{\xi}_{r})_{i_{1}, \dots, i_{n}}\right\}_{j} = \left\{\alpha^{k}(\boldsymbol{\xi}_{r}) \wedge \beta^{l}(\boldsymbol{\xi}_{r})_{i_{1}, \dots, i_{n}}\right\}_{j}
        ```    
        where 
            - ```\beta^{l}_{i_{1}, \dots, i_{n}}(\boldsymbol{\xi}_{r}) is the ``(i_{1}, \dots, i_{n})``-th 
            basis of `\beta^{l} evaluated at the point ``\boldsymbol{ξ}_{r}``
            - ```\alpha^{k}``` is single valued (no basis) therefore no index.
            - ``\left\{\right}_{j}`` is the ``j``-th component of the expression inside the brackets, e.g., 
                for a 1-form in 2D ``\left\{\alpha_{1}\mathrm{d}\xi^{1} + \alpha_{2}\mathrm{d}\xi^{2}\right\}_{j} = \alpha_{j}``
                corresponds to the coefficient of ``\mathrm{d}\xi^{j}``.

        Note that the case where ```\beta^{l}``` is the one with expression rank 0 is the same.

        2.
        ```math 
            \texttt{wedge_eval[j][r, s₁, …, sₘ, i₁, …, iₙ]} = \left\{\left(\alpha^{k} \wedge \beta^{l}\right)(\boldsymbol{\xi}_{r})_{s_{1}, \dots, s_{m}, i_{1}, \dots, i_{n}}\right\}_{j} = \left\{\alpha^{k}(\boldsymbol{\xi}_{r})_{s_{1}, \dots, s_{m}} \wedge \beta^{l}(\boldsymbol{\xi}_{r})_{i_{1}, \dots, i_{n}}\right\}_{j}
        ```    
        where 
            - ```\beta^{l}_{i_{1}, \dots, i_{n}}(\boldsymbol{\xi}_{r}) is the ``(i_{1}, \dots, i_{n})``-th 
            basis of `\beta^{l} evaluated at the point ``\boldsymbol{ξ}_{r}``
            - ```\alpha^{k}_{s_{1}, \dots, s_{m}}(\boldsymbol{\xi}_{r}) is the ``(s_{1}, \dots, s_{m})``-th 
            basis of `\alpha^{k} evaluated at the point ``\boldsymbol{ξ}_{r}``
            - ``\left\{\right}_{j}`` is the ``j``-th component of the expression inside the brackets, e.g., 
                for a 1-form in 2D ``\left\{\alpha_{1}\mathrm{d}\xi^{1} + \alpha_{2}\mathrm{d}\xi^{2}\right\}_{j} = \alpha_{j}``
                corresponds to the coefficient of ``\mathrm{d}\xi^{j}``.
- `wedge_indices`: Vector of Vectors containing the indices of the basis functions for the rank expresion_rank_1 x expression_rank_2
        product. `wedge_indices[k]` contains the indices for dimension k + 1 of wedge_eval[j] (the first
        dimension corresponds to the evaluation points).
"""
function evaluate_wedge(form_expression_1::AbstractFormExpression{manifold_dim, form_rank_1, expression_rank_1, G}, form_expression_2::AbstractFormExpression{manifold_dim, form_rank_2, expression_rank_2, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank_1, form_rank_2, expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{manifold_dim}}
    # Check if total (expression) rank of wedge expression is valid
    if (expression_rank_1 + expression_rank_2) > 2
        throw(ArgumentError("Wedge of FormExpressions requires expressions with total expression rank smaller than three, got: $expression_rank_1 and $expression_rank_2"))
    end

    # Check if total (form) rank of wedge expression is valid
    if (form_rank_1 + form_rank_2) > manifold_dim
        throw(ArgumentError("The wedge product form rank (got $(form_rank_1 + form_rank_2)) cannot exceed the manifold dimension (got $manifold_dim)."))
    end
    
    return _evaluate_wedge(form_expression_1, form_expression_2, element_id, xi)
end

# 0-forms ∧ 0-forms
function _evaluate_wedge(form_expression_1::AbstractFormExpression{manifold_dim, 0, expression_rank_1, G}, form_expression_2::AbstractFormExpression{manifold_dim, 0, expression_rank_2, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{manifold_dim}}

    # Evaluate the two forms that make up the wedge product 
    form_expression_1_eval, form_expression_1_indices = evaluate(form_expression_1, element_id, xi)  # 0-form 
    form_expression_2_eval, form_expression_2_indices = evaluate(form_expression_2, element_id, xi)  # 0-form

    # Compute the wedge product between the 0-form and the 0-form
    
    # Given 
    #   α⁰ := α
    #   β⁰ := β  
    # then 
    #   α⁰ ∧ β⁰ = (αβ)⁰
    
    # Get the number of basis in each expression 
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points 
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge 
    # In this case because it is the wedge of a 0-form with a 0-form 
    # therefore there is only one component
    # n_components_wedge = 1

    # Allocate memory space for wedge_eval
    wedge_eval = [zeros(Float64, n_evaluation_points, num_basis_form_expression_1..., num_basis_form_expression_2...)]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    wedge_component_idx = 1
    form_expression_1_component_idx = 1
    form_expression_2_component_idx = 1
    for form_expression_1_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_1))
        for form_expression_2_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_2))
            wedge_eval[wedge_component_idx][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .= @views form_expression_1_eval[form_expression_1_component_idx][:, form_expression_1_basis_idx] .* form_expression_2_eval[form_expression_2_component_idx][:, form_expression_2_basis_idx]
        end
    end
    
    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms
function _evaluate_wedge(form_expression_1::AbstractFormExpression{manifold_dim, form_rank_1, expression_rank_1, G}, form_expression_2::AbstractFormExpression{manifold_dim, 0, expression_rank_2, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank_1, expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{manifold_dim}}
    # Evaluate the two forms that make up the wedge product 
    form_expression_1_eval, form_expression_1_indices = evaluate(form_expression_1, element_id, xi)  # k-form 
    form_expression_2_eval, form_expression_2_indices = evaluate(form_expression_2, element_id, xi)  # 0-form

    # Compute the wedge product between the 0-form and the 0-form
    
    # Given 
    #   αᵏ := ∑ᵢαᵢ dᵏξᵢ  (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth) 
    #   β⁰ := β  
    # then 
    #   αᵏ ∧ β⁰ = ∑ᵢ αᵢ β dᵏξᵢ
    
    # Get the number of components of the wedge 
    # In this case because it is the wedge of a k-form with a 0-form 
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_1 = binomial(manifold_dim, form_rank_1)
    n_components_wedge = n_components_form_expression_1

    # Get the number of basis in each expression 
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points 
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    wedge_eval = [zeros(Float64, n_evaluation_points, num_basis_form_expression_1..., num_basis_form_expression_2...) for _ ∈ 1:n_components_form_expression_1]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    for wedge_component_idx ∈ 1:n_components_form_expression_1
        for form_expression_1_basis_idx ∈ CartesianIndices(Tuple(num_basis_form_expression_1))
            for form_expression_2_basis_idx ∈ CartesianIndices(Tuple(num_basis_form_expression_2))
                wedge_eval[wedge_component_idx][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .= @views form_expression_1_eval[wedge_component_idx][:, form_expression_1_basis_idx] .* form_expression_2_eval[1][:, form_expression_2_basis_idx]
            end
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms
function _evaluate_wedge(form_expression_1::AbstractFormExpression{manifold_dim, 0, expression_rank_1, G}, form_expression_2::AbstractFormExpression{manifold_dim, form_rank_2, expression_rank_2, G}, element_id::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank_2, expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{manifold_dim}}
    # Evaluate the two forms that make up the wedge product 
    form_expression_1_eval, form_expression_1_indices = evaluate(form_expression_1, element_id, xi)  # 0-form 
    form_expression_2_eval, form_expression_2_indices = evaluate(form_expression_2, element_id, xi)  # k-form

    # Compute the wedge product between the 0-form and the 0-form
    
    # Given 
    #   α⁰ := α 
    #   βᵏ := ∑ᵢβᵢ dᵏξᵢ (note that here we use dᵏξᵢ to mean dξᵢ for 1-forms, d²ξ₁ = dξ₂dξ₃, etc for 2-forms, and so forth) 
    # then 
    #   α⁰ ∧ βᵏ = ∑ᵢ α βᵢ dᵏξᵢ
    
    # Get the number of components of the wedge 
    # In this case because it is the wedge of a 0-form with a k-form 
    # there will be has many components in the wedge as components in the k-form
    n_components_form_expression_2 = binomial(manifold_dim, form_rank_2)
    n_components_wedge = n_components_form_expression_2

    # Get the number of basis in each expression 
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points 
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    wedge_eval = [zeros(Float64, n_evaluation_points, num_basis_form_expression_1..., num_basis_form_expression_2...) for _ ∈ 1:n_components_form_expression_2]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    for wedge_component_idx ∈ 1:n_components_form_expression_2
        for form_expression_1_basis_idx ∈ CartesianIndices(Tuple(num_basis_form_expression_1))
            for form_expression_2_basis_idx ∈ CartesianIndices(Tuple(num_basis_form_expression_2))
                wedge_eval[wedge_component_idx][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .= @views form_expression_1_eval[1][:, form_expression_1_basis_idx] .* form_expression_2_eval[wedge_component_idx][:, form_expression_2_basis_idx]
            end
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D
function _evaluate_wedge(form_expression_1::AbstractFormExpression{2, 1, expression_rank_1, G}, form_expression_2::AbstractFormExpression{2, 1, expression_rank_2, G}, element_id::Int, xi::NTuple{2, Vector{Float64}}) where {expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{2}}
    # Evaluate the two forms that make up the wedge product 
    form_expression_1_eval, form_expression_1_indices = evaluate(form_expression_1, element_id, xi)  # 1-form 
    form_expression_2_eval, form_expression_2_indices = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D
    
    # Given 
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2], then
    #   α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression 
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points 
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 2

    # Allocate memory space for wedge_eval
    wedge_eval = [zeros(Float64, n_evaluation_points, num_basis_form_expression_1..., num_basis_form_expression_2...) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    wedge_component = 1 # There is only 1 because it is a 2-form in 2D
    for form_expression_1_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_1))
        for form_expression_2_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_2))
            wedge_eval[wedge_component][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .= @views (form_expression_1_eval[1][:, form_expression_1_basis_idx] .* form_expression_2_eval[2][:, form_expression_2_basis_idx]) .- (form_expression_1_eval[2][:, form_expression_1_basis_idx] .* form_expression_2_eval[1][:, form_expression_2_basis_idx])
        end
    end
    
    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D
function _evaluate_wedge(form_expression_1::AbstractFormExpression{3, 1, expression_rank_1, G}, form_expression_2::AbstractFormExpression{3, 1, expression_rank_2, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product 
    form_expression_1_eval, form_expression_1_indices = evaluate(form_expression_1, element_id, xi)  # 1-form 
    form_expression_2_eval, form_expression_2_indices = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D
    
    # Given 
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ²
    
    # Get the number of basis in each expression 
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points 
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 3

    # Allocate memory space for wedge_eval
    wedge_eval = [zeros(Float64, n_evaluation_points, num_basis_form_expression_1..., num_basis_form_expression_2...) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for form_expression_1_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_1))
        for form_expression_2_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_2))
            wedge_eval[1][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .= @views (form_expression_1_eval[2][:, form_expression_1_basis_idx] .* form_expression_2_eval[3][:, form_expression_2_basis_idx]) .- (form_expression_1_eval[3][:, form_expression_1_basis_idx] .* form_expression_2_eval[2][:, form_expression_2_basis_idx])
        end
    end
    
    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for form_expression_1_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_1))
        for form_expression_2_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_2))
            wedge_eval[2][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .= @views (form_expression_1_eval[3][:, form_expression_1_basis_idx] .* form_expression_2_eval[1][:, form_expression_2_basis_idx]) .- (form_expression_1_eval[1][:, form_expression_1_basis_idx] .* form_expression_2_eval[3][:, form_expression_2_basis_idx])
        end
    end
    
    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for form_expression_1_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_1))
        for form_expression_2_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_2))
            wedge_eval[3][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .= @views (form_expression_1_eval[1][:, form_expression_1_basis_idx] .* form_expression_2_eval[2][:, form_expression_2_basis_idx]) .- (form_expression_1_eval[2][:, form_expression_1_basis_idx] .* form_expression_2_eval[1][:, form_expression_2_basis_idx])
        end
    end
    
    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D
function _evaluate_wedge(form_expression_1::AbstractFormExpression{3, 1, expression_rank_1, G}, form_expression_2::AbstractFormExpression{3, 2, expression_rank_2, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product 
    form_expression_1_eval, form_expression_1_indices = evaluate(form_expression_1, element_id, xi)  # 1-form 
    form_expression_2_eval, form_expression_2_indices = evaluate(form_expression_2, element_id, xi)  # 2-form

    # Compute the wedge product between the 1-form and the 2-form in 3D
    
    # Given 
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β² := β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ²
    # with i = [1, 2, 3], then
    #   α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    
    # Get the number of basis in each expression 
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points 
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 1-form wedge with a 2-form, therefore the 
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    wedge_eval = [zeros(Float64, n_evaluation_points, num_basis_form_expression_1..., num_basis_form_expression_2...) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_expression_1_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_1))
        for form_expression_2_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_2))
            for form_components_idx in 1:3
                wedge_eval[1][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .+= @views (form_expression_1_eval[form_components_idx][:, form_expression_1_basis_idx] .* form_expression_2_eval[form_components_idx][:, form_expression_2_basis_idx])
            end
        end
    end
    
    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D
function _evaluate_wedge(form_expression_1::AbstractFormExpression{3, 2, expression_rank_1, G}, form_expression_2::AbstractFormExpression{3, 1, expression_rank_2, G}, element_id::Int, xi::NTuple{3, Vector{Float64}}) where {expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product 
    form_expression_1_eval, form_expression_1_indices = evaluate(form_expression_1, element_id, xi)  # 2-form 
    form_expression_2_eval, form_expression_2_indices = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 2-form and the 1-form in 3D
    
    # Given 
    #   α² := α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ²
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    
    # Get the number of basis in each expression 
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points 
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 2-form wedge with a 1-form, therefore the 
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    wedge_eval = [zeros(Float64, n_evaluation_points, num_basis_form_expression_1..., num_basis_form_expression_2...) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_expression_1_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_1))
        for form_expression_2_basis_idx in CartesianIndices(Tuple(num_basis_form_expression_2))
            for form_components_idx in 1:3
                wedge_eval[1][:, form_expression_1_basis_idx, form_expression_2_basis_idx] .+= @views (form_expression_1_eval[form_components_idx][:, form_expression_1_basis_idx] .* form_expression_2_eval[form_components_idx][:, form_expression_2_basis_idx])
            end
        end
    end
    
    return wedge_eval, wedge_indices
end