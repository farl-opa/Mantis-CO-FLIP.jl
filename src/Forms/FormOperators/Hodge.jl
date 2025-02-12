############################################################################################
#                                 Hodge star operator ★                                    #
############################################################################################

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
function evaluate_hodge_star(
    form_expression::AbstractFormExpression{manifold_dim,0,expression_rank,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim,expression_rank,G<:Geometry.AbstractGeometry{manifold_dim}}
    if expression_rank > 1
        throw(
            ArgumentError(
                "Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank",
            ),
        )
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
function evaluate_hodge_star(
    form_expression::AbstractFormExpression{manifold_dim,manifold_dim,expression_rank,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim,expression_rank,G<:Geometry.AbstractGeometry{manifold_dim}}
    if expression_rank > 1
        throw(
            ArgumentError(
                "Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank",
            ),
        )
    end

    _, sqrt_g = Geometry.metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # Because we restrict ourselves to expression with rank 0 or 1, we can
    # extract the first set of indices (only set), for higher ranks, we need
    # to change this
    n_indices = length(form_indices[1])

    hodge_eval = Vector{Matrix{Float64}}(undef, 1)

    # ⋆α₁ⁿdξ₁∧…∧dξₙ = α₁ⁿ(√det(gᵢⱼ))⁻¹.
    hodge_eval[1] = reshape(form_eval[1] .* (sqrt_g .^ (-1)), (:, n_indices)) # TODO: is this reshape needed?

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
function evaluate_hodge_star(
    form_expression::AbstractFormExpression{2,1,expression_rank,G},
    element_id::Int,
    xi::NTuple{2,Vector{Float64}},
) where {expression_rank,G<:Geometry.AbstractGeometry{2}}
    if expression_rank > 1
        throw(
            ArgumentError(
                "Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank",
            ),
        )
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
function evaluate_hodge_star(
    form_expression::AbstractFormExpression{3,1,expression_rank,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank,G<:Geometry.AbstractGeometry{3}}
    if expression_rank > 1
        throw(
            ArgumentError(
                "Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank",
            ),
        )
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
            hodge_eval[k_hodge_component] .+= @views form_eval[k_expression_component] .*
                inv_g[
                :, k_hodge_component, k_expression_component
            ]
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
function evaluate_hodge_star(
    form_expression::AbstractFormExpression{3,2,expression_rank,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank,G<:Geometry.AbstractGeometry{3}}
    if expression_rank > 1
        throw(
            ArgumentError(
                "Hodge-star only valid for expressions with expression rank < 2, got: $expression_rank",
            ),
        )
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
            hodge_eval[k_hodge_component] .+= @views form_eval[k_expression_component] .* g[
                :, k_hodge_component, k_expression_component
            ]
        end

        # Divide by √det(gᵢⱼ) to get (α₁gₖ₁+α₂gₖ₂+α₃gₖ₃) / √det(gᵢⱼ)
        hodge_eval[k_hodge_component] ./= sqrt_g
    end

    return hodge_eval, form_indices
end
