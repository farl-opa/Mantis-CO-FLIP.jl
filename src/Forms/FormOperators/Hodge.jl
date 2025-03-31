############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    HodgeStar{manifold_dim, form_rank, expression_rank, G, F} <:
    AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}

Represents the hodge star of an `AbstractFormExpression`.

# Fields
- `form::AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}`: The form to
    which the hodge star is applied.
- `label::String`: The hodge star label. This is a concatenation of `"★"` with the
    label of `form`.

# Type parameters
- `manifold_dim`: Dimension of the manifold. 
- `form_rank`: The form rank of the hodge star. If the form rank of `form` is `k`
    then `form_rank` is `manifold_dim - k`.
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0,
    with one single set of basis forms have rank 1, with two sets of basis forms have rank
    2. Higher ranks are not possible.
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the underlying geometry.
- `F <: Forms.AbstractFormExpression{manifold_dim, manifold_dim-form_rank, expression_rank,
    G}`: The type of `form`.

# Inner Constructors
- `HodgeStar(form::F)`: General constructor.
"""
struct HodgeStar{manifold_dim, form_rank, expression_rank, G, F} <:
       AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
    form::F
    label::String

    function HodgeStar(
        form::F
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    }
        if expression_rank > 1
            msg_1 = "Hodge-star only valid for expressions with expression rank < 2. "
            msg_2 = "Expression rank $(expression_rank) was given."
            throw(ArgumentError(msg_1 * msg_2))
        end
        hodge_star_rank = manifold_dim - form_rank

        return new{manifold_dim, hodge_star_rank, expression_rank, G, F}(
            form, "★" * get_label(form)
        )
    end
end

"""
    get_form(hodge_star::HodgeStar)

Returns the form to which the hodge star is applied.

# Arguments
- `hodge_star::HodgeStar`: The hodge star structure.

# Returns
- `<:AbstractFormExpression`: The form to which the hodge star is applied.
"""
get_form(hodge_star::HodgeStar) = hodge_star.form

"""
    get_geometry(hodge_star::HodgeStar)

Returns the geometry of the form associated with the hodge star.

# Arguments
- `hodge_star::HodgeStar`: The hodge star structure.

# Returns
- `<:Geometry.AbstractGeometry`: The geometry of the form.
"""
get_geometry(hodge_star::HodgeStar) = get_geometry(get_form(hodge_star))

"""
    evaluate(
        hodge_star::HodgeStar{manifold_dim},
        element_id::Int,
        xi::NTuple{manifold_dim, Vector{Float64}},
    ) where {manifold_dim}

Computes the hodge star at the element given by `element_id`, and canonical points `xi`.

# Arguments
- `hodge_star::HodgeStar{manifold_dim}`: The hodge star structure.
- `element_id::Int`: The element identifier.
- `xi::NTuple{manifold_dim, Vector{Float64}`: The set of canonical points.

# Returns
- `::Vector{Array{Float64, expression_rank + 1}}`: The evaluated hodge star. The number of
    entries in the `Vector` is `binomial(manifold_dim, manifold_dim - form_rank)`. The size
    of the `Array` is `(num_eval_points, num_basis)`, where `num_eval_points =
    prod(length.(xi))` and `num_basis` is the number of basis functions used to represent
    the `form` on `element_id` ― for `expression_rank = 0` the inner `Array` is equivalent
    to a `Vector`.
"""
function evaluate(
    hodge_star::HodgeStar{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim}
    return _evaluate_hodge_star(get_form(hodge_star), element_id, xi)
end

############################################################################################
#                                     Abstract method                                      #
############################################################################################

function _evaluate_hodge_star(
    form::AbstractFormExpression{manifold_dim, form_rank, expression_rank, G},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry}
    throw(ArgumentError("Method not implement for type $(typeof(form))."))
end

############################################################################################
#                                       Combinations                                       #
############################################################################################

# 0-forms (manifold_dim)
function _evaluate_hodge_star(
    form_expression::AbstractFormExpression{manifold_dim, 0, expression_rank, G},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    _, sqrt_g = Geometry.metric(form_expression.geometry, element_id, xi)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)
    # We restrict the evaluation of hodge-star to expression of rank 1 and lower, therefore
    # we can select just the first index of the vector of indices, for higher rank
    # expression we will need to change this
    n_indices = length(form_indices[1])

    hodge_eval = Vector{Matrix{Float64}}(undef, 1)

    # ⋆α₁⁰ = α₁⁰√det(gᵢⱼ) dξ₁∧…∧dξₙ.
    hodge_eval[1] = reshape(form_eval[1] .* sqrt_g, (:, n_indices)) # TODO: is this reshape needed?

    return hodge_eval, form_indices
end

# n-forms (manifold_dim)
function _evaluate_hodge_star(
    form_expression::AbstractFormExpression{manifold_dim, manifold_dim, expression_rank, G},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
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
function _evaluate_hodge_star(
    form_expression::AbstractFormExpression{2, 1, expression_rank, G},
    element_id::Int,
    xi::NTuple{2, Vector{Float64}},
) where {expression_rank, G <: Geometry.AbstractGeometry{2}}
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
function _evaluate_hodge_star(
    form_expression::AbstractFormExpression{3, 1, expression_rank, G},
    element_id::Int,
    xi::NTuple{3, Vector{Float64}},
) where {expression_rank, G <: Geometry.AbstractGeometry{3}}
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
function _evaluate_hodge_star(
    form_expression::AbstractFormExpression{3, 2, expression_rank, G},
    element_id::Int,
    xi::NTuple{3, Vector{Float64}},
) where {expression_rank, G <: Geometry.AbstractGeometry{3}}
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
