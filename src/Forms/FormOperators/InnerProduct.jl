############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    InnerProduct{F1, F2, G}

Represents the inner product between two differential forms.

# Fields
- `form_1 <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}`: The first
    form.
- `form_2 <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}`: The second
    form.

# Type parameters
- `F1 <: Forms.AbstractFormExpression`: The type of `form_1`.
- `F2 <: Forms.AbstractFormExpression`: The type of `form_2`.
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the geometry where the forms are
    defined.

# Inner Constructors
- `InnerProduct(form_1::F1, form_2::F2)`: General constructor.
"""
struct InnerProduct{F1, F2, G}
    form_1::F1
    form_2::F2

    function InnerProduct(
        form_1::F1,
        form_2::F2,
    ) where {
        manifold_dim,
        form_rank_1,
        form_rank_2,
        expression_rank_1,
        expression_rank_2,
        G,
        F1 <: AbstractFormExpression{manifold_dim, form_rank_1, expression_rank_1, G},
        F2 <: AbstractFormExpression{manifold_dim, form_rank_2, expression_rank_2, G},
    }
        if expression_rank_1 > 1
            throw(ArgumentError(
                "Inner product only valid between expressions with expression rank < 2, got:
                $expression_rank_1",
            ))
        end
        if expression_rank_2 > 1
            throw(ArgumentError(
                "Inner product only valid between expressions with expression rank < 2, got:
                $expression_rank_2",
            ))
        end

        return new{F1, F2, G}(form_1, form_2)
    end

    function Base.:*(form_1::AbstractFormExpression, form_2::AbstractFormExpression)
        return InnerProduct(form_1, form_2)
    end
end

"""
    get_forms(inner_product::InnerProduct)

Returns the forms to which the inner product is applied.

# Arguments
- `inner_product::InnerProduct`: The inner product structure.

# Returns
- `<:AbstractFormExpression`: The first form to which the inner product is applied.
- `<:AbstractFormExpression`: The second form to which the inner product is applied.
"""
function get_forms(inner_product::InnerProduct)
    return inner_product.form_1, inner_product.form_2
end

"""
    get_geometry(inner_product::InnerProduct)

Returns the geometry of the forms associated with the inner product.

# Arguments
- `inner_product::InnerProduct`: The inner product structure.

# Returns
- `<:Geometry.AbstractGeometry`: The geometry of the forms.
"""
get_geometry(inner_product::InnerProduct) = get_geometry(get_forms(inner_product))

"""
    evaluate(
        inner_product::InnerProduct{F1, F2, G},
        element_id::Int,
        quad_rule::Quadrature.AbstractElementQuadratureRule{manifold_dim},
    ) where {manifold_dim, F1, F2, G <: Geometry.AbstractGeometry{manifold_dim}}

Computes the inner product at the element given by `element_id`, and canonical points given
by `quad_rule`.

# Arguments
- `inner_product::InnerProduct`: The inner product structure.
- `element_id::Int`: The element identifier.
- `quad_rule::Quadrature.AbstractElementQuadratureRule{manifold_dim}`: The quadrature rule used for
    integration.

# Returns
- `prod_form_rows::Vector{Int}`: Indices of the first form for each index of the
    linear indexing `l(i,j) = i + (j-1)*n_indices_1` where `n_indices_1` is the number of
    basis functions in the first form expression and `i` and `j` are the local basis indices
    of the first and second expression, respectively. I.e., `prod_form_rows[l(i,j)]` stores
    the global basis index of the local basis `i` when integrated against local basis `j`.
- `prod_form_cols::Vector{Int}`: Indices of the second form for each index of the
    linear indexing `l(i,j) = i + (j-1)*n_indices_1` where `n_indices_1` is the number of
    basis functions in the first form expression and `i` and `j` are the local basis indices
    of the first and second expression, respectively. I.e., `prod_form_cols[l(i,j)]` stores
    the global basis index of the local basis `j` when integrated against local basis `i`.
- `prod_form_eval::Vector{Float64}`: Evaluation of the inner product for each index of the
    linear indexing `l(i,j) = i + (j-1)*n_indices_1` where `n_indices_1` is the number of
    basis functions in the first form expression and `i` and `j` are the local basis indices
    of the first and second expression, respectively. I.e., `prod_form_eval[l(i,j)]` stores
    the result of local basis `i` integrated against local basis `j`.
- `::Vector{Array{Float64, expression_rank + 1}}`: The evaluated inner product. The number of
    entries in the `Vector` is `binomial(manifold_dim, manifold_dim - form_rank)`. The size
    of the `Array` is `(num_eval_points, num_basis)`, where `num_eval_points =
    prod(length.(xi))` and `num_basis` is the number of basis functions used to represent
    the `form` on `element_id` ― for `expression_rank = 0` the inner `Array` is equivalent
    to a `Vector`.
"""
function evaluate(
    inner_product::InnerProduct{F1, F2, G},
    element_id::Int,
    quad_rule::Quadrature.AbstractElementQuadratureRule{manifold_dim},
) where {manifold_dim, F1, F2, G <: Geometry.AbstractGeometry{manifold_dim}}
    forms = get_forms(inner_product)

    return _evaluate_inner_product(forms[1], forms[2], element_id, quad_rule)
end
############################################################################################
#                                     Abstract method                                      #
############################################################################################

function _evaluate_inner_product(
    form_1::AbstractFormExpression{manifold_dim, form_rank_1, expression_rank_1, G},
    form_2::AbstractFormExpression{manifold_dim, form_rank_2, expression_rank_2, G},
    element_id::Int,
    quad_rule::Quadrature.AbstractElementQuadratureRule{manifold_dim},
) where {
    manifold_dim,
    form_rank_1,
    form_rank_2,
    expression_rank_1,
    expression_rank_2,
    G <: Geometry.AbstractGeometry{manifold_dim},
}
    throw(ArgumentError(
        "Method not implemented for types $(typeof(form_1)) and $(typeof(form_2))."
    ))
end

############################################################################################
#                                       Combinations                                       #
############################################################################################

# (0-forms, 0-forms)
function _evaluate_inner_product(
    form_expression1::AbstractFormExpression{manifold_dim, 0, expression_rank_1, G},
    form_expression2::AbstractFormExpression{manifold_dim, 0, expression_rank_2, G},
    element_id::Int,
    quad_rule::Quadrature.AbstractElementQuadratureRule{manifold_dim},
) where {
    manifold_dim,
    expression_rank_1,
    expression_rank_2,
    G <: Geometry.AbstractGeometry{manifold_dim},
}
    _, sqrt_g = Geometry.metric(
        get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes
    )

    # evaluate both the form expressions; it is assumed that the output is of the following
    # form:
    # form_eval::Vector{Matrix{Float64}} of length 1, where each matrix is of size (number
    # of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)

    # Get dimension of each form expression
    # Since we restrict to expresions with expression rank 0 or 1, we only have one set of
    # indices
    # If we wish to have inner product between expressions with higher ranks, this needs to
    # be changed
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    # Form 1: α⁰ = α⁰₁
    # Form 2: β⁰ = β⁰₁
    # ⟨α⁰, β⁰⟩ = ∫α⁰₁β⁰₁ⱼ√det(g)dξ¹∧…∧dξⁿ

    prod_form_rows = Vector{Int}(undef, n_indices_1 * n_indices_2)
    prod_form_cols = Vector{Int}(undef, n_indices_1 * n_indices_2)
    prod_form_eval = zeros(n_indices_1 * n_indices_2)

    prod_form_rows, prod_form_cols, prod_form_eval = _inner_product_0_form_component!(
        prod_form_rows,
        prod_form_cols,
        prod_form_eval,
        quad_rule,
        sqrt_g,
        form1_eval,
        form1_indices[1],
        form2_eval,
        form2_indices[1],
        n_indices_1,
        n_indices_2,
    ) # Evaluates α⁰₁β⁰₁√det(g)

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function _inner_product_0_form_component!(
    prod_form_rows::Vector{Int},
    prod_form_cols::Vector{Int},
    prod_form_eval::Vector{Float64},
    quad_rule,
    sqrt_g,
    form1_eval,
    form1_indices,
    form2_eval,
    form2_indices,
    n_indices_1,
    n_indices_2,
)
    quad_weights = Quadrature.get_weights(quad_rule)

    for j in 1:n_indices_2
        for i in 1:n_indices_1
            linear_idx = i + (j - 1) * n_indices_1

            prod_form_rows[linear_idx] = form1_indices[i]
            prod_form_cols[linear_idx] = form2_indices[j]

            for k in eachindex(quad_weights)
                prod_form_eval[linear_idx] +=
                    quad_weights[k] * form1_eval[1][k, i] * form2_eval[1][k, j] * sqrt_g[k]
            end
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# (n-forms, n-forms)
function _evaluate_inner_product(
    form_expression1::AbstractFormExpression{
        manifold_dim, manifold_dim, expression_rank_1, G
    },
    form_expression2::AbstractFormExpression{
        manifold_dim, manifold_dim, expression_rank_2, G
    },
    element_id::Int,
    quad_rule::Quadrature.AbstractElementQuadratureRule{manifold_dim},
) where {
    manifold_dim,
    expression_rank_1,
    expression_rank_2,
    G <: Geometry.AbstractGeometry{manifold_dim},
}
    _, sqrt_g = Geometry.metric(
        get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes
    )

    # form_eval::Vector{Matrix{Float64}} of length 1, where each matrix is of size (number
    # of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)

    # Get dimension of each form expression
    # Since we restrict to expresions with expression rank 0 or 1, we only have one set of
    # indices
    # If we wish to have inner product between expressions with higher ranks, this needs to
    # be changed
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    # Form 1: αⁿ = αⁿ₁dξ¹∧…∧dξⁿ
    # Form 2: βⁿ = βⁿ₁dξ¹∧…∧dξⁿ
    # ⟨αⁿ, βⁿ⟩ = ∫αⁿ₁βⁿ₁ⱼ√det(g)⁻¹dξ¹∧…∧dξⁿ

    prod_form_rows = Vector{Int}(undef, n_indices_1 * n_indices_2)
    prod_form_cols = Vector{Int}(undef, n_indices_1 * n_indices_2)
    prod_form_eval = zeros(n_indices_1 * n_indices_2)

    prod_form_rows, prod_form_cols, prod_form_eval = _inner_product_n_form_component!(
        prod_form_rows,
        prod_form_cols,
        prod_form_eval,
        quad_rule,
        sqrt_g,
        form1_eval,
        form1_indices[1],
        form2_eval,
        form2_indices[1],
        n_indices_1,
        n_indices_2,
    ) # Evaluates αⁿ₁βⁿ₁ⱼ√det(g)⁻¹

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function _inner_product_n_form_component!(
    prod_form_rows::Vector{Int},
    prod_form_cols::Vector{Int},
    prod_form_eval::Vector{Float64},
    quad_rule,
    sqrt_g,
    form1_eval,
    form1_indices,
    form2_eval,
    form2_indices,
    n_indices_1,
    n_indices_2,
)
    quad_weights = Quadrature.get_weights(quad_rule)

    for j in 1:n_indices_2
        for i in 1:n_indices_1
            linear_idx = i + (j - 1) * n_indices_1

            prod_form_rows[linear_idx] = form1_indices[i]
            prod_form_cols[linear_idx] = form2_indices[j]

            for k in eachindex(quad_weights)
                prod_form_eval[linear_idx] +=
                    quad_weights[k] * form1_eval[1][k, i] * form2_eval[1][k, j] / sqrt_g[k]
            end
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# (1-forms, 1-forms)
function _evaluate_inner_product(
    form_expression1::AbstractFormExpression{manifold_dim, 1, expression_rank_1, G},
    form_expression2::AbstractFormExpression{manifold_dim, 1, expression_rank_2, G},
    element_id::Int,
    quad_rule::Quadrature.AbstractElementQuadratureRule{manifold_dim},
) where {
    manifold_dim,
    expression_rank_1,
    expression_rank_2,
    G <: Geometry.AbstractGeometry{manifold_dim},
}
    inv_g, _, sqrt_g = Geometry.inv_metric(
        get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes
    )

    # form_eval::Vector{Matrix{Float64}} of length manifold_dim, where each matrix is of
    # size (number of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)

    # Get dimension of each form expression
    # Since we restrict to expresions with expression rank 0 or 1, we only have one set of
    # indices
    # If we wish to have inner product between expressions with higher ranks, this needs to
    # be changed
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    n_prod_indices = n_indices_1 * n_indices_2

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

    for i in 1:manifold_dim
        for j in 1:manifold_dim
            prod_form_rows, prod_form_cols, prod_form_eval =
            _inner_product_1_form_component!(
                prod_form_rows,
                prod_form_cols,
                prod_form_eval,
                quad_rule,
                inv_g,
                sqrt_g,
                form1_eval,
                form1_indices[1],
                form2_eval,
                form2_indices[1],
                n_indices_1,
                n_indices_2,
                (i, j),
            ) # Evaluates α¹ᵢβ¹ⱼgⁱʲ√det(g)
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function _inner_product_1_form_component!(
    prod_form_rows::Vector{Int},
    prod_form_cols::Vector{Int},
    prod_form_eval::Vector{Float64},
    quad_rule,
    inv_g,
    sqrt_g,
    form1_eval,
    form1_indices,
    form2_eval,
    form2_indices,
    n_indices_1,
    n_indices_2,
    form_idxs::NTuple{2, Int},
)
    quad_weights = Quadrature.get_weights(quad_rule)

    for j in 1:n_indices_2
        for i in 1:n_indices_1
            linear_idx = i + (j - 1) * n_indices_1

            prod_form_rows[linear_idx] = form1_indices[i]
            prod_form_cols[linear_idx] = form2_indices[j]

            for k in eachindex(quad_weights)
                prod_form_eval[linear_idx] +=
                    quad_weights[k] *
                    form1_eval[form_idxs[1]][k, i] *
                    form2_eval[form_idxs[2]][k, j] *
                    inv_g[k, form_idxs[1], form_idxs[2]] *
                    sqrt_g[k]
            end
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

# (2-forms, 2-forms) in 3D
function _evaluate_inner_product(
    form_expression1::AbstractFormExpression{3, 2, expression_rank_1, G},
    form_expression2::AbstractFormExpression{3, 2, expression_rank_2, G},
    element_id::Int,
    quad_rule::Quadrature.AbstractElementQuadratureRule{3},
) where {expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{3}}
    inv_g, _, sqrt_g = Geometry.inv_metric(
        get_geometry(form_expression1, form_expression2), element_id, quad_rule.nodes
    )

    # assumes that the output is of the following form:
    # form_eval::Vector{Matrix{Float64}} of length 3, where each matrix is of size (number
    # of evaluation points)x(dimension of form_expression on this element)
    # form_indices::Vector{Int} of length (dimension of form_expression on this element)
    form1_eval, form1_indices = evaluate(form_expression1, element_id, quad_rule.nodes)
    form2_eval, form2_indices = evaluate(form_expression2, element_id, quad_rule.nodes)

    # Get dimension of each form expression
    # Since we restrict to expresions with expression rank 0 or 1, we only have one set of
    # indices
    # If we wish to have inner product between expressions with higher ranks, this needs to
    # be changed
    n_indices_1 = length(form1_indices[1])
    n_indices_2 = length(form2_indices[1])

    n_prod_indices = n_indices_1 * n_indices_2

    # Form space 1: α² = α²₁dξ₂∧dξ₃ + α²₂dξ₃∧dξ₁ + α²₃dξ₁∧dξ₂
    # Form space 2: β² = β²₁dξ₂∧dξ₃ + β²₂dξ₃∧dξ₁ + β²₃dξ₁∧dξ₂
    # ⟨α², β²⟩ = ∫α²ᵢβ²ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)dξ¹∧dξ²∧dξ³

    prod_form_rows = zeros(Int, n_prod_indices)
    prod_form_cols = zeros(Int, n_prod_indices)
    prod_form_eval = zeros(Float64, n_prod_indices)

    for i in 1:3
        for j in 1:3
            prod_form_rows, prod_form_cols, prod_form_eval =
            _inner_product_2_form_component!(
                prod_form_rows,
                prod_form_cols,
                prod_form_eval,
                quad_rule,
                inv_g,
                sqrt_g,
                form1_eval,
                form1_indices[1],
                form2_eval,
                form2_indices[1],
                n_indices_1,
                n_indices_2,
                (i, j),
            ) # Evaluates α¹ᵢβ¹ⱼ(gⁱ¹ʲ¹gⁱ²ʲ²-gⁱ¹ʲ²gⁱ²ʲ¹)√det(g)
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end

function _inner_product_2_form_component!(
    prod_form_rows::Vector{Int},
    prod_form_cols::Vector{Int},
    prod_form_eval::Vector{Float64},
    quad_rule,
    inv_g,
    sqrt_g,
    form1_eval,
    form1_indices,
    form2_eval,
    form2_indices,
    n_indices_1,
    n_indices_2,
    form_idxs::NTuple{2, Int},
)
    quad_weights = Quadrature.get_weights(quad_rule)

    inv_indices_1 = (mod(form_idxs[1], 3) + 1, mod(form_idxs[1] + 1, 3) + 1)
    inv_indices_2 = (mod(form_idxs[2], 3) + 1, mod(form_idxs[2] + 1, 3) + 1)

    inv_g_factor =
        inv_g[:, inv_indices_1[1], inv_indices_2[1]] .*
        inv_g[:, inv_indices_1[2], inv_indices_2[2]] .-
        inv_g[:, inv_indices_1[1], inv_indices_2[2]] .*
        inv_g[:, inv_indices_1[2], inv_indices_2[1]]

    for j in 1:n_indices_2
        for i in 1:n_indices_1
            linear_idx = i + (j - 1) * n_indices_1

            prod_form_rows[linear_idx] = form1_indices[i]
            prod_form_cols[linear_idx] = form2_indices[j]

            for k in eachindex(quad_weights)
                prod_form_eval[linear_idx] +=
                    quad_rule.weights[k] *
                    form1_eval[form_idxs[1]][k, i] *
                    form2_eval[form_idxs[2]][k, j] *
                    inv_g_factor[k] *
                    sqrt_g[k]
            end
        end
    end

    return prod_form_rows, prod_form_cols, prod_form_eval
end
