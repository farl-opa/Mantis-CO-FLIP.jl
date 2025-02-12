
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
function evaluate_sharp(
    form_expression::AbstractFormExpression{manifold_dim,1,expression_rank,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim,expression_rank,G<:Geometry.AbstractGeometry{manifold_dim}}
    manifold_dim >= 2 || throw(
        ArgumentError(
            "Manifold dimension should be 2 or 3 for 1-forms. Dimension $manifold_dim was given.",
        ),
    )

    inv_g, _, _ = Geometry.inv_metric(form_expression.geometry, element_id, xi)

    num_form_components = manifold_dim # = binomial(manifold_dim, 1)

    form_eval, form_indices = evaluate(form_expression, element_id, xi)

    sharp_eval = Vector{Matrix{Float64}}(undef, num_form_components)

    # ♯: dξⁱ ↦ ♯(dξⁱ) = gⁱʲ∂ⱼ
    for component in 1:num_form_components
        sharp_eval[component] = @views hcat(
            [form_eval[i] .* inv_g[:, i, component] for i in 1:num_form_components]...
        )
    end

    return sharp_eval, form_indices
end
