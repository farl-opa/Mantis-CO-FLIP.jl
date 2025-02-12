
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
- `element_idx::Int`: Index of the element to evaluate.
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
function evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,form_rank_1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{manifold_dim,form_rank_2,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {
    manifold_dim,
    form_rank_1,
    form_rank_2,
    expression_rank_1,
    expression_rank_2,
    G<:Geometry.AbstractGeometry{manifold_dim},
}
    # Check if total (expression) rank of wedge expression is valid
    if (expression_rank_1 + expression_rank_2) > 2
        throw(
            ArgumentError(
                "Wedge of FormExpressions requires expressions with total expression rank smaller than three, got: $expression_rank_1 and $expression_rank_2",
            ),
        )
    end

    # Check if total (form) rank of wedge expression is valid
    if (form_rank_1 + form_rank_2) > manifold_dim
        throw(
            ArgumentError(
                "The wedge product form rank (got $(form_rank_1 + form_rank_2)) cannot exceed the manifold dimension (got $manifold_dim).",
            ),
        )
    end

    return _evaluate_wedge(form_expression_1, form_expression_2, element_id, xi)
end

# 0-forms ∧ 0-forms (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,0,0,G},
    form_expression_2::AbstractFormExpression{manifold_dim,0,0,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim,G<:Geometry.AbstractGeometry{manifold_dim}}

    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 0-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   β⁰ := β
    # then
    #   α⁰ ∧ β⁰ = (αβ)⁰

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a 0-form
    # therefore there is only one component
    # n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only one component because both expressions are 0-forms.
    # Only one column because both expressions have expresion rank 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1)]

    # Assign wedge_indices: only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    for cart_ind in CartesianIndices(wedge_eval[1])
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ 0-forms (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,0,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{manifold_dim,0,0,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim,expression_rank_1,G<:Geometry.AbstractGeometry{manifold_dim}}

    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 0-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   β⁰ := β
    # then
    #   α⁰ ∧ β⁰ = (αβ)⁰

    # Get the number of basis for the first expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a 0-form
    # therefore there is only one component
    # n_components_wedge = 1

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only one component because both expressions are 0-forms.
    # The second expression has expresion rank 0 so the size is the same as the first
    wedge_eval = [
        Array{Float64,1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = form_expression_1_indices # only the first expression has indices because the second expression has expression rank 0

    # Compute the wedge evaluation
    # Only need to loop ofter the basis of the first expression because the second
    # expression has expression rank 0
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][point, basis_id] = form_expression_1_eval[1][point, basis_id] *
            form_expression_2_eval[1][point, 1]
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ 0-forms (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,0,0,G},
    form_expression_2::AbstractFormExpression{manifold_dim,0,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim,expression_rank_2,G<:Geometry.AbstractGeometry{manifold_dim}}

    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 0-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 0-form

    # Compute the wedge product between the 0-form and the 0-form

    # Given
    #   α⁰ := α
    #   β⁰ := β
    # then
    #   α⁰ ∧ β⁰ = (αβ)⁰

    # Get the number of basis for the second expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of components of the wedge
    # In this case because it is the wedge of a 0-form with a 0-form
    # therefore there is only one component
    # n_components_wedge = 1

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only one component because both expressions are 0-forms.
    # The second expression has expresion rank 0 so the size is the same as the first
    wedge_eval = [
        Array{Float64,1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = form_expression_2_indices # only the first expression has indices because the second expression has expression rank 0

    # Compute the wedge evaluation
    # Only need to loop ofter the basis of the first expression because the second
    # expression has expression rank 0
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][point, basis_id] = form_expression_1_eval[1][point, 1] *
            form_expression_2_eval[1][point, basis_id]
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ 0-forms (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,0,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{manifold_dim,0,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {
    manifold_dim,
    expression_rank_1,
    expression_rank_2,
    G<:Geometry.AbstractGeometry{manifold_dim},
}

    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 0-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 0-form

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
    wedge_eval = [
        Array{Float64,1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] =
            form_expression_1_eval[1][point, basis_id_1] *
            form_expression_2_eval[1][point, basis_id_2]
    end

    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,form_rank_1,0,G},
    form_expression_2::AbstractFormExpression{manifold_dim,0,0,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim,form_rank_1,G<:Geometry.AbstractGeometry{manifold_dim}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # k-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 0-form

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

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge] # Only one column because both expressions have expresion rank 0

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each
    # expression
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[wedge_component_idx][cart_ind] *
                form_expression_2_eval[1][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,form_rank_1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{manifold_dim,0,0,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {
    manifold_dim,form_rank_1,expression_rank_1,G<:Geometry.AbstractGeometry{manifold_dim}
}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # k-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 0-form

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

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    # The second expression has expresion rank 0 so the size is the same as the first expression's evaluation
    wedge_eval = [
        Array{Float64,1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: only the first expression has indices because the second
    # expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[wedge_component_idx][cart_ind] *
                form_expression_2_eval[1][point, 1]
        end
    end

    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,form_rank_1,0,G},
    form_expression_2::AbstractFormExpression{manifold_dim,0,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {
    manifold_dim,form_rank_1,expression_rank_2,G<:Geometry.AbstractGeometry{manifold_dim}
}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # k-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 0-form

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
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    # The second first has expresion rank 0 so the size is the same as the second expression's evaluation
    wedge_eval = [
        Array{Float64,1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ) for _ in 1:n_components_wedge
    ]

    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] =
                form_expression_1_eval[wedge_component_idx][point, 1] *
                form_expression_2_eval[1][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# k-forms ∧ 0-forms (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,form_rank_1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{manifold_dim,0,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {
    manifold_dim,
    form_rank_1,
    expression_rank_1,
    expression_rank_2,
    G<:Geometry.AbstractGeometry{manifold_dim},
}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # k-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 0-form

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

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, both forms have arbitrary expression rank, hence the num_basis_from_expression_i, but only the first from has a variable form_rank
    wedge_eval = [
        Array{Float64,1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ) for _ in 1:n_components_wedge
    ]

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] = 
                form_expression_1_eval[wedge_component_idx][point, basis_id_1] *
                form_expression_2_eval[1][point, basis_id_2]
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,0,0,G},
    form_expression_2::AbstractFormExpression{manifold_dim,form_rank_2,0,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim,form_rank_2,G<:Geometry.AbstractGeometry{manifold_dim}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 0-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # k-form

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

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge]

    # Assign wedge_indices: only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            wedge_eval[wedge_component_idx][cart_ind] = 
                form_expression_1_eval[1][cart_ind] *
                form_expression_2_eval[wedge_component_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,0,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{manifold_dim,form_rank_2,0,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {
    manifold_dim,form_rank_2,expression_rank_1,G<:Geometry.AbstractGeometry{manifold_dim}
}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 0-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # k-form

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

    # Get the number of basis in the first expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1, but only the first from has a variable form_rank
    wedge_eval = [
        Array{Float64,1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] = 
                form_expression_1_eval[1][cart_ind] *
                form_expression_2_eval[wedge_component_idx][point, 1]
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,0,0,G},
    form_expression_2::AbstractFormExpression{manifold_dim,form_rank_2,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {
    manifold_dim,form_rank_2,expression_rank_2,G<:Geometry.AbstractGeometry{manifold_dim}
}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 0-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # k-form

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

    # Get the number of basis in the first expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # In this case, only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2, but only the first from has a variable form_rank
    wedge_eval = [
        Array{Float64,1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] = 
                form_expression_1_eval[1][point, 1] *
                form_expression_2_eval[wedge_component_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 0-forms ∧ k-forms (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{manifold_dim,0,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{manifold_dim,form_rank_2,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {
    manifold_dim,
    form_rank_2,
    expression_rank_1,
    expression_rank_2,
    G<:Geometry.AbstractGeometry{manifold_dim},
}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 0-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # k-form

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
    wedge_eval = [
        Array{Float64,1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    for wedge_component_idx in 1:n_components_wedge
        for cart_ind in CartesianIndices(wedge_eval[wedge_component_idx])
            (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
            wedge_eval[wedge_component_idx][cart_ind] = 
                form_expression_1_eval[1][point, basis_id_1] *
                form_expression_2_eval[wedge_component_idx][point, basis_id_2]
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{2,1,0,G},
    form_expression_2::AbstractFormExpression{2,1,0,G},
    element_id::Int,
    xi::NTuple{2,Vector{Float64}},
) where {G<:Geometry.AbstractGeometry{2}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2], then
    #   α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1)]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    # Only one component because it is a 2-form in 2D
    for cart_ind in CartesianIndices(wedge_eval[1])
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[2][cart_ind] -
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{2,1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{2,1,0,G},
    element_id::Int,
    xi::NTuple{2,Vector{Float64}},
) where {expression_rank_1,G<:Geometry.AbstractGeometry{2}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2], then
    #   α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1
    wedge_eval = [
        Array{Float64,1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    # Only one component because it is a 2-form in 2D
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[2][point, 1] -
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[1][point, 1]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{2,1,0,G},
    form_expression_2::AbstractFormExpression{2,1,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{2,Vector{Float64}},
) where {expression_rank_2,G<:Geometry.AbstractGeometry{2}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2], then
    #   α¹ ∧ β¹ = (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Allocate memory space for wedge_eval
    # Only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2
    wedge_eval = [
        Array{Float64,1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    # Only one component because it is a 2-form in 2D
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[1][point, 1] * form_expression_2_eval[2][cart_ind] -
            form_expression_1_eval[2][point, 1] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 2D (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{2,1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{2,1,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{2,Vector{Float64}},
) where {expression_rank_1,expression_rank_2,G<:Geometry.AbstractGeometry{2}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

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

    # Allocate memory space for wedge_eval
    wedge_eval = [
        Array{Float64,1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ),
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # Only one component because it is a 2-form in 2D
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[1][point, basis_id_1] *
                form_expression_2_eval[2][point, basis_id_2] -
            form_expression_1_eval[2][point, basis_id_1] *
                form_expression_2_eval[1][point, basis_id_2]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,1,0,G},
    form_expression_2::AbstractFormExpression{3,1,0,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 3

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for cart_ind in CartesianIndices(wedge_eval[1])
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[3][cart_ind] -
            form_expression_1_eval[3][cart_ind] * form_expression_2_eval[2][cart_ind]
    end

    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for cart_ind in CartesianIndices(wedge_eval[2])
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[3][cart_ind] * form_expression_2_eval[1][cart_ind] -
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[3][cart_ind]
    end

    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for cart_ind in CartesianIndices(wedge_eval[3])
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[2][cart_ind] -
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{3,1,0,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_1,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 3

    # Allocate memory space for wedge_eval
    # Only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1
    wedge_eval = [
        Array{Float64,1 + expression_rank_1}(
            undef, n_evaluation_points, num_basis_form_expression_1...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[3][point, 1] -
            form_expression_1_eval[3][cart_ind] * form_expression_2_eval[2][point, 1]
    end

    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for cart_ind in CartesianIndices(wedge_eval[2])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[3][cart_ind] * form_expression_2_eval[1][point, 1] -
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[3][point, 1]
    end

    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for cart_ind in CartesianIndices(wedge_eval[3])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[1][cart_ind] * form_expression_2_eval[2][point, 1] -
            form_expression_1_eval[2][cart_ind] * form_expression_2_eval[1][point, 1]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,1,0,G},
    form_expression_2::AbstractFormExpression{3,1,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_2,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 1-form and the 1-form in 2D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α¹ ∧ β¹ = (α₂β₃ - α₃β₂)dξ²∧dξ³ + (α₃β₁ - α₁β₃)dξ³∧dξ¹ + (α₁β₂ - α₂β₁)dξ¹∧dξ²

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have two 1-forms so the wedge is a 2-form
    n_components_wedge = 3

    # Allocate memory space for wedge_eval
    # Only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2
    wedge_eval = [
        Array{Float64,1 + expression_rank_2}(
            undef, n_evaluation_points, num_basis_form_expression_2...
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[2][point, 1] * form_expression_2_eval[3][cart_ind] -
            form_expression_1_eval[3][point, 1] * form_expression_2_eval[2][cart_ind]
    end

    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for cart_ind in CartesianIndices(wedge_eval[2])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[3][point, 1] * form_expression_2_eval[1][cart_ind] -
            form_expression_1_eval[1][point, 1] * form_expression_2_eval[3][cart_ind]
    end

    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for cart_ind in CartesianIndices(wedge_eval[3])
        (point, basis_id) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[1][point, 1] * form_expression_2_eval[2][cart_ind] -
            form_expression_1_eval[2][point, 1] * form_expression_2_eval[1][cart_ind]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 1-forms in 3D (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{3,1,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_1,expression_rank_2,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

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
    wedge_eval = [
        Array{Float64,1 + expression_rank_1 + expression_rank_2}(
            undef,
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        ) for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₂β₃ - α₃β₂)dξ²∧dξ³
    for cart_ind in CartesianIndices(wedge_eval[1])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[2][point, basis_id_1] * 
                form_expression_2_eval[3][point, basis_id_2] -
            form_expression_1_eval[3][point, basis_id_1] *
                form_expression_2_eval[2][point, basis_id_2]
    end

    # (α₃β₁ - α₁β₃)dξ³∧dξ¹
    for cart_ind in CartesianIndices(wedge_eval[2])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[3][point, basis_id_1] *
                form_expression_2_eval[1][point, basis_id_2] -
            form_expression_1_eval[1][point, basis_id_1] *
                form_expression_2_eval[3][point, basis_id_2]
    end

    # (α₁β₂ - α₂β₁)dξ¹∧dξ²
    for cart_ind in CartesianIndices(wedge_eval[3])
        (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
        wedge_eval[1][cart_ind] = 
            form_expression_1_eval[1][point, basis_id_1] *
                form_expression_2_eval[2][point, basis_id_2] -
            form_expression_1_eval[2][point, basis_id_1] *
                form_expression_2_eval[1][point, basis_id_2]
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,1,0,G},
    form_expression_2::AbstractFormExpression{3,2,0,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 2-form

    # Compute the wedge product between the 1-form and the 2-form in 3D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β² := β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ²
    # with i = [1, 2, 3], then
    #   α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 1-form wedge with a 2-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            wedge_eval[1][cart_ind] += 
                form_expression_1_eval[form_components_idx][cart_ind] *
                form_expression_2_eval[form_components_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{3,2,0,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_1,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 2-form

    # Compute the wedge product between the 1-form and the 2-form in 3D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β² := β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ²
    # with i = [1, 2, 3], then
    #   α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 1-form wedge with a 2-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1
    wedge_eval = [
        zeros(
            n_evaluation_points, num_basis_form_expression_1...
        )::Array{Float64,1 + expression_rank_1} for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] += 
                form_expression_1_eval[form_components_idx][cart_ind] *
                form_expression_2_eval[form_components_idx][point, 1]
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,1,0,G},
    form_expression_2::AbstractFormExpression{3,2,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_2,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 2-form

    # Compute the wedge product between the 1-form and the 2-form in 3D

    # Given
    #   α¹ := ∑ᵢ αᵢ dξⁱ
    #   β² := β₁dξ²∧dξ³ + β₂dξ³∧dξ¹ + β₃dξ¹∧dξ²
    # with i = [1, 2, 3], then
    #   α¹ ∧ β² = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 1-form wedge with a 2-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2
    wedge_eval = [
        zeros(
            n_evaluation_points, num_basis_form_expression_2...
        )::Array{Float64,1 + expression_rank_2} for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] += 
                form_expression_1_eval[form_components_idx][point, 1] *
                form_expression_2_eval[form_components_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 1-forms ∧ 2-forms in 3D (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,1,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{3,2,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_1,expression_rank_2,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 1-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 2-form

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
    wedge_eval = [
        zeros(
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        )::Array{Float64,1 + expression_rank_1 + expression_rank_2} for
        _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] += 
                form_expression_1_eval[form_components_idx][point, basis_id_1] *
                form_expression_2_eval[form_components_idx][point, basis_id_2]
        end
    end

    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D (expression_rank_1 = 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,2,0,G},
    form_expression_2::AbstractFormExpression{3,1,0,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 2-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 2-form and the 1-form in 3D

    # Given
    #   α² := α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ²
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 2-form wedge with a 1-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only one column because both expression ranks are 0
    wedge_eval = [zeros(Float64, n_evaluation_points, 1) for _ in 1:n_components_wedge]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only one index because both expressions have expresion rank 0
    wedge_indices = [[1]]

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            wedge_eval[1][cart_ind] += 
                form_expression_1_eval[form_components_idx][cart_ind] *
                form_expression_2_eval[form_components_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D (expression_rank_1 > 0 & expression_rank_2 = 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,2,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{3,1,0,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_1,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 2-form
    form_expression_2_eval, _ = evaluate(form_expression_2, element_id, xi)  # 1-form

    # Compute the wedge product between the 2-form and the 1-form in 3D

    # Given
    #   α² := α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ²
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_1 = size.(form_expression_1_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 2-form wedge with a 1-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only the first expression has arbitrary expression rank, hence the num_basis_from_expression_1
    wedge_eval = [
        zeros(
            n_evaluation_points, num_basis_form_expression_1...
        )::Array{Float64,1 + expression_rank_1} for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the first expression has indices because the second expression has expression rank 0
    wedge_indices = form_expression_1_indices

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] += 
                form_expression_1_eval[form_components_idx][cart_ind] *
                form_expression_2_eval[form_components_idx][point, 1]
        end
    end

    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D (expression_rank_1 = 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,2,0,G},
    form_expression_2::AbstractFormExpression{3,1,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_2,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, _ = evaluate(form_expression_1, element_id, xi)  # 2-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

    # Compute the wedge product between the 2-form and the 1-form in 3D

    # Given
    #   α² := α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ²
    #   β¹ := ∑ᵢ βᵢ dξⁱ
    # with i = [1, 2, 3], then
    #   α² ∧ β¹ = (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³

    # Get the number of basis in each expression
    num_basis_form_expression_2 = size.(form_expression_2_indices, 1)

    # Get the number of evaluation points
    n_evaluation_points = size(form_expression_1_eval[1], 1)  # all components are evaluated at the same points and both forms are evaluated at the same points

    # Get the number of components of the wedge
    # in this case we have a 2-form wedge with a 1-form, therefore the
    # results has only one component because it is a 3-form in 3D
    n_components_wedge = 1

    # Allocate memory space for wedge_eval
    # Only the second expression has arbitrary expression rank, hence the num_basis_from_expression_2
    wedge_eval = [
        zeros(
            n_evaluation_points, num_basis_form_expression_2...
        )::Array{Float64,1 + expression_rank_2} for _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    # Only the second expression has indices because the first expression has expression rank 0
    wedge_indices = form_expression_2_indices

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] += 
                form_expression_1_eval[form_components_idx][point, 1] *
                form_expression_2_eval[form_components_idx][cart_ind]
        end
    end

    return wedge_eval, wedge_indices
end

# 2-forms ∧ 1-forms in 3D (expression_rank_1 > 0 & expression_rank_2 > 0)
function _evaluate_wedge(
    form_expression_1::AbstractFormExpression{3,2,expression_rank_1,G},
    form_expression_2::AbstractFormExpression{3,1,expression_rank_2,G},
    element_id::Int,
    xi::NTuple{3,Vector{Float64}},
) where {expression_rank_1,expression_rank_2,G<:Geometry.AbstractGeometry{3}}
    # Evaluate the two forms that make up the wedge product
    form_expression_1_eval, form_expression_1_indices = evaluate(
        form_expression_1, element_id, xi
    )  # 2-form
    form_expression_2_eval, form_expression_2_indices = evaluate(
        form_expression_2, element_id, xi
    )  # 1-form

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
    wedge_eval = [
        zeros(
            n_evaluation_points,
            num_basis_form_expression_1...,
            num_basis_form_expression_2...,
        )::Array{Float64,1 + expression_rank_1 + expression_rank_2} for
        _ in 1:n_components_wedge
    ]

    # Assign wedge_indices: wedge_indices is just the concatenation of the indices of each expression
    wedge_indices = vcat(form_expression_1_indices, form_expression_2_indices)

    # Compute the wedge evaluation
    # (α₁β₁ + α₂β₂ + α₃β₃) dξ¹∧dξ²∧dξ³
    for form_components_idx in 1:3
        for cart_ind in CartesianIndices(wedge_eval[1])
            (point, basis_id_1, basis_id_2) = Tuple(cart_ind)
            wedge_eval[1][cart_ind] += 
                form_expression_1_eval[form_components_idx][point, basis_id_1] *
                form_expression_2_eval[form_components_idx][point, basis_id_2]
        end
    end

    return wedge_eval, wedge_indices
end
