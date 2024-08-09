using Subscripts

"""
    FormField{manifold_dim, form_rank, G, FS} <: AbstractFormField{manifold_dim, form_rank, G}

Represents a differential form field.

# Fields
- `form_space::FS`: The form space associated with this field
- `coefficients::Vector{Float64}`: Coefficients of the form field
- `label::String`: Label for the form field

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the geometry
- `FS`: Type of the form space

# Inner Constructors
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F})`: Constructor for 0-forms and n-forms
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη})`: Constructor for 1-forms in 2D
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ})`: Constructor for 1-forms and 2-forms in 3D
"""
struct FormField{manifold_dim, form_rank, G, FS} <: AbstractFormField{manifold_dim, form_rank, G}
    geometry::G
    form_space::FS
    coefficients::Vector{Float64}
    label::String

    """
        FormField(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

    Construct a FormField with zero coefficients.

    # Arguments
    - `form_space::FS`: The form space
    - `label::String`: Label for the form field

    # Returns
    - `FormField`: A new FormField instance with zero coefficients
    """
    function FormField(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
        # Here we just generate the coefficients as zeros, i.e., initialize the coefficients
        n_dofs = get_num_basis(form_space)
        coefficients = zeros(Float64, n_dofs)
        label = label * super("$form_rank")
        new{manifold_dim, form_rank, G, FS}(get_geometry(form_space), form_space, coefficients, label)
    end
end

@doc raw"""
    evaluate(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}

Evaluate a FormField at given points.

# Arguments
- `form::FormField`: The form field to evaluate.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `form_eval::Vector{Vector{Float64}}`: Evaluated form field.
        `form_eval[i][j]` is the component `i`` of the form evaluated at the tensor
        product point `j`. The components are, by convention:
        0-forms: single component.
        1-forms:
            2D: ``[\mathrm{d}\xi_{1}, \mathrm{d}\xi_{2}]``
            3D: ``[\mathrm{d}\xi_{1}, \mathrm{d}\xi_{2}, \mathrm{d}\xi_{3}]``
        2-forms:
            3D: ``[\mathrm{d}\xi_{2}\wedge\mathrm{d}\xi_{3}, \mathrm{d}\xi_{3}\wedge\mathrm{d}\xi_{1}, \mathrm{d}\xi_{1}\wedge\mathrm{d}\xi_{2}]``
        n-forms: single component
- `form_indices`: Vector of length `n_form_components`, where each element is a Vector{Int} of length 1 with value 1, since for a 
        `FormField` there is only one `basis`. This is done for consistency with `FormBasis`.

# Sizes
- `form_eval`: Vector of length `n_form_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
        `n_evaluation_points = n_1 * ... * n_n`, where `n_i` is the number of points in the component `i` of the tensor product Tuple.
"""
function evaluate(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}
    print("Evaluating $(form.label) \n")
    n_form_components = binomial(manifold_dim, form_rank)
    form_basis_eval, form_basis_indices = evaluate(form.form_space, element_idx, xi)

    form_eval = Vector{Vector{Float64}}(undef, n_form_components)

    for form_component_idx in 1:n_form_components
        form_eval[form_component_idx] = form_basis_eval[form_component_idx] * form.coefficients[form_basis_indices[form_component_idx]]
    end

    form_indices = ones(Int, n_form_components)

    return form_eval, form_indices
end

@doc raw"""
    evaluate_exterior_derivative(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}

Evaluate the exterior derivative of a FormField at given points.

# Arguments
- `form::FormField`: The form field to evaluate
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `d_form_eval::Vector{Vector{Float64}}`: Evaluated exterior derivative of the form field.
        Follows the same data format as evaluate: `d_form_eval[i][j]` is the component `i` 
        of the exterior derivative evaluated at the tensor product `j`.

# Sizes
- `d_form_eval`: Vector of length `n_derivative_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
        `n_evaluation_points = n_1 * ... * n_n`, where `n_i` is the number of points in the component `i` of the tensor product Tuple.
- `d_form_indices`: Vector of length `n_derivative_components`, where each element is a Vector{Int} of length 1 of value 1, since for a 
        `FormField` there is only one `basis`. This is done for consistency with `FormBasis`.
"""
function evaluate_exterior_derivative(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}
    print("Evaluating exterior derivative of $(form.label) \n")
    n_form_components = binomial(manifold_dim, form_rank)
    d_form_basis_eval, form_basis_indices = evaluate_exterior_derivative(form.form_space, element_idx, xi)
    n_derivative_components = size(d_form_basis_eval, 1)
    
    d_form_eval = Vector{Vector{Float64}}(undef, n_derivative_components)

    # This algorithm works for all exterior derivatives, as long as the 
    # exterior derivative of the basis of the form space are correctly 
    # implemented. The algorithm works in the following way.
    # 
    # The number of components, n_form_components, of the original k-form, 
    # αᵏ, is given by
    #
    #   n_form_components = binomial(manifold_dim, form_rank)
    #
    # This means that αᵏ is given by n_form_components sets of coefficients, aᵢ,
    # one set per component, and an equal number of sets of basis functions, Nⱼ,
    # 
    #   aᵢ = [aᵢ₁, ... , aᵢᵣ] with i = 1, ..., n_form_components and r = Mᵢ the number of coefficients (basis) in this component
    #   Nᵢ = [Nᵢ₁, ..., Nᵢᵣ], with i and r as above
    # 
    # The computation (evaluation) of the k-form is to loop over the components and multiply
    # the basis by the coefficients (the basis as evaluated at the points of interest)
    # 
    #  αᵏᵢ = ∑ʳⱼ₌₁ aᵢⱼNᵢⱼ
    #
    # The computation of the derivative is the same, as long as there is
    # consistency on the computation of the derivatives of the basis. The derivative of
    # the k-form can be directly computed as
    #
    #   d αᵏᵢ = d ∑ʳⱼ₌₁ aᵢⱼNᵢⱼ = ∑ʳⱼ₌₁ aᵢⱼ d Nᵢⱼ
    #
    # The components of the derivatives of the basis need to be compatible, i.e., we need to use canonical basis:
    #       1-forms: dξ₁, dξ₂, dξ₃                    (in this order)
    #       2-forms: dξ₂ ∧ dξ₃, dξ₃ ∧ dξ₁, dξ₁ ∧ dξ₂  (in this order)
    #       3-forms: dξ₁ ∧ dξ₂ ∧ dξ₃
    # 
    for derivative_form_component_idx in 1:n_derivative_components
        d_form_eval[derivative_form_component_idx] = d_form_basis_eval[derivative_form_component_idx] * form.coefficients[form_basis_indices[derivative_form_component_idx]]
    end

    d_form_indices = ones(Int, n_derivative_components)

    return d_form_eval, d_form_indices
end

@doc raw"""
    evaluate_hodge_star(form::FormField{manifold_dim, form_rank, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}

Evaluate the Hodge star ⋆ of a FormField at given points.

# Arguments
- `form::FormField`: The form field to evaluate
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `form_eval::Vector{Vector{Float64}}`: Evaluated Hodge star of the form field.
        Follows the same data format as evaluate: `d_form_eval[i][j]` is the component `i` 
        of the exterior derivative evaluated at the tensor product `j`.

# Sizes
- `form_eval`: Vector of length `n_derivative_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
        `n_evaluation_points = n_1 * ... * n_n`, where `n_i` is the number of points in the component `i` of the tensor product Tuple.
- `form_indices`: Vector of length `n_derivative_components`, where each element is a Vector{Int} of length 1 of value 1, since for a 
        `FormField` there is only one `basis`. This is done for consistency with `FormBasis`.
"""
function evaluate_hodge_star(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}
    n_form_components = binomial(manifold_dim, form_rank)
    form_basis_eval, form_basis_indices = evaluate_hodge_star(form.form_space, element_idx, xi)

    form_eval = Vector{Vector{Float64}}(undef, n_form_components)

    for form_component ∈ 1:n_form_components
        form_eval[form_component] = form_basis_eval[form_component] * form.coefficients[form_basis_indices[form_component]]
    end

    form_indices = ones(Int, n_form_components)

    return form_eval, form_indices
end

@doc raw"""
    evaluate_inner_product(form::FormField{manifold_dim, form_rank, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}

Evaluate the Hodge star ⋆ of a FormField at given points.

# Arguments
- `form::FormField`: The form field to evaluate
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `form_eval::Vector{Vector{Float64}}`: Evaluated Hodge star of the form field.
        Follows the same data format as evaluate: `d_form_eval[i][j]` is the component `i` 
        of the exterior derivative evaluated at the tensor product `j`.

# Sizes
- `form_eval`: Vector of length `n_derivative_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
        `n_evaluation_points = n_1 * ... * n_n`, where `n_i` is the number of points in the component `i` of the tensor product Tuple.
- `form_indices`: Vector of length `n_derivative_components`, where each element is a Vector{Int} of length 1 of value 1, since for a 
        `FormField` there is only one `basis`. This is done for consistency with `FormBasis`.
"""
function evaluate_inner_product(form1::FormField{manifold_dim, form_rank, G, FS1}, form2::FormField{manifold_dim, form_rank, G, FS2}, element_idx::Int, quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS1 <: AbstractFormSpace{manifold_dim, form_rank, G}, FS2 <: AbstractFormSpace{manifold_dim, form_rank, G}}

    product_rows, product_cols, product_basis_eval = inner_product(form1.form_space,form2.form_space, element_idx, quad_rule)
    n_components = length(product_basis_eval)

    product_sum = 0.0

    for component ∈ 1:n_components
        product_sum += product_basis_eval[component]' * (form2.coefficients[product_cols[component]] .* form1.coefficients[product_rows[component]])
    end

    return product_sum
end

@doc raw"""
    FormExpression{n, k, F} <: AbstractFormExpression{n, k}

Represents an expression involving differential forms.

# Fields
- `children::F`: Tuple of child forms or expressions.
- `op::String`: Operator applied to the children.
- `label::String`: Label for the expression.

# Type parameters
- `n`: Dimension of the manifold.
- `k`: Rank of the resulting form.
- `G`: Type of the geometry
- `F`: Type of the children tuple. Can be another FormExpression, a FormField, in the future also FormSpaces.
"""
struct FormExpression{manifold_dim, form_rank, G, F} <:AbstractFormExpression{manifold_dim, form_rank, G}
    geometry::G
    children::F
    op::String
    label::String

    function FormExpression(form::Tuple{F}, op::String) where {F <: AbstractFormExpression{manifold_dim, child_form_rank, G}} where {manifold_dim, child_form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
        label = op * form[1].label
        if op == "🟉"
            form_rank = manifold_dim - child_form_rank

        elseif op == "d"
            @assert (F <: FormField) || (F <: AbstractFormSpace) "Exterior derivative can only be applied to FormFields."
            form_rank = child_form_rank + 1
        end
        new{manifold_dim, form_rank, G, Tuple{F}}(get_geometry(form...), form, op, label)
    end

    function FormExpression(forms::Tuple{F_1, F_2}, expression_rank::Int, op::String) where {F_1 <: AbstractFormExpression{manifold_dim, form_rank_1, G}, F_2 <: AbstractFormExpression{manifold_dim, form_rank_2, G}} where {manifold_dim, form_rank_1, form_rank_2, G <: Geometry.AbstractGeometry{manifold_dim}}
        if typeof(forms[1]) <: FormField
            label_1 = forms[1].label
        else
            label_1 = "(" * forms[1].label * ")" * super("$form_rank_1")
        end

        if typeof(forms[2]) <: FormField
            label_2 = forms[2].label
        else
            label_2 = "(" * forms[2].label * ")" * super("$form_rank_2")
        end
        
        label = label_1 * " " * op * " " * label_2
        new{manifold_dim, expression_rank, G, Tuple{F_1, F_2}}(get_geometry(forms...), forms, op, label)
    end
end

@doc raw"""
    evaluate(form::FormExpression{manifold_dim, form_rank, G, Tuple{F}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F <: AbstractFormExpression, G <: Geometry.AbstractGeometry{manifold_dim}}

Evaluate a unary FormExpression at given points.

# Arguments
- `form::FormExpression`: The form expression to evaluate
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `form_eval::Vector{Vector{Float64}}`: Evaluated form expression at the tensor product points.

# Sizes
- `form_eval`: Vector of length `n_form_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
"""
function evaluate(form::FormExpression{manifold_dim, form_rank, G, Tuple{F}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F <: AbstractFormExpression}
    print("Evaluating: " * form.label * "\n")
    if form.op == "d"
        form_eval = evaluate_exterior_derivative(form.children[1], element_idx, xi)

    elseif form.op == "🟉"
        form_eval = evaluate_hodge_star(form.children[1], element_idx, xi)
    end

    return form_eval
end

@doc raw"""
    evaluate(form::FormExpression{manifold_dim, form_rank, G, Tuple{F1, F2}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F1 <: AbstractFormExpression, F2 <: AbstractFormExpression}

Evaluate a binary FormExpression at given points.

# Arguments
- `form::FormExpression`: The form expression to evaluate
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `form_eval::Float64`: Evaluated form expression (placeholder)

# Note: This function is not fully implemented yet.
"""
function evaluate(form::FormExpression{manifold_dim, form_rank, G, Tuple{F1, F2}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F1 <: AbstractFormExpression, F2 <: AbstractFormExpression}
    print("Evaluating: " * form.label * "\n")
    throw("∧ not imlemented yet: (form.children[1].label, form.children[2].label)")
    form_eval = 1.0
    return form_eval
end

@doc raw"""
    wedge(form_1::F_1, form_2::F_2) {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1, G1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2, G2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2, G1 <: Geometry.AbstractGeometry{manifold_dim_1}, G2 <: Geometry.AbstractGeometry{manifold_dim_2}}

Create a wedge product expression of two forms.

# Arguments
- `form_1::F_1`: First form
- `form_2::F_2`: Second form

# Returns
- `FormExpression`: A new FormExpression representing the wedge product.
"""
function wedge(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1, G1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2, G2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2, G1 <: Geometry.AbstractGeometry{manifold_dim_1}, G2 <: Geometry.AbstractGeometry{manifold_dim_2}}
    return FormExpression((form_1, form_2), form_rank_1 + form_rank_2, "∧")
end

@doc raw"""
    hodge(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Create a Hodge star expression of a form.

# Arguments
- `form::F`: The form to apply the Hodge star operator to

# Returns
- `FormExpression`: A new FormExpression representing the Hodge star operation
"""
function hodge(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    return FormExpression((form,), "🟉")
end

@doc raw"""
    exterior_derivative(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Create an exterior derivative expression of a form.

# Arguments
- `form::F`: The form to apply the exterior derivative to

# Returns
- `FormExpression`: A new FormExpression representing the exterior derivative
"""
function exterior_derivative(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    return FormExpression((form,), "d")
end


