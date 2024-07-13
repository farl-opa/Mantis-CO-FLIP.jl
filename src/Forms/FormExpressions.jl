using Subscripts

"""
    FormField{manifold_dim, form_rank, FS} <: AbstractFormField{manifold_dim, form_rank}

Represents a differential form field.

# Fields
- `form_space::FS`: The form space associated with this field
- `coefficients::Vector{Float64}`: Coefficients of the form field
- `label::String`: Label for the form field

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `FS`: Type of the form space

# Inner Constructors
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F})`: Constructor for 0-forms and n-forms
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη})`: Constructor for 1-forms in 2D
- `FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F_dξ, F_dη, F_dζ})`: Constructor for 1-forms and 2-forms in 3D
"""
struct FormField{manifold_dim, form_rank, FS} <: AbstractFormField{manifold_dim, form_rank}
    form_space::FS
    coefficients::Vector{Float64}
    label::String

    """
        FormField(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank}}

    Construct a FormField with zero coefficients.

    # Arguments
    - `form_space::FS`: The form space
    - `label::String`: Label for the form field

    # Returns
    - `FormField`: A new FormField instance with zero coefficients
    """
    function FormField(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank}} where {manifold_dim, form_rank}
        # Here we just generate the coefficients as zeros, i.e., initialize the coefficients
        n_dofs = get_dim(form_space)
        coefficients = zeros(Float64, n_dofs)
        label = label * super("$form_rank")
        new{manifold_dim, form_rank, FS}(form_space, coefficients, label)
    end
end

@doc raw"""
    evaluate(form::FormField{manifold_dim, form_rank, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}

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

# Sizes
- `form_eval`: Vector of length `n_form_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
        `n_evaluation_points = n_1 * ... * n_n`, where `n_i` is the number of points in the component `i` of the tensor product Tuple.
"""
function evaluate(form::FormField{manifold_dim, form_rank, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}
    print("Evaluating $(form.label) \n")
    n_form_components = binomial(manifold_dim, form_rank)
    form_basis_eval, form_basis_indices = evaluate(form.form_space, element_idx, xi)

    form_eval = Vector{Vector{Float64}}(undef, n_form_components)

    for form_component_idx in 1:n_form_components
        form_eval[form_component_idx] = form_basis_eval[form_component_idx] * form.coefficients[form_basis_indices[form_component_idx]]
    end

    return form_eval
end

@doc raw"""
    evaluate_exterior_derivative(form::FormField{manifold_dim, form_rank, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}

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
"""
function evaluate_exterior_derivative(form::FormField{manifold_dim, form_rank, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}
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

    return d_form_eval
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
- `F`: Type of the children tuple. Can be another FormExpression, a FormField, in the future also FormSpaces.
"""
struct FormExpression{n, k, F} <:AbstractFormExpression{n, k}
    children::F
    op::String
    label::String

    function FormExpression(form::Tuple{F}, op::String) where {F <: AbstractFormExpression{manifold_dim, child_form_rank}} where {manifold_dim, child_form_rank}
        label = op * form[1].label
        if op == "⋆"
            form_rank = manifold_dim - child_form_rank

        elseif op == "d"
            @assert F <: FormField "Exterior derivative can only be applied to FormFields."
            form_rank = child_form_rank + 1
        end
        new{manifold_dim, form_rank, Tuple{F}}(form, op, label)
    end

    function FormExpression(forms::Tuple{F_1, F_2}, expression_rank::Int, op::String) where {F_1 <: AbstractFormExpression{manifold_dim, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim, form_rank_2}} where {manifold_dim, form_rank_1, form_rank_2}
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
        new{manifold_dim, expression_rank, Tuple{F_1, F_2}}(forms, op, label)
    end
end

@doc raw"""
    evaluate(form::FormExpression{manifold_dim, form_rank, Tuple{F}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, F <: AbstractFormExpression}

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
function evaluate(form::FormExpression{manifold_dim, form_rank, Tuple{F}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, F <: AbstractFormExpression}
    print("Evaluating: " * form.label * "\n")
    if form.op == "d"
        form_eval = evaluate_exterior_derivative(form.children[1], element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}})

    elseif form.op == "⋆"
        throw("⋆ not implemented yet: form[1].label")
    end

    return form_eval
end

@doc raw"""
    evaluate(form::FormExpression{manifold_dim, form_rank, Tuple{F1, F2}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, F1 <: AbstractFormExpression, F2 <: AbstractFormExpression}

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
function evaluate(form::FormExpression{manifold_dim, form_rank, Tuple{F1, F2}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, F1 <: AbstractFormExpression, F2 <: AbstractFormExpression}
    print("Evaluating: " * form.label * "\n")
    throw("∧ not imlemented yet: (form.children[1].label, form.children[2].label)")
    form_eval = 1.0
    return form_eval
end

@doc raw"""
    wedge(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}

Create a wedge product expression of two forms.

# Arguments
- `form_1::F_1`: First form
- `form_2::F_2`: Second form

# Returns
- `FormExpression`: A new FormExpression representing the wedge product.
"""
function wedge(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}
    return FormExpression((form_1, form_2), form_rank_1 + form_rank_2, "∧")
end

@doc raw"""
    hodge(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}

Create a Hodge star expression of a form.

# Arguments
- `form::F`: The form to apply the Hodge star operator to

# Returns
- `FormExpression`: A new FormExpression representing the Hodge star operation
"""
function hodge(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    return FormExpression((form,), "⋆")
end

@doc raw"""
    exterior_derivative(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}

Create an exterior derivative expression of a form.

# Arguments
- `form::F`: The form to apply the exterior derivative to

# Returns
- `FormExpression`: A new FormExpression representing the exterior derivative
"""
function exterior_derivative(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    return FormExpression((form,), "d")
end


