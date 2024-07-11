# import Graphs
# import GraphPlot

using Subscripts

# G₁ = Graphs.DiGraph(3) # graph with 3 vertices

# # make a triangle
# Graphs.add_edge!(G₁, 1, 2)
# Graphs.add_edge!(G₁, 1, 3)
# Graphs.add_edge!(G₁, 2, 3)

# Graphs.add_vertices!(G₁, 1)

# GraphPlot.gplot(G₁, nodelabel=1:3)

struct FormField{manifold_dim, form_rank, FS} <: AbstractFormField{manifold_dim, form_rank}
    form_space::FS
    coefficients::Vector{Float64}
    label::String

    function FormField(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank}} where {manifold_dim, form_rank}
        # Here we just generate the coefficients as zeros, i.e., initialize the coefficients
        n_dofs = get_dim(form_space)
        coefficients = zeros(Float64, n_dofs)
        label = label * super("$form_rank")
        new{manifold_dim, form_rank, FS}(form_space, coefficients, label)
    end
end

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

function evaluate(form::FormExpression{manifold_dim, form_rank, Tuple{F}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, F <: AbstractFormExpression}
    print("Evaluating: " * form.label * "\n")
    if form.op == "d"
        form_eval = evaluate_exterior_derivative(form.children[1], element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}})

    elseif form.op == "⋆"
        throw("⋆ not implemented yet: form[1].label")
    end

    return form_eval
end

function evaluate(form::FormExpression{manifold_dim, form_rank, Tuple{F1, F2}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, F1 <: AbstractFormExpression, F2 <: AbstractFormExpression}
    print("Evaluating: " * form.label * "\n")
    throw("∧ not imlemented yet: (form.children[1].label, form.children[2].label)")
    form_eval = 1.0
    return form_eval
end

function wedge(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}
    return FormExpression((form_1, form_2), form_rank_1 + form_rank_2, "∧")
end

function hodge(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    return FormExpression((form,), "⋆")
end

function exterior_derivative(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    return FormExpression((form,), "d")
end


