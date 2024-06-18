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

abstract type AbstractFormExpression{manifold_dim, form_rank} end
abstract type AbstractFormField{manifold_dim, form_rank} <: AbstractFormExpression{manifold_dim, form_rank} end

struct FormFieldTrialBasis{manifold_dim, form_rank, FS} <:AbstractFormField{manifold_dim, form_rank}
    form_space::FS
    label::String

    function FormFieldTrialBasis(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank}} where {manifold_dim, form_rank}
        label = label * super("$form_rank")
        new{manifold_dim, form_rank, FS}(form_space, label)
    end
end

struct FormFieldTestBasis{manifold_dim, form_rank, FS} <:AbstractFormField{manifold_dim, form_rank}
    form_space::FS
    label::String

    function FormFieldTestBasis(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank}} where {manifold_dim, form_rank}
        label = label * super("$form_rank")
        new{manifold_dim, form_rank, FS}(form_space, label)
    end
end

struct FormField{manifold_dim, form_rank, FS} <:AbstractFormField{manifold_dim, form_rank}
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

function evaluate(form::FormField{manifold_dim, 0, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0}}
    print("Evaluating $(form.label) \n")
    form_basis_eval, form_basis_indices = evaluate(form.form_space, element_idx, xi)
    form_eval = form_basis_eval * form.coefficients[form_basis_indices]

    return form_eval
end

function evaluate(form::FormField{manifold_dim, manifold_dim, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, manifold_dim}}
    print("Evaluating $(form.label) \n")
    form_basis_eval, form_basis_indices = evaluate(form.form_space, element_idx, xi)
    form_eval = form_basis_eval * form.coefficients[form_basis_indices]

    return form_eval
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

struct FormExpression{n, k, F} <:AbstractFormExpression{n, k}
    children::F
    op::String
    label::String

    function FormExpression(form::F, op::String) where {F <: AbstractFormExpression{manifold_dim, child_form_rank}} where {manifold_dim, child_form_rank}
        label = op * form.label
        if op == "⋆"
            form_rank = manifold_dim - child_form_rank

        elseif op == "d"
            @assert F <: FormField "Exterior derivative can only be applied to FormFields."
            form_rank = child_form_rank + 1
        end
        new{manifold_dim, form_rank, F}(form, op, label)
    end

    function FormExpression(form_1::F_1, form_2::F_2, expression_rank::Int, op::String) where {F_1 <: AbstractFormExpression{manifold_dim, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim, form_rank_2}} where {manifold_dim, form_rank_1, form_rank_2}
        if typeof(form_1) <: FormField
            label_1 = form_1.label
        else
            label_1 = "(" * form_1.label * ")" * super("$form_rank_1")
        end

        if typeof(form_2) <: FormField
            label_2 = form_2.label
        else
            label_2 = "(" * form_2.label * ")" * super("$form_rank_2")
        end
        
        label = label_1 * " " * op * " " * label_2
        new{manifold_dim, expression_rank, Tuple{F_1, F_2}}((form_1, form_2), op, label)
    end
end

function evaluate(form_expression::FormExpression{manifold_dim, form_rank, F}) where {manifold_dim, form_rank, F}
    print("Evaluating: " * form_expression.label * "\n")
    if form_expression.op == "∧"
        print("Evaluating children\n")
        # wedge operator
        child_1_eval = evaluate(form_expression.children[1])
        child_2_eval = evaluate(form_expression.children[2])

        print("Computing wedge product numerically\n")
        form_expression_eval = 1.0 # wedge(child_1_eval, child_2_eval)
    elseif form_expression.op == "⋆"
        print("Evaluating child\n")
        # Hodge-star operator
        child_eval = evaluate(form_expression.children[1])

        metric = 2.0
        print("Computing Hodge-star numerically\n")
        form_expression_eval = 2.0 # hodge(child_eval, metric)
    end

    return form_expression_eval
end

function wedge(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}
    return FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
end

function hodge(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    return FormExpression(form, "⋆")
end

function exterior_derivative(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    return FormExpression(form, "d")
end


