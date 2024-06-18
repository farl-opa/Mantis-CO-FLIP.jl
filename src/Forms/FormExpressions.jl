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

abstract type AbstractFormExpression{n, k} end
abstract type AbstractFormField{n, k} end

# struct FormField{manifold_dim, form_rank, FS} <:AbstractFormExpression{manifold_dim, form_rank}
#     form_space::FS
#     coefficients::Vector{Float64}

#     function FormField(form_space::FS, coefficients::Vector{Float64}) where {FS <: AbstractFormSpace{manifold_dim, form_rank}} where {manifold_dim, form_rank}
#         # Need to have a check if the number of coefficients is the same as the number of dofs 
#         # The numbering of the coefficients here is the numbering of the basis, which is given by the space
#         # We need to make this clear
#         new{manifold_dim, form_rank, FS}(form_space, coefficients)
#     end
# end

struct FormField{manifold_dim, form_rank, FS} <:AbstractFormExpression{manifold_dim, form_rank}
    form_space::FS
    coefficients::Vector{Float64}
    label::String

    function FormField(form_space::FS, label::String) where {FS <: AbstractFormSpace{manifold_dim, form_rank}} where {manifold_dim, form_rank}
        # Here we just generate the coefficients as zeros, for now just a number (10), but this needs
        # to be the number of dofs of the form_space, so we need a method that returns this for each form_space.
        coefficients = zeros(Float64, 10)
        new{manifold_dim, form_rank, FS}(form_space, coefficients, label)
    end
end

function evaluate(form::FormField{manifold_dim, form_rank, FS}) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}
    print("Evaluating $(form.label)" * sub("$form_rank") * "\n")
    return 5.0
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
            label_1 = form_1.label * super("$form_rank_1")
        else
            label_1 = "(" * form_1.label * ")" * super("$form_rank_1")
        end

        if typeof(form_2) <: FormField
            label_2 = form_2.label * super("$form_rank_2")
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


