# import Graphs
# import GraphPlot

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

struct FormField{manifold_dim, form_rank} <:AbstractFormExpression{manifold_dim, form_rank}
    label::String

    function FormField(manifold_dim::Int, form_rank::Int, label::String)
        # Need to have a check if the number of coefficients is the same as the number of dofs 
        # The numbering of the coefficients here is the numbering of the basis, which is given by the space
        # We need to make this clear
        new{manifold_dim, form_rank}(label)
    end
end

struct FormExpression{n, k, F} <:AbstractFormExpression{n, k}
    children::F
    op::String
    label::String

    function FormExpression(form::F, op::String, label::String) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
        new{manifold_dim, form_rank, F}(form, op, label)
    end

    function FormExpression(form_1::F_1, form_2::F_2, expression_rank::Int, op::String) where {F_1 <: AbstractFormExpression{manifold_dim, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim, form_rank_2}} where {manifold_dim, form_rank_1, form_rank_2}
        if typeof(form_1) <: FormField
            label_1 = form_1.label 
        else
            label_1 = "(" * form_1.label * ")"
        end

        if typeof(form_2) <: FormField
            label_2 = form_2.label 
        else
            label_2 = "(" * form_2.label * ")"
        end
        
        label = label_1 * " " * op * " " * label_2
        new{manifold_dim, expression_rank, Tuple{F_1, F_2}}((form_1, form_2), op, label)
    end
end

# function evaluate(form::FormExpression{n, k, F}) where {n, k, F}
#     print("Evaluating: " * form.label * "\n")
# end

struct DualFormExpression{n, k, F1, F2} <:AbstractFormExpression{n, k}
    children::Tuple{F1, F2}
    op::String
    label::String

    function FormExpression(form_1::F_1, form_2::F_2, expression_rank::Int, op::String) where {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}
        @assert manifold_dim_1 == manifold_dim_2 "Dimension of manifolds of children mismatch."
        if typeof(form_1) <: FormFieldExpression
            label_1 = form_1.label 
        else
            label_1 = "(" * form_1.label * ")"
        end

        if typeof(form_2) <: FormFieldExpression
            label_2 = form_2.label 
        else
            label_2 = "(" * form_2.label * ")"
        end
        
        label = label_1 * " " * op * " " * label_2
        new{manifold_dim_1, expression_rank, F_1, F_2}((form_1, form_2), op, label)
    end
end


function (∧)(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2}} where {manifold_dim_1, form_rank_1, manifold_dim_2, form_rank_2}
    return FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
end

function (⋆)(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank}} where {manifold_dim, form_rank}
    return FormExpression(form_1, form_2, form_rank_1 + form_rank_2, "∧")
end


function evaluate(form::FormExpression)
    print("Evaluating: " * form.label * "\n")
    for form_child in form.children 
        evaluate(form_child)
    end 
end

