############################################################################################
#                                        Structures                                        #
############################################################################################

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
- `expression_rank`: The rank of the expression. 0-rank means no basis forms in expression, 1-rank 
      means one set of basis forms in the expression, 2-rank means two sets of basis forms in the 
      expression. Higher ranks are not possible.
- `G`: Type of the geometry
- `F`: Type of the children tuple. Can be another FormExpression, a FormField, in the future also FormSpaces.
"""
struct FormExpression{manifold_dim, form_rank, expression_rank, G, F} <:AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}
    geometry::G
    children::F
    op::String
    label::String

    function FormExpression(form::Tuple{F}, op::String) where {F <: AbstractFormExpression{manifold_dim, child_form_rank, child_expression_rank, G}} where {manifold_dim, child_form_rank, child_expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
        label = op * form[1].label
        if op == "★"
            form_rank = manifold_dim - child_form_rank

        elseif op == "d"
            @assert (F <: FormField) || (F <: AbstractFormSpace) "Exterior derivative can only be applied to FormFields."
            form_rank = child_form_rank + 1
        end
        new{manifold_dim, form_rank, child_expression_rank, G, Tuple{F}}(get_geometry(form...), form, op, label)
    end

    function FormExpression(forms::Tuple{F_1, F_2}, op::String) where {F_1 <: AbstractFormExpression{manifold_dim, form_rank_1, expression_rank_1, G}, F_2 <: AbstractFormExpression{manifold_dim, form_rank_2, expression_rank_2, G}} where {manifold_dim, form_rank_1, form_rank_2, expression_rank_1, expression_rank_2, G <: Geometry.AbstractGeometry{manifold_dim}}
        if typeof(forms[1]) <: FormField
            label_1 = forms[1].label
        else
            label_1 = "(" * forms[1].label * ")" * Subscripts.super("$form_rank_1")
        end

        if typeof(forms[2]) <: FormField
            label_2 = forms[2].label
        else
            label_2 = "(" * forms[2].label * ")" * Subscripts.super("$form_rank_2")
        end
        
        if op == "∧"
            if (expression_rank_1 + expression_rank_2) > 2
                throw(ArgumentError("Wedge of FormExpressions requires expressions with total expression rank smaller than three, got:expression_rank_1 andexpression_rank_2"))
            end
            if (form_rank_1 + form_rank_2) > manifold_dim
                throw(ArgumentError("Wedge of FormExpressions requires expressions with total form rank smaller than the manifold dimension ($manifold_dim), got:form_rank_1 andform_rank_2"))
            end
            # With the wedge sum the resulting form has a rank equal to the sum of the two forms
            form_rank = form_rank_1 + form_rank_2
            
        elseif op == "-"
            if form_rank_1 != form_rank_2
                throw(ArgumentError("Subtraction of FormExpressions requires expressions with the same rank, got:form_rank_1 andform_rank_2"))
            end
            # Since they are the same the resulting expression has the same rank
            form_rank = form_rank_1
    
        elseif op == "+"
            if form_rank_1 != form_rank_2
                throw(ArgumentError("Subtraction of FormExpressions requires expressions with the same rank, got:form_rank_1 andform_rank_2"))
            end
            # Since they are the same the resulting expression has the same rank
            form_rank = form_rank_1
        end

        label = label_1 * " " * op * " " * label_2
        new{manifold_dim, form_rank, expression_rank_1 + expression_rank_2, G, Tuple{F_1, F_2}}(get_geometry(forms...), forms, op, label)
    end
end

############################################################################################
#                                    Evaluation methods                                    #
############################################################################################

@doc raw"""
    evaluate(form::FormExpression{manifold_dim, form_rank, expression_rank, G, Tuple{F}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F <: AbstractFormExpression, G <: Geometry.AbstractGeometry{manifold_dim}}

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
function evaluate(form::FormExpression{manifold_dim, form_rank, expression_rank, G, Tuple{F}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F <: AbstractFormExpression}
    #print("Evaluating: " * form.label * "\n")
    if form.op == "d"
        form_eval = evaluate_exterior_derivative(form.children[1], element_idx, xi)

    elseif form.op == "★"
        form_eval = evaluate_hodge_star(form.children[1], element_idx, xi)
    end

    return form_eval
end

@doc raw"""
    evaluate(form::FormExpression{manifold_dim, form_rank, expression_rank, G, Tuple{F1, F2}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F1 <: AbstractFormExpression, F2 <: AbstractFormExpression}

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
function evaluate(form::FormExpression{manifold_dim, form_rank, expression_rank, G, Tuple{F1, F2}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F1 <: AbstractFormExpression, F2 <: AbstractFormExpression}
    #print("Evaluating: " * form.label * "\n")
    if form.op == "∧"
        form_eval = evaluate_wedge(form.children[1], form.children[2], element_idx, xi)
        
    elseif form.op == "-"
        form_1_eval = evaluate(form.children[1], element_idx, xi)
        form_2_eval = evaluate(form.children[2], element_idx, xi)
        form_eval = form_1_eval[1] - form_2_eval[1], form_1_eval[2]

    elseif form.op == "+"
        form_1_eval = evaluate(form.children[1], element_idx, xi)
        form_2_eval = evaluate(form.children[2], element_idx, xi)
        form_eval = form_1_eval[1] + form_2_eval[1], form_1_eval[2]
    end

    return form_eval
end


# TODO Why is this function here? The one above is not sufficient?
#      "-" and "+" should only exist for FormFields and AnalyticalFormFields.
#      It makes no sense for FormSpaces. As it is it will try to evaluate it for FormSpaces.
#      Should only be valid for FormExpressions of expression_rank = 0.
function evaluate(form::FormExpression{manifold_dim, form_rank, expression_rank, G, Tuple{F1, F2}}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}, F1 <: AbstractFormField, F2 <: AbstractFormField}
    #print("Evaluating: " * form.label * "\n")
    if form.op == "∧"
        form_eval = evaluate_wedge(form.children[1], form.children[2], element_idx, xi)

    elseif form.op == "-"
        form_1_eval = evaluate(form.children[1], element_idx, xi)
        form_2_eval = evaluate(form.children[2], element_idx, xi)
        form_eval = form_1_eval[1] - form_2_eval[1], form_1_eval[2]

    elseif form.op == "+"
        form_1_eval = evaluate(form.children[1], element_idx, xi)
        form_2_eval = evaluate(form.children[2], element_idx, xi)
        form_eval = form_1_eval[1] + form_2_eval[1], form_1_eval[2]
    end

    return form_eval
end

@doc raw"""
    wedge(form_1::F_1, form_2::F_2) {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1, expression_rank_1, G1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2, expression_rank_2, G2}} where {manifold_dim_1, form_rank_1, expression_rank_1, manifold_dim_2, form_rank_2, expression_rank_2, G1 <: Geometry.AbstractGeometry{manifold_dim_1}, G2 <: Geometry.AbstractGeometry{manifold_dim_2}}

Create a wedge product expression of two forms.

# Arguments
- `form_1::F_1`: First form
- `form_2::F_2`: Second form

# Returns
- `FormExpression`: A new FormExpression representing the wedge product.
"""
function wedge(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim_1, form_rank_1, expression_rank_1, G1}, F_2 <: AbstractFormExpression{manifold_dim_2, form_rank_2, expression_rank_2, G2}} where {manifold_dim_1, form_rank_1, expression_rank_1, manifold_dim_2, form_rank_2, expression_rank_2, G1 <: Geometry.AbstractGeometry{manifold_dim_1}, G2 <: Geometry.AbstractGeometry{manifold_dim_2}}
    return FormExpression((form_1, form_2), "∧")
end

# TODO This seems correct, partially.
"""
    Base.:-(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormField{manifold_dim, form_rank, 0, G}, F_2 <: AbstractFormField{manifold_dim, form_rank, 0, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Subtract two form fields.

# Arguments
- `form_1::F_1`: First form field
- `form_2::F_2`: Second form field

# Returns
- `FormExpression`: A new FormExpression representing the subtraction of the two form fields.
"""
function Base.:-(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim, form_rank, 0, G}, F_2 <: AbstractFormExpression{manifold_dim, form_rank, 0, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    return FormExpression((form_1, form_2), "-")
end

# TODO This seems correct, partially.
"""
    Base.:+(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormField{manifold_dim, form_rank, 0, G}, F_2 <: AbstractFormField{manifold_dim, form_rank, 0, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Add two form fields.

# Arguments
- `form_1::F_1`: First form field
- `form_2::F_2`: Second form field

# Returns
- `FormExpression`: A new FormExpression representing the addition of the two form fields.
"""
function Base.:+(form_1::F_1, form_2::F_2) where {F_1 <: AbstractFormExpression{manifold_dim, form_rank, 0, G}, F_2 <: AbstractFormExpression{manifold_dim, form_rank, 0, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    return FormExpression((form_1, form_2), "+")
end

@doc raw"""
    hodge(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}} where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Create a Hodge star expression of a form.

# Arguments
- `form::F`: The form to apply the Hodge star operator to

# Returns
- `FormExpression`: A new FormExpression representing the Hodge star operation
"""
function hodge(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}} where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    return FormExpression((form,), "★")
end

@doc raw"""
    exterior_derivative(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}

Create an exterior derivative expression of a form.

# Arguments
- `form::F`: The form to apply the exterior derivative to

# Returns
- `FormExpression`: A new FormExpression representing the exterior derivative
"""
function exterior_derivative(form::F) where {F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank, G}} where {manifold_dim, form_rank, expression_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    return FormExpression((form,), "d")
end

############################################################################################
#                                         Includes                                         #
############################################################################################

include("FormFields.jl")
include("FormSpaces.jl")

