
function _compute_square_error_per_element(computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2") where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, TF1 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, TF2 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, Q <: Quadrature.QuadratureRule{manifold_dim}}
    num_elements = Geometry.get_num_elements(Forms.get_geometry(computed_sol))
    result = Vector{Float64}(undef, num_elements)

    for elem_id in 1:1:num_elements
        difference = computed_sol - exact_sol
        if norm == "L2"
            result[elem_id] = sum(Forms.evaluate_inner_product(difference, difference, elem_id, quad_rule)[3])
        elseif norm == "Linf"
            println("WARNING: The Linf evaluation only uses the quadrature nodes as evaluation points!")
            result[elem_id] = maximum(Forms.evaluate(difference, elem_id, Quadrature.get_quadrature_nodes(quad_rule))[1][1])
        elseif norm == "H1"
            Error("Computing the H1 norm still needs to be updated.")
            d_difference = Forms.exterior_derivative(difference)
            result[elem_id] = sum(Forms.evaluate_inner_product(d_difference, d_difference, elem_id, quad_rule)[3])
        end
    end

    return result
end


function compute_error_per_element(computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2") where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, TF1 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, TF2 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, Q <: Quadrature.QuadratureRule{manifold_dim}}
    partial_result = _compute_square_error_per_element(computed_sol, exact_sol, quad_rule, norm)
    if norm == "Linf"
        return partial_result
    elseif norm == "L2" || norm == "H1"
        return sqrt.(partial_result)
    else
        throw(ArgumentError("Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."))
    end
end

function compute_error_total(computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2") where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, TF1 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, TF2 <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, Q <: Quadrature.QuadratureRule{manifold_dim}}
    partial_result = _compute_square_error_per_element(computed_sol, exact_sol, quad_rule, norm)
    if norm == "Linf"
        return maximum(partial_result)
    elseif norm == "L2" || norm == "H1"
        return sqrt(sum(partial_result))
    else
        throw(ArgumentError("Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."))
    end
end