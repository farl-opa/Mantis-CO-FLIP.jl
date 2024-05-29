

import .. Geometry
import .. FunctionSpaces


struct Dirichlet{n} <: AbstractBoundaryForms
    label_value_dict::Dict{String, Float64}
end

function get_estimated_nnz_bc_ext(bc_type::Dirichlet{n}, bc_space::FunctionSpaces.AbstractFunctionSpace) where {n}
    if n == 1
        return 2 * (FunctionSpaces.get_polynomial_degree(bc_space)+1), 2
    else
        error("Not implemented")
    end
end






struct Neumann{n, FN} <: AbstractBoundaryForms
    normal_derivatives::FN

    quad_nodes::NTuple{n, Vector{Float64}}
    quad_weights::NTuple{n, Vector{Float64}}
end

function (self::Neumann)(eval_point::Float64, label::String)
    if test_supported_bases[test_linear_idx] in [1,2,3]
        bc = compute_inner_product_L2(self.quad_nodes[2], 
                                    self.quad_weights[2],
                                    x -> self.normal_derivatives(x, label), 
                                    view(test_basis_evals[(0, 0)], :, test_linear_idx))
    elseif test_supported_bases[test_linear_idx] == 4
        bc = compute_inner_product_L2(self.quad_nodes[1], 
                                    self.quad_weights[1],
                                    x -> self.neumann_function(x, "bottom"), 
                                    view(test_basis_evals[(0, 0)], :, test_linear_idx))
    elseif test_supported_bases[test_linear_idx] == 6
        bc = compute_inner_product_L2(self.quad_nodes[1], 
                                    self.quad_weights[1],
                                    x -> self.neumann_function(x, "top"), 
                                    view(test_basis_evals[(0, 0)], :, test_linear_idx))
    elseif test_supported_bases[test_linear_idx] in [7,8,9]
        bc = compute_inner_product_L2(self.quad_nodes[2], 
                                    self.quad_weights[2],
                                    x -> self.neumann_function(x, "right"), 
                                    view(test_basis_evals[(0, 0)], :, test_linear_idx))
    else
        bc = 0.0
    end
    println(bc, " ", test_supported_bases[test_linear_idx], " ", test_linear_idx)
end

