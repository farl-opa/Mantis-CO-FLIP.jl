
import .. Geometry
import .. FunctionSpaces

# We can consider parameterising on the dimension here as well.
struct PoissonBilinearForm1D{Frhs, Fbilinear, Flinear, Ttrial, Ttest, TG} <: AbstractBilinearForms
    forcing::Frhs #where F <: Function

    bilinear_form_function::Fbilinear #where Fb <: Function
    linear_form_function::Flinear #where Fl <: Function

    space_trial::Ttrial#FunctionSpaces.AbstractFunctionSpace#AbstractFiniteElementSpace{1}
    space_test::Ttest#FunctionSpaces.AbstractFunctionSpace#AbstractFiniteElementSpace{1}

    geometry::TG#Geometry.AbstractGeometry{1, m} where {m}

    quad_nodes::Vector{Float64}
    quad_weights::Vector{Float64}
end


# Every bilinear form will need the functions defined below. These are 
# used by the global assembler to set up the problem.
function get_num_elements(PB::PoissonBilinearForm1D)
    return Geometry.get_num_elements(PB.geometry)
end

function get_problem_size(PB::PoissonBilinearForm1D)
    return FunctionSpaces.get_dim(PB.space_trial), FunctionSpaces.get_dim(PB.space_test)
end

function get_estimated_nnz_per_elem(PB::PoissonBilinearForm1D)
    return FunctionSpaces.get_max_supported_basis(PB.space_trial) * FunctionSpaces.get_max_supported_basis(PB.space_test), FunctionSpaces.get_max_supported_basis(PB.space_test)
end

@doc raw"""
    compute_inner_product_L2(quad_nodes, quad_weights, fxi::F, gxi::F) where {F <: Function}

Compute the ``L^2``-inner product between f and g.

# Arguments
- `quad_nodes::Vector{Float64}`: (mapped) quadrature points.
- `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
- `fxi::Function`: one of the two function to compute the inner product with. To be evaluated at the quadrature nodes.
- `gxi::Function`: one of the two function to compute the inner product with. To be evaluated at the quadrature nodes.
"""
function compute_inner_product_L2(quad_nodes, quad_weights, fxi::F, gxi::F) where {F <: Function}
    result = 0.0
    for node_idx in eachindex(quad_weights)
        result += quad_weights[node_idx] * (fxi(quad_nodes[node_idx]) * gxi(quad_nodes[node_idx]))
    end
    return result
end

@doc raw"""
    compute_inner_product_L2(quad_nodes, quad_weights, fxi::T, gxi::T) where {T <: AbstractArray{Float64, 1}}

Compute the ``L^2``-inner product between f and g.

# Arguments
- `quad_nodes::Vector{Float64}`: (mapped) quadrature points.
- `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
- `fxi<:AbstractArray{Float64, 1}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
- `gxi<:AbstractArray{Float64, 1}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
"""
function compute_inner_product_L2(quad_nodes, quad_weights, fxi::T, gxi::T) where {T <: AbstractArray{Float64, 1}}
    result = 0.0
    for node_idx in eachindex(quad_weights)
        result += quad_weights[node_idx] * (fxi[node_idx] * gxi[node_idx])
    end
    return result
end

@doc raw"""
    compute_inner_product_L2(quad_nodes, quad_weights, fxi::F, gxi::T) where {F <: Function, T <: AbstractArray{Float64, 1}

Compute the ``L^2``-inner product between f and g.

# Arguments
- `quad_nodes::Vector{Float64}`: (mapped) quadrature points.
- `quad_weights::Vector{Float64}`: (mapped) quadrature weights.
- `fxi::Function`: one of the two function to compute the inner product with. To be evaluated at the quadrature nodes.
- `gxi<:AbstractArray{Float64, 1}`: one of the two function to compute the inner product with. Already evaluated at the quadrature nodes.
"""
function compute_inner_product_L2(quad_nodes, quad_weights, fxi::F, gxi::T) where {F <: Function, T <: AbstractArray{Float64, 1}}
    result = 0.0
    for node_idx in eachindex(quad_weights)
        result += quad_weights[node_idx] * (fxi(quad_nodes[node_idx]) * gxi[node_idx])
    end
    return result
end

function (PB::PoissonBilinearForm1D{Frhs, Fbilinear, Flinear, Ttrial, Ttest, TG} where {Frhs, Fbilinear, Flinear, Ttrial, Ttest, TG})(element_id) 
    # Computed bases and their derivatives on the current element and in 
    # the physical domain, so no need to transform them!
    trial_basis_evals, trial_supported_bases = FunctionSpaces.evaluate(PB.space_trial, element_id, (PB.quad_nodes,), 1)
    test_basis_evals, test_supported_bases = FunctionSpaces.evaluate(PB.space_test, element_id, (PB.quad_nodes,), 1)
    
    # Compute the quantities related to the geometry. We use the metric 
    # to prevent the need to invert the Jacobian.
    mapped_nodes = Geometry.evaluate(PB.geometry, element_id, (PB.quad_nodes,))
    jacobian = Geometry.jacobian(PB.geometry, element_id, (PB.quad_nodes,))
    #metric_inv = inv(transpose(jacobian) * jacobian)
    metric_inv = 1.0 ./ (jacobian .* jacobian)

    mapped_weights = jacobian .* PB.quad_weights
    
    # Count the number of supported basis on this element.
    n_supported_bases_trial = length(trial_supported_bases)
    n_supported_bases_test = length(test_supported_bases)
    n_supported_total = n_supported_bases_trial * n_supported_bases_test

    # Pre-allocate the local matrices (their row, colum, and value vectors).
    A_row_idx = Vector{Int}(undef, n_supported_total)
    A_col_idx = Vector{Int}(undef, n_supported_total)
    A_elem = Vector{Float64}(undef, n_supported_total)
    b_col_idx = Vector{Int}(undef, n_supported_bases_test)
    b_elem = Vector{Float64}(undef, n_supported_bases_test)

    for test_linear_idx in 1:1:n_supported_bases_test

        for trial_linear_idx in 1:1:n_supported_bases_trial
            idx = (test_linear_idx - 1) * n_supported_bases_test + trial_linear_idx

            A_row_idx[idx] = trial_supported_bases[trial_linear_idx]
            A_col_idx[idx] = test_supported_bases[test_linear_idx]

            Aij = compute_inner_product_L2(mapped_nodes, 
                                           mapped_weights, 
                                           view(metric_inv .* trial_basis_evals[1], :, trial_linear_idx), 
                                           view(test_basis_evals[1], :, test_linear_idx))
            
            # # Inner product, can be done using a separate function.
            # Aij = 0.0
            # for node_idx in eachindex(PB.quad_weights)
            #     trial_dxi = trial_basis_evals[1][node_idx, trial_linear_idx]
            #     test_dxj = test_basis_evals[1][node_idx, test_linear_idx]
            #     mapped_weights = jacobian[node_idx] * PB.quad_weights[node_idx]
            #     Aij += mapped_weights * PB.bilinear_form_function(trial_dxi, test_dxj)
            # end
            
            A_elem[idx] = Aij
        end


        b_col_idx[test_linear_idx] = test_supported_bases[test_linear_idx]

        bi = compute_inner_product_L2(mapped_nodes, 
                                      mapped_weights,
                                      PB.forcing, 
                                      view(test_basis_evals[0], :, test_linear_idx))
        
        # Inner product, can be done using a separate function.
        # Li = 0.0
        # for node_idx in eachindex(mapped_nodes, PB.quad_weights)
        #     test_i = test_basis_evals[0][node_idx, test_linear_idx]
        #     f_i = PB.forcing(mapped_nodes[node_idx])
        #     mapped_weights = jacobian[node_idx] * PB.quad_weights[node_idx]
        #     Li += mapped_weights * PB.linear_form_function(test_i, f_i)
        # end
        
        b_elem[test_linear_idx] = bi
    end
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # column_indices, values for the vector part.
    return (A_row_idx, A_col_idx, A_elem), (b_col_idx, b_elem)
    
end
