
import .. Geometry
import .. FunctionSpaces
import .. Forms
import .. Quadrature

import ... Main  # For testing only, to be able to use Main.@code_warntype (but not when precompiling!)





"""
    struct WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest, G} <: AbstractInputs

Contains the required data for a one variable FEM.

# Fields
- `forcing::Frhs <: Function`: Forcing function, must take in an NTuple{manifold_dim, Float64} as argument.
- `space_trial::Ttrial <: FunctionSpaces.AbstractFunctionSpace`: Trial (Solution) space.
- `space_test::Ttest <: FunctionSpaces.AbstractFunctionSpace`: Test space.
- `geometry::TG <: Geometry.AbstractGeometry{manifold_dim}`: Geometry.
- `quad_nodes::NTuple{manifold_dim, Vector{Float64}}`: Quadrature nodes.
- `quad_weights::Vector{Float64}`: Vector of the tensor product of the quadrature weights.

# See also
[`Quadrature.tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F) where {domain_dim, F <: Function}`](@ref) to compute the quadrature weights.
"""
struct WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest} <: AbstractInputs
    forcing::Frhs

    space_trial::Ttrial  # n-form that needs to be solved for.
    space_test::Ttest  # n-form test functions.

    quad_rule::Quadrature.QuadratureRule{manifold_dim}

    function WeakFormInputs(forcing::Frhs, 
                            space_trial::Ttrial, 
                            space_test::Ttest,
                            quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, form_rank, 
                            G <: Geometry.AbstractGeometry{manifold_dim},
                            Frhs <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}, # These spaces current have the same form_rank, do we want this?
                            Ttrial <: Forms.AbstractFormExpression{manifold_dim, form_rank, G},
                            Ttest <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}}
        
        new{manifold_dim, Frhs, Ttrial, Ttest}(forcing, space_trial, space_test, quad_rule)
    end
end

# Every bilinear form will need the functions defined below. These are 
# used by the global assembler to set up the problem.
function get_num_elements(wf::WeakFormInputs)
    geo = Forms.get_geometry(wf.space_trial)
    return Geometry.get_num_elements(geo)
end

function get_problem_size(wf::WeakFormInputs)
    return Forms.get_num_basis(wf.space_trial), Forms.get_num_basis(wf.space_test)
end

function get_estimated_nnz_per_elem(wf::WeakFormInputs)
    return Forms.get_max_local_dim(wf.space_trial) * Forms.get_max_local_dim(wf.space_test), Forms.get_max_local_dim(wf.space_test)
end

function get_boundary_dof_indices(wf::WeakFormInputs)
    return Forms.get_boundary_dof_indices(wf.space_trial)
end


@doc raw"""
    poisson_non_mixed(inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Bilinear form for the computation of the Poisson equation on the given element.

This function computes the contribution of the given element of both the 
bilinear and linear form for the Poisson equation. The associated weak 
formulation is:

For given ``f^n \in L^2 \Lambda^n (\Omega)``, find ``\phi^n \in H^1_0 \Lambda^n (\Omega)`` such that 
```math
\int_{\Omega} d^\star \phi^n \wedge \star d^\star \varphi^n = -\int_{\Omega} f^n \wedge \star \varphi^n \quad \forall \ \varphi^n \in H^1_0 \Lambda^n (\Omega)
```

# Arguments
- `inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}`: weak form setup.
- `elem_id::NTuple{n,Int}`: element for which to compute the contribution.
"""
function poisson_non_mixed(inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    
    if Forms.get_form_rank(inputs.space_trial) == manifold_dim
        # Top form, so codifferentials are needed
        # The bilinear form is the inner product between the 
        # codifferential of the trial form with the codifferential of 
        # the test form.
        trial_form = Forms.hodge(Forms.exterior_derivative(Forms.hodge(inputs.space_trial)))
        test_form = Forms.hodge(Forms.exterior_derivative(Forms.hodge(inputs.space_test)))
    else
        # Zero form (for the time being, more form ranks can be added 
        # later), so only exterior derivative is needed.
        trial_form = Forms.exterior_derivative(inputs.space_trial)
        test_form = Forms.exterior_derivative(inputs.space_test)
    end

    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(test_form, trial_form, element_id, inputs.quad_rule)

    println(element_id)
    display(A_row_idx)
    display(A_col_idx)
    display(A_elem)

    # The linear form is the inner product between the trial form and 
    # the forcing function which is a form of an appropriate rank.
    b_row_idx, b_col_idx, b_elem = Forms.evaluate_inner_product(inputs.space_test, inputs.forcing, element_id, inputs.quad_rule)
    display(b_row_idx)
    display(b_col_idx)
    display(b_elem)
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx[1], b_elem[1]) # Take the first component out of one component
    
end




# """
#     struct WeakFormInputs{n, Frhs, Ttrial, Ttest, TG} <: AbstractInputs

# Contains the required data for a one variable FEM.

# # Fields
# - `forcing::Frhs <: Function`: Forcing function, must take in an NTuple{n, Float64} as argument.
# - `space_trial::Ttrial <: FunctionSpaces.AbstractFunctionSpace`: Trial (Solution) space.
# - `space_test::Ttest <: FunctionSpaces.AbstractFunctionSpace`: Test space.
# - `geometry::TG <: Geometry.AbstractGeometry{n}`: Geometry.
# - `quad_nodes::NTuple{n, Vector{Float64}}`: Quadrature nodes.
# - `quad_weights::Vector{Float64}`: Vector of the tensor product of the quadrature weights.

# # See also
# [`Quadrature.tensor_product_rule(p::NTuple{domain_dim, Int}, quad_rule::F) where {domain_dim, F <: Function}`](@ref) to compute the quadrature weights.
# """
# struct WeakFormInputs{n, Frhs, Ttrial, Ttest, TG} <: AbstractInputs
#     forcing::Frhs

#     space_trial::Ttrial
#     space_test::Ttest

#     geometry::TG

#     # Quadrature rule in reference domain. Nodes per dimension, weights 
#     # as returned from Mantis.Quadrature.tensor_product_weights
#     quad_nodes::NTuple{n, Vector{Float64}}
#     quad_weights::Vector{Float64}

#     function WeakFormInputs(forcing::Frhs, 
#                             space_trial::Ttrial, 
#                             space_test::Ttest, 
#                             geometry::TG, 
#                             quad_nodes::NTuple{n, Vector{Float64}}, 
#                             quad_weights::Vector{Float64}) where {n, 
#                             Frhs <: Function, 
#                             Ttrial <: FunctionSpaces.AbstractFunctionSpace, 
#                             Ttest <: FunctionSpaces.AbstractFunctionSpace,
#                             TG <: Geometry.AbstractGeometry{n}}
        
#         new{n, Frhs, Ttrial, Ttest, TG}(forcing, space_trial, space_test, 
#                                            geometry, quad_nodes, quad_weights)
#     end
# end

# # Every bilinear form will need the functions defined below. These are 
# # used by the global assembler to set up the problem.
# function get_num_elements(wf::WeakFormInputs)
#     return Geometry.get_num_elements(wf.geometry)
# end

# function get_problem_size(wf::WeakFormInputs)
#     return FunctionSpaces.get_num_basis(wf.space_trial), FunctionSpaces.get_num_basis(wf.space_test)
# end

# function get_estimated_nnz_per_elem(wf::WeakFormInputs)
#     return FunctionSpaces.get_max_local_dim(wf.space_trial) * FunctionSpaces.get_max_local_dim(wf.space_test), FunctionSpaces.get_max_local_dim(wf.space_test)
# end

# function get_boundary_dof_indices(wf::WeakFormInputs)
#     return FunctionSpaces.get_boundary_dof_indices(wf.space_trial)
# end


# @doc raw"""
#     poisson_weak_form_1(inputs::WeakFormInputs{n, Frhs, Ttrial, Ttest, TG}, element_id) where {n, Frhs, Ttrial, Ttest, TG}

# Bilinear form for the computation of the Poisson equation on the given element.

# This function computes the contribution of the given element of both the 
# bilinear and linear form for the Poisson equation. The associated weak 
# formulation is:

# For given ``f \in L^2(\Omega)``, find ``\phi \in H^1(\Omega)`` such that 
# ```math
# \int_{\Omega} \nabla \varphi \cdot \nabla \phi \;d\Omega = \int_{\Omega} \varphi f(x) \;d\Omega \quad \forall \ \varphi \in H^1(\Omega)
# ```

# Note that there are not boundary conditions specified. To solve what the 
# global assembler returns, one should add an extra condition (e.g. average 
# of ``\phi`` is zero)

# # Arguments
# - `elem_id::NTuple{n,Int}`: element for which to compute the contribution.
# """
# function poisson_weak_form_1(inputs::WeakFormInputs{n, Frhs, Ttrial, Ttest, TG}, element_id) where {n, Frhs, Ttrial, Ttest, TG}
#     # Computed bases and their derivatives.
#     trial_basis_evals, trial_supported_bases = FunctionSpaces.evaluate(inputs.space_trial, element_id, inputs.quad_nodes, 1)
#     test_basis_evals, test_supported_bases = FunctionSpaces.evaluate(inputs.space_test, element_id, inputs.quad_nodes, 1)
    
#     # Compute the quantities related to the geometry.
#     mapped_nodes = Geometry.evaluate(inputs.geometry, element_id, inputs.quad_nodes)
#     metric_inv, _, jac_det = Geometry.inv_metric(inputs.geometry, element_id, inputs.quad_nodes)

#     # Compute rhs on mapped nodes.
#     # It can be more efficient to avoid this, but that is easier when we 
#     # have a better interface for the inner products.
#     fxy = inputs.forcing.(NTuple{n, Vector{Float64}}(mapped_nodes[:,i] for i in 1:1:n)...)
    
#     # Count the number of supported basis on this element.
#     n_supported_bases_trial = length(trial_supported_bases)
#     n_supported_bases_test = length(test_supported_bases)
#     n_supported_total = n_supported_bases_trial * n_supported_bases_test

#     # Pre-allocate the local matrices (their row, colum, and value vectors).
#     A_row_idx = Vector{Int}(undef, n_supported_total)
#     A_col_idx = Vector{Int}(undef, n_supported_total)
#     A_elem = Vector{Float64}(undef, n_supported_total)
#     b_col_idx = Vector{Int}(undef, n_supported_bases_test)
#     b_elem = Vector{Float64}(undef, n_supported_bases_test)

#     # Not exactly elegant, but we need a way to easily and stably 
#     # extract things like a gradient from the evaluations. The filter 
#     # iterator only iterates over the keys for which the sum is 
#     # 1 (so all partial derivatives of order 1).
#     key_itr_test = sort(collect(Iterators.filter(x -> sum(x) == 1 ? true : false, keys(test_basis_evals))), rev=true)
#     key_itr_trial = sort(collect(Iterators.filter(x -> sum(x) == 1 ? true : false, keys(trial_basis_evals))), rev=true)

#     for test_linear_idx in 1:1:n_supported_bases_test

#         # See the comment below.
#         if n == 1
#             grad_test = NTuple{n, SubArray{Matrix{Float64}, 1, Array{Matrix{Float64},1}, Tuple{Base.Slice{Base.OneTo{Int}}}, true}}(view(test_basis_evals[key], :, test_linear_idx) for key in key_itr_test)
#         else
#             grad_test = NTuple{n, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}}(view(test_basis_evals[key], :, test_linear_idx) for key in key_itr_test)
#         end

#         for trial_linear_idx in 1:1:n_supported_bases_trial
#             idx = (test_linear_idx - 1) * n_supported_bases_test + trial_linear_idx

#             A_row_idx[idx] = trial_supported_bases[trial_linear_idx]
#             A_col_idx[idx] = test_supported_bases[test_linear_idx]

#             # Not exactly elegant. In order to make this type stable, I 
#             # made sure that the size of the tuple is known (there are n 
#             # partial derivatives of interest in dimension n) and I 
#             # specified the type of the SubArray.
#             #grad_trail = NTuple{n, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}}(view(trial_basis_evals[key], :, trial_linear_idx) for key in key_itr_trial)
#             if n == 1
#                 grad_trail = NTuple{n, SubArray{Matrix{Float64}, 1, Array{Matrix{Float64},1}, Tuple{Base.Slice{Base.OneTo{Int}}}, true}}(view(trial_basis_evals[key], :, trial_linear_idx) for key in key_itr_trial)
#             else
#                 grad_trail = NTuple{n, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}}(view(trial_basis_evals[key], :, trial_linear_idx) for key in key_itr_trial)
#             end

#             Aij = compute_inner_product_L2(jac_det, inputs.quad_weights, 
#                                            metric_inv, grad_trail, grad_test)
            
#             A_elem[idx] = Aij
#         end


#         b_col_idx[test_linear_idx] = test_supported_bases[test_linear_idx]

#         bi = compute_inner_product_L2(jac_det, inputs.quad_weights, 
#                                       LinearAlgebra.I, fxy,
#                                       view(test_basis_evals[n == 1 ? 0 : NTuple{n, Int}(zeros(Int, n))], :, test_linear_idx))

#         b_elem[test_linear_idx] = bi
#     end
    
#     # The output should be the contribution to the left-hand-side matrix 
#     # A and right-hand-side vector b. The outputs are tuples of 
#     # row_indices, column_indices, values for the matrix part and 
#     # column_indices, values for the vector part.
#     return (A_row_idx, A_col_idx, A_elem), (b_col_idx, b_elem)
    
# end


# @doc raw"""
#     poisson_weak_form_1(inputs::WeakFormInputs{n, Frhs, Ttrial, Ttest}, element_id) where {n, Frhs, Ttrial, Ttest, TG}

# Bilinear form for the computation of the Poisson equation on the given element.

# This function computes the contribution of the given element of both the 
# bilinear and linear form for the Poisson equation. The associated weak 
# formulation is:

# For given ``f \in L^2(\Omega)``, find ``\phi \in H^1(\Omega)`` such that 
# ```math
# \int_{\Omega} \nabla \varphi \cdot \nabla \phi \;d\Omega = \int_{\Omega} \varphi f(x) \;d\Omega \quad \forall \ \varphi \in H^1(\Omega)
# ```

# Note that there are not boundary conditions specified. To solve what the 
# global assembler returns, one should add an extra condition (e.g. average 
# of ``\phi`` is zero)

# # Arguments
# - `elem_id::NTuple{n,Int}`: element for which to compute the contribution.
# """
# function l2_weak_form(inputs::WeakFormInputs{n, Frhs, Ttrial, Ttest, TG}, element_id) where {n, Frhs, Ttrial, Ttest, TG}
#     # Computed bases and their derivatives.
#     trial_basis_evals, trial_supported_bases = FunctionSpaces.evaluate(inputs.space_trial, element_id, inputs.quad_nodes, 0)
#     test_basis_evals, test_supported_bases = FunctionSpaces.evaluate(inputs.space_test, element_id, inputs.quad_nodes, 0)
    
#     # Compute the quantities related to the geometry.
#     mapped_nodes = Geometry.evaluate(inputs.geometry, element_id, inputs.quad_nodes)
#     _, _, jac_det = Geometry.inv_metric(inputs.geometry, element_id, inputs.quad_nodes)

#     # Compute rhs on mapped nodes.
#     # It can be more efficient to avoid this, but that is easier when we 
#     # have a better interface for the inner products.
#     fxy = inputs.forcing.(NTuple{n, Vector{Float64}}(mapped_nodes[:,i] for i in 1:1:n)...)
    
#     # Count the number of supported basis on this element.
#     n_supported_bases_trial = length(trial_supported_bases)
#     n_supported_bases_test = length(test_supported_bases)
#     n_supported_total = n_supported_bases_trial * n_supported_bases_test

#     # Pre-allocate the local matrices (their row, colum, and value vectors).
#     A_row_idx = Vector{Int}(undef, n_supported_total)
#     A_col_idx = Vector{Int}(undef, n_supported_total)
#     A_elem = Vector{Float64}(undef, n_supported_total)
#     b_col_idx = Vector{Int}(undef, n_supported_bases_test)
#     b_elem = Vector{Float64}(undef, n_supported_bases_test)

#     for test_linear_idx in 1:1:n_supported_bases_test

#         for trial_linear_idx in 1:1:n_supported_bases_trial
#             idx = (test_linear_idx - 1) * n_supported_bases_test + trial_linear_idx

#             A_row_idx[idx] = trial_supported_bases[trial_linear_idx]
#             A_col_idx[idx] = test_supported_bases[test_linear_idx]

#             Aij = compute_inner_product_L2(jac_det, inputs.quad_weights, 
#                                            LinearAlgebra.I, view(trial_basis_evals[n == 1 ? 0 : NTuple{n, Int}(zeros(Int, n))], :, trial_linear_idx), 
#                                            view(test_basis_evals[n == 1 ? 0 : NTuple{n, Int}(zeros(Int, n))], :, test_linear_idx))
            
#             A_elem[idx] = Aij
#         end


#         b_col_idx[test_linear_idx] = test_supported_bases[test_linear_idx]

#         bi = compute_inner_product_L2(jac_det, inputs.quad_weights, 
#                                       LinearAlgebra.I, fxy,
#                                       view(test_basis_evals[n == 1 ? 0 : NTuple{n, Int}(zeros(Int, n))], :, test_linear_idx))

#         b_elem[test_linear_idx] = bi
#     end
    
#     # The output should be the contribution to the left-hand-side matrix 
#     # A and right-hand-side vector b. The outputs are tuples of 
#     # row_indices, column_indices, values for the matrix part and 
#     # column_indices, values for the vector part.
#     return (A_row_idx, A_col_idx, A_elem), (b_col_idx, b_elem)
    
# end

