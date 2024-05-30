
import .. Geometry
import .. FunctionSpaces
import .. Quadrature

import ... Main



# # We can consider parameterising on the dimension here as well.
# struct PoissonBilinearForm1D{Frhs, Fbilinear, Flinear, Ttrial, Ttest, TG} <: AbstractBilinearForms
#     forcing::Frhs

#     bilinear_form_function::Fbilinear
#     linear_form_function::Flinear

#     space_trial::Ttrial
#     space_test::Ttest

#     geometry::TG

#     quad_nodes::Vector{Float64}
#     quad_weights::Vector{Float64}
# end

# # Every bilinear form will need the functions defined below. These are 
# # used by the global assembler to set up the problem.
# function get_num_elements(PB::PoissonBilinearForm1D)
#     return Geometry.get_num_elements(PB.geometry)
# end

# function get_problem_size(PB::PoissonBilinearForm1D)
#     return FunctionSpaces.get_dim(PB.space_trial), FunctionSpaces.get_dim(PB.space_test)
# end

# function get_estimated_nnz_per_elem(PB::PoissonBilinearForm1D)
#     return FunctionSpaces.get_max_local_dim(PB.space_trial) * FunctionSpaces.get_max_local_dim(PB.space_test), FunctionSpaces.get_max_local_dim(PB.space_test)
# end

# @doc raw"""
#     (PB::PoissonBilinearForm1D{Frhs, Fbilinear, Flinear, Ttrial, Ttest, TG} where {Frhs, Fbilinear, Flinear, Ttrial, Ttest, TG})(element_id) 

# Bilinear form for the computation of the 1D Poisson equation on the given element.

# This function computes the contribution of the given element of both the 
# bilinear and linear form for the 1D Poisson equation. The associated 
# weak formulation is:

# For given ``f \in L^2(\Omega)``, find ``\phi \in H^1(\Omega)`` such that 
# ```math
# \int_{L_{left}}^{L_{right}} \frac{d \varphi}{d x} \frac{d \phi}{d x} \;dx = \int_{L_{left}}^{L_{right}} \varphi f(x) \;dx \quad \forall \ \varphi \in H_0^1(\Omega)
# ```

# Note that the global assembler is currently taking care of the boundary 
# contributions, and it ensures that the non-zero test functions on the 
# boundary are removed.

# # Arguments
# - `elem_id::Int`: element for which to compute the contribution.
# """
# function (PB::PoissonBilinearForm1D{Frhs, Fbilinear, Flinear, Ttrial, Ttest, TG} where {Frhs, Fbilinear, Flinear, Ttrial, Ttest, TG})(element_id) 
#     # Computed bases and their derivatives on the current element and in 
#     # the physical domain, so no need to transform them! Because the 
#     # mesh is chosen such that all elements are of length 1, the 
#     # transformations do not change the results. A metric term is 
#     # currently included to show how this can be approached, but it too does not change the result.
#     trial_basis_evals, trial_supported_bases = FunctionSpaces.evaluate(PB.space_trial, element_id, (PB.quad_nodes,), 1)
#     test_basis_evals, test_supported_bases = FunctionSpaces.evaluate(PB.space_test, element_id, (PB.quad_nodes,), 1)
    
#     # Compute the quantities related to the geometry. We use the metric 
#     # to prevent the need to invert the Jacobian.
#     mapped_nodes = Geometry.evaluate(PB.geometry, element_id, (PB.quad_nodes,))
#     metric_inv, jac_det, metric = Geometry.inv_metric(PB.geometry, element_id, (PB.quad_nodes,))
#     # jacobian = Geometry.jacobian(PB.geometry, element_id, (PB.quad_nodes,))
#     # #metric_inv = inv(transpose(jacobian) * jacobian)
#     # metric_inv = 1.0 ./ (jacobian .* jacobian)

#     mapped_weights = jac_det .* PB.quad_weights
    
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

#             Aij = compute_inner_product_L2(mapped_nodes, 
#                                            mapped_weights, 
#                                            view(metric_inv .* trial_basis_evals[1], :, trial_linear_idx), 
#                                            view(test_basis_evals[1], :, test_linear_idx))
            
#             A_elem[idx] = Aij
#         end


#         b_col_idx[test_linear_idx] = test_supported_bases[test_linear_idx]

#         bi = compute_inner_product_L2(mapped_nodes, 
#                                       mapped_weights,
#                                       PB.forcing, 
#                                       view(test_basis_evals[0], :, test_linear_idx))
        
#         b_elem[test_linear_idx] = bi
#     end
    
#     # The output should be the contribution to the left-hand-side matrix 
#     # A and right-hand-side vector b. The outputs are tuples of 
#     # row_indices, column_indices, values for the matrix part and 
#     # column_indices, values for the vector part.
#     return (A_row_idx, A_col_idx, A_elem), (b_col_idx, b_elem)
    
# end







# Example only, not used:

# # Mixed 1D Poisson problem.
# struct PoissonBilinearFormMixed1D{Frhs, Fbilinear1, Fbilinear2, Fbilinear3, Flinear, Ttrialphi, Ttestphi, Ttrialsigma, Ttestsigma, TG} <: AbstractBilinearForms
#     forcing::Frhs

#     bilinear_form_tausigma_function::Fbilinear1
#     bilinear_form_dtausigma_function::Fbilinear2
#     bilinear_form_phidsigma_function::Fbilinear3
#     linear_form_function::Flinear

#     space_trial_phi::Ttrialphi
#     space_test_phi::Ttestphi
#     space_trial_sigma::Ttrialsigma
#     space_test_sigma::Ttestsigma

#     geometry::TG

#     quad_nodes::Vector{Float64}
#     quad_weights::Vector{Float64}
# end

# function get_num_elements(PB::PoissonBilinearFormMixed1D)
#     return Geometry.get_num_elements(PB.geometry)
# end

# function get_problem_size(PB::PoissonBilinearFormMixed1D)
#     return (FunctionSpaces.get_dim(PB.space_trial_sigma) + FunctionSpaces.get_dim(PB.space_trial_phi), 
#             FunctionSpaces.get_dim(PB.space_test_sigma) + FunctionSpaces.get_dim(PB.space_test_phi))
# end

# function get_estimated_nnz_per_elem(PB::PoissonBilinearFormMixed1D)
#     nnz_A = ((FunctionSpaces.get_max_local_dim(PB.space_trial_sigma) + FunctionSpaces.get_max_local_dim(PB.space_trial_phi)) 
#              * (FunctionSpaces.get_max_local_dim(PB.space_test_sigma) + FunctionSpaces.get_max_local_dim(PB.space_test_phi)))
#     nnz_b = FunctionSpaces.get_max_local_dim(PB.space_test_sigma) + FunctionSpaces.get_max_local_dim(PB.space_test_phi)
#     return nnz_A, nnz_b
# end

# @doc raw"""
#     (PB::PoissonBilinearFormMixed1D{Frhs, Fbilinear1, Fbilinear2, Fbilinear3, Flinear, Ttrialphi, Ttestphi, Ttrialsigma, Ttestsigma, TG} where {Frhs, Fbilinear1, Fbilinear2, Fbilinear3, Flinear, Ttrialphi, Ttestphi, Ttrialsigma, Ttestsigma, TG})(element_id) 

# Bilinear form for the computation of the 1D mixed Poisson equation on the given element.

# This function computes the contribution of the given element of both the 
# bilinear and linear form for the 1D Mixed Poisson equation. The 
# associated weak formulation is:

# For given ``f \in L^2(\Omega)``, find ``(\sigma, \phi) \in H^1(\Omega) \times L^2(\Omega)`` such that 
# ```math
# \begin{align*}
# \int_{L_{left}}^{L_{right}} \tau \sigma \;dx - \int_{L_{left}}^{L_{right}} \frac{d \tau}{d x} \phi \;dx = 0 \quad & \forall \ \tau \in H^1(\Omega) \\
# \int_{L_{left}}^{L_{right}} \varphi \frac{d \sigma}{d x} \;dx = \int_{L_{left}}^{L_{right}} \varphi f(x) \;dx \quad & \forall \ \varphi \in L^2(\Omega)
# \end{align*}
# ```

# # Arguments
# - `elem_id::Int`: element for which to compute the contribution.
# """
# function (PB::PoissonBilinearFormMixed1D{Frhs, Fbilinear1, Fbilinear2, Fbilinear3, Flinear, Ttrialphi, Ttestphi, Ttrialsigma, Ttestsigma, TG} where {Frhs, Fbilinear1, Fbilinear2, Fbilinear3, Flinear, Ttrialphi, Ttestphi, Ttrialsigma, Ttestsigma, TG})(element_id) 
#     # Computed bases and their derivatives on the current element and in 
#     # the physical domain, so no need to transform them! Because the 
#     # mesh is chosen such that all elements are of length 1, the 
#     # transformations do not change the results. A metric term is 
#     # currently included to show how this can be approached, but it too does not change the result.
#     trial_basis_phi_evals, trial_supported_bases_phi = FunctionSpaces.evaluate(PB.space_trial_phi, element_id, (PB.quad_nodes,), 1)
#     test_basis_phi_evals, test_supported_bases_phi = FunctionSpaces.evaluate(PB.space_test_phi, element_id, (PB.quad_nodes,), 1)

#     trial_basis_sigma_evals, trial_supported_bases_sigma = FunctionSpaces.evaluate(PB.space_trial_sigma, element_id, (PB.quad_nodes,), 1)
#     test_basis_sigma_evals, test_supported_bases_sigma = FunctionSpaces.evaluate(PB.space_test_sigma, element_id, (PB.quad_nodes,), 1)

#     offset_phi = FunctionSpaces.get_dim(PB.space_trial_sigma)
    
#     # Compute the quantities related to the geometry.
#     mapped_nodes = Geometry.evaluate(PB.geometry, element_id, (PB.quad_nodes,))
#     jacobian = Geometry.jacobian(PB.geometry, element_id, (PB.quad_nodes,))
#     jacobian_inv = 1.0 ./ jacobian

#     mapped_weights = jacobian .* PB.quad_weights
    
#     # Count the number of supported basis on this element.
#     n_supported_bases_trial = length(trial_supported_bases_sigma) + length(trial_supported_bases_phi)
#     n_supported_bases_test = length(test_supported_bases_sigma) + length(test_supported_bases_phi)
#     n_supported_total = n_supported_bases_trial * n_supported_bases_test

#     # Pre-allocate the local matrices (their row, colum, and value vectors).
#     # These are all possible non-zero values. Because of the structure 
#     # of the mixed problem, we know that one of the blocks will be a 
#     # zero block. If we specify the problem like this, we have a 
#     # explicitly add zeros to the block matrix or the global assembler 
#     # will miscount and they will still be undef values in our result. 
#     A_row_idx = Vector{Int}(undef, n_supported_total)
#     A_col_idx = Vector{Int}(undef, n_supported_total)
#     A_elem = Vector{Float64}(undef, n_supported_total)
#     b_col_idx = Vector{Int}(undef, n_supported_bases_test)
#     b_elem = Vector{Float64}(undef, n_supported_bases_test)

#     for test_linear_idx in 1:1:n_supported_bases_test

#         for trial_linear_idx in 1:1:n_supported_bases_trial
#             idx = (test_linear_idx - 1) * n_supported_bases_test + trial_linear_idx

#             if trial_linear_idx <= length(trial_supported_bases_sigma) && test_linear_idx <= length(test_supported_bases_sigma)
#                 # First bilinear form with tau and sigma
#                 A_row_idx[idx] = test_supported_bases_sigma[test_linear_idx]
#                 A_col_idx[idx] = trial_supported_bases_sigma[trial_linear_idx]

#                 Aij = compute_inner_product_L2(mapped_nodes, 
#                                                mapped_weights, 
#                                                view(test_basis_sigma_evals[0], :, test_linear_idx), 
#                                                view(trial_basis_sigma_evals[0], :, trial_linear_idx))
                
#                 A_elem[idx] = Aij
#             elseif trial_linear_idx > length(trial_supported_bases_sigma) && test_linear_idx <= length(test_supported_bases_sigma)
#                 # Second bilinear form with dtau and phi
#                 A_row_idx[idx] = test_supported_bases_sigma[test_linear_idx]
#                 A_col_idx[idx] = trial_supported_bases_phi[trial_linear_idx - length(trial_supported_bases_sigma)] + offset_phi

#                 Aij = -compute_inner_product_L2(mapped_nodes, 
#                                                 mapped_weights, 
#                                                 view(trial_basis_phi_evals[0], :, trial_linear_idx - length(trial_supported_bases_sigma)), 
#                                                 view(jacobian_inv .* test_basis_sigma_evals[1], :, test_linear_idx))
                
#                 A_elem[idx] = Aij
#             elseif trial_linear_idx <= length(trial_supported_bases_sigma) && test_linear_idx > length(test_supported_bases_sigma)
#                 # Third bilinear form with varphi and dsigma
#                 A_row_idx[idx] = test_supported_bases_phi[test_linear_idx - length(test_supported_bases_sigma)] + offset_phi
#                 A_col_idx[idx] = trial_supported_bases_sigma[trial_linear_idx]

#                 Aij = compute_inner_product_L2(mapped_nodes, 
#                                                mapped_weights, 
#                                                view(jacobian_inv .* trial_basis_sigma_evals[1], :, trial_linear_idx), 
#                                                view(test_basis_phi_evals[0], :, test_linear_idx - length(test_supported_bases_sigma)))
                
#                 A_elem[idx] = Aij
            
#             else
#                 # Fourth bilinear form which is identically zero.
#                 # For now, these must be specified or the global 
#                 # assembler will get it wrong.
#                 A_row_idx[idx] = test_supported_bases_phi[test_linear_idx - length(test_supported_bases_sigma)] + offset_phi
#                 A_col_idx[idx] = trial_supported_bases_phi[trial_linear_idx - length(trial_supported_bases_sigma)] + offset_phi
                
#                 A_elem[idx] = 0.0
#             end
#         end


#         if test_linear_idx <= length(test_supported_bases_sigma)
#             # R.h.s. of the first equation is 0.
#             b_col_idx[test_linear_idx] = test_supported_bases_sigma[test_linear_idx]
#             b_elem[test_linear_idx] = 0.0
#         else
#             # R.h.s. of the second equation is given by the L^2 inner 
#             # product with the forcing function.
#             b_col_idx[test_linear_idx] = test_supported_bases_phi[test_linear_idx - length(test_supported_bases_sigma)] + offset_phi

#             bi = compute_inner_product_L2(mapped_nodes, 
#                                           mapped_weights,
#                                           PB.forcing, 
#                                           view(test_basis_phi_evals[0], :, test_linear_idx-length(test_supported_bases_sigma)))
            
#             b_elem[test_linear_idx] = bi
#         end
#     end
    
#     # The output should be the contribution to the left-hand-side matrix 
#     # A and right-hand-side vector b. The outputs are tuples of 
#     # row_indices, column_indices, values for the matrix part and 
#     # column_indices, values for the vector part.
#     return (A_row_idx, A_col_idx, A_elem), (b_col_idx, b_elem)
    
# end










"""
    struct PetrovGalerkinOneVariable{n, Frhs, Ttrial, Ttest, TG} <: AbstractBilinearForms

Contains the required data for a Petrov-Galerkin FEM for one variable.

# Fields
- `forcing::Frhs <: Function`: Forcing function, must take in an NTuple{n, Float64} as argument.
- `space_trial::Ttrial <: FunctionSpaces.AbstractFunctionSpace`: Trial (Solution) space.
- `space_test::Ttest <: FunctionSpaces.AbstractFunctionSpace`: Test space.
- `geometry::TG <: Geometry.AbstractGeometry{n,m}`: Geometry.
- `quad_nodes::NTuple{n, Vector{Float64}}`: Quadrature nodes.
- `quad_weights::Vector{Float64}`: Vector of the tensor product of the quadrature weights.

# See also
[`tensor_product_weights(weights_1d::NTuple{n, Vector{Float64}}) where {n}`](@ref) to compute the quadrature weights.
"""
struct PoissonBilinearForm{n, m, Frhs, Ttrial, Ttest, TG} <: AbstractBilinearForms
    forcing::Frhs

    space_trial::Ttrial
    space_test::Ttest

    geometry::TG

    # Quadrature rule in reference domain. Nodes per dimension, weights 
    # as returned from Mantis.Quadrature.tensor_product_weights
    quad_nodes::NTuple{n, Vector{Float64}}
    quad_weights::Vector{Float64}

    function PoissonBilinearForm(forcing::Frhs, 
                                 space_trial::Ttrial, 
                                 space_test::Ttest, 
                                 geometry::TG, 
                                 quad_nodes::NTuple{n, Vector{Float64}}, 
                                 quad_weights::Vector{Float64}) where {n, m, 
                                 Frhs <: Function, 
                                 Ttrial <: FunctionSpaces.AbstractFunctionSpace, 
                                 Ttest <: FunctionSpaces.AbstractFunctionSpace,
                                 TG <: Geometry.AbstractGeometry{n,m}}
        
        new{n, m, Frhs, Ttrial, Ttest, TG}(forcing, space_trial, space_test, 
                                           geometry, quad_nodes, quad_weights)
    end

    # Constructor for the standard Galerkin setup. Essential the same as above, 
    # but the test and trial spaces are of the same type.
    function Galerkin(forcing::Frhs, 
                      space_trial::Tspace, 
                      space_test::Tspace, 
                      geometry::TG, 
                      quad_nodes::NTuple{n, Vector{Float64}}, 
                      quad_weights::Vector{Float64}) where {n, 
                      Frhs <: Function, 
                      Tspace <: FunctionSpaces.AbstractFunctionSpace, 
                      TG <: Geometry.AbstractGeometry{n,m} where {m}}

        new{n, m, Frhs, Tspace, Tspace, TG}(forcing, space_trial, space_test, 
                                           geometry, quad_nodes, quad_weights)
    end
end

# If the names of the variables in the struct are consistent across 
# multiple bilinear forms, we may be able to define some of these 
# functions below (especially the number of elements) for the general 
# case. This will be less error prone, but can fail if the geometry 
# variable name in the struct is not called geometry.

# Every bilinear form will need the functions defined below. These are 
# used by the global assembler to set up the problem.
function get_num_elements(PB::PoissonBilinearForm)
    return Geometry.get_num_elements(PB.geometry)
end

function get_problem_size(PB::PoissonBilinearForm)
    return FunctionSpaces.get_dim(PB.space_trial), FunctionSpaces.get_dim(PB.space_test)
end

function get_estimated_nnz_per_elem(PB::PoissonBilinearForm)
    return FunctionSpaces.get_max_local_dim(PB.space_trial) * FunctionSpaces.get_max_local_dim(PB.space_test), FunctionSpaces.get_max_local_dim(PB.space_test)
end


# Maybe we should turn this into a function that takes something like 
# the above struct as input. Then, depending the struct (= type) of 
# this input, it can change its behaviour. I think this may reduce the 
# number of structs that must be made, as they can be more universal. 
# This would also improve the names I think.
@doc raw"""
    (PB::PoissonBilinearForm{n, Frhs, Ttrial, Ttest, TG} where {n, Frhs, Ttrial, Ttest, TG})(element_id) 

Bilinear form for the computation of the Poisson equation on the given element.

This function computes the contribution of the given element of both the 
bilinear and linear form for the Poisson equation. The associated weak 
formulation is:

For given ``f \in L^2(\Omega)``, find ``\phi \in H^1(\Omega)`` such that 
```math
\int_{\Omega} \nabla \varphi \cdot \nabla \phi \;d\Omega = \int_{\Omega} \varphi f(x) \;d\Omega \quad \forall \ \varphi \in H^1(\Omega)
```

Note that there are not boundary conditions specified. To solve what the 
global assembler returns, one should add an extra condition (e.g. average 
of ``\phi`` is zero)

# Arguments
- `elem_id::NTuple{n,Int}`: element for which to compute the contribution.
"""
function (self::PoissonBilinearForm{n, Frhs, Ttrial, Ttest, TG})(element_id) where {n, Frhs, Ttrial, Ttest, TG}
    # Computed bases and their derivatives.
    trial_basis_evals, trial_supported_bases = FunctionSpaces.evaluate(self.space_trial, element_id, self.quad_nodes, 1)
    test_basis_evals, test_supported_bases = FunctionSpaces.evaluate(self.space_test, element_id, self.quad_nodes, 1)
    
    # Compute the quantities related to the geometry.
    mapped_nodes = Geometry.evaluate(self.geometry, element_id, self.quad_nodes)
    metric_inv, metric, jac_det = Geometry.inv_metric(self.geometry, element_id, self.quad_nodes)

    # Compute rhs on mapped nodes.
    # It can be more efficient to avoid this, but that is easier when we 
    # have a better interface for the inner products.
    fxy = self.forcing.(NTuple{n, Vector{Float64}}(mapped_nodes[:,i] for i in 1:1:n)...)
    
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

        # See the comment below.
        key_itr_test = Iterators.filter(x -> sum(x) == 1 ? true : false, keys(trial_basis_evals))
        grad_test = NTuple{n, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}(view(test_basis_evals[key], :, test_linear_idx) for key in key_itr_test)

        for trial_linear_idx in 1:1:n_supported_bases_trial
            idx = (test_linear_idx - 1) * n_supported_bases_test + trial_linear_idx

            A_row_idx[idx] = trial_supported_bases[trial_linear_idx]
            A_col_idx[idx] = test_supported_bases[test_linear_idx]

            # Not exactly elegant, but we need a way to easily and stably 
            # extract things like a gradient from the evaluations. The filter 
            # iterator only iterates over the keys for which the sum is 
            # 1 (so all partial derivatives of order 1). In order to make 
            # this type stable, I made sure that the size of the tuple is 
            # known (there are n partial derivatives of interest in 
            # dimension n) and I specified the type of the SubArray.
            key_itr_trial = Iterators.filter(x -> sum(x) == 1 ? true : false, keys(trial_basis_evals))
            grad_trail = NTuple{n, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}(view(trial_basis_evals[key], :, trial_linear_idx) for key in key_itr_trial)
            

            Aij = compute_vec_inner_product_L2(jac_det, self.quad_weights, 
                                               metric_inv,
                                               grad_trail, 
                                               grad_test)
            
            A_elem[idx] = Aij
        end


        b_col_idx[test_linear_idx] = test_supported_bases[test_linear_idx]

        bi = compute_vec_inner_product_L2(jac_det, self.quad_weights,
                                          LinearAlgebra.I,
                                          fxy,
                                          view(test_basis_evals[n == 1 ? 0 : NTuple{n, Int}(zeros(Int, n))], :, test_linear_idx))

        b_elem[test_linear_idx] = bi
    end
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # column_indices, values for the vector part.
    return (A_row_idx, A_col_idx, A_elem), (b_col_idx, b_elem)
    
end

