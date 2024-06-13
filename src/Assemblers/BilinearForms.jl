
import .. Geometry
import .. FunctionSpaces
import .. Quadrature

import ... Main  # For testing only, to be able to use Main.@code_warntype (but not when precompiling!)





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
[`Quadrature.tensor_product_rule(p::NTuple{n, Int}, quad_rule::F) where {n, F <: Function}`](@ref) to compute the quadrature weights.
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

    # Not exactly elegant, but we need a way to easily and stably 
    # extract things like a gradient from the evaluations. The filter 
    # iterator only iterates over the keys for which the sum is 
    # 1 (so all partial derivatives of order 1).
    key_itr_test = sort(collect(Iterators.filter(x -> sum(x) == 1 ? true : false, keys(test_basis_evals))), rev=true)
    key_itr_trial = sort(collect(Iterators.filter(x -> sum(x) == 1 ? true : false, keys(trial_basis_evals))), rev=true)

    for test_linear_idx in 1:1:n_supported_bases_test

        # See the comment below.
        grad_test = NTuple{n, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}}(view(test_basis_evals[key], :, test_linear_idx) for key in key_itr_test)

        for trial_linear_idx in 1:1:n_supported_bases_trial
            idx = (test_linear_idx - 1) * n_supported_bases_test + trial_linear_idx

            A_row_idx[idx] = trial_supported_bases[trial_linear_idx]
            A_col_idx[idx] = test_supported_bases[test_linear_idx]

            # Not exactly elegant. In order to make this type stable, I 
            # made sure that the size of the tuple is known (there are n 
            # partial derivatives of interest in dimension n) and I 
            # specified the type of the SubArray.
            grad_trail = NTuple{n, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}}(view(trial_basis_evals[key], :, trial_linear_idx) for key in key_itr_trial)
            

            Aij = compute_inner_product_L2(jac_det, self.quad_weights, 
                                           metric_inv, grad_trail, grad_test)
            
            A_elem[idx] = Aij
        end


        b_col_idx[test_linear_idx] = test_supported_bases[test_linear_idx]

        bi = compute_inner_product_L2(jac_det, self.quad_weights, 
                                      LinearAlgebra.I, fxy,
                                      view(test_basis_evals[n == 1 ? 0 : NTuple{n, Int}(zeros(Int, n))], :, test_linear_idx))

        b_elem[test_linear_idx] = bi
    end
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # column_indices, values for the vector part.
    return (A_row_idx, A_col_idx, A_elem), (b_col_idx, b_elem)
    
end

