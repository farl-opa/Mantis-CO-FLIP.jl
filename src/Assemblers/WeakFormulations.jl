
import .. Geometry
import .. FunctionSpaces
import .. Forms
import .. Quadrature

import ... Main  # For testing only, to be able to use Main.@code_warntype (but not when precompiling!)





@doc raw"""
    struct WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest, G} <: AbstractInputs

Contains the required data for a one variable FEM that uses forms.

# Fields
- `forcing::Frhs <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}`: Forcing function defined as a form.
- `space_trial::Ttrial <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}`: Trial (Solution) space.
- `space_test::Ttest <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}`: Test space.
- `quad_rule::Quadrature.QuadratureRule{manifold_dim}`: Quadrature rule.
"""
struct WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest} <: AbstractInputs
    forcing::Frhs

    space_trial::Ttrial
    space_test::Ttest

    quad_rule::Quadrature.QuadratureRule{manifold_dim}

    function WeakFormInputs(forcing::Frhs, 
                            space_trial::Ttrial, 
                            space_test::Ttest,
                            quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, form_rank, 
                            G <: Geometry.AbstractGeometry{manifold_dim},
                            Frhs <: Forms.AbstractFormExpression{manifold_dim, form_rank, G},
                            Ttrial <: Forms.AbstractFormSpace{manifold_dim, form_rank, G},
                            Ttest <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}}
        
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

# function get_boundary_dof_indices(wf::WeakFormInputs)
#     return Forms.get_boundary_dof_indices(wf.space_trial)
# end


@doc raw"""
    poisson_non_mixed(inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Bilinear form for the computation of the Poisson equation on the given element.

This function computes the contribution of the given element of both the 
bilinear and linear form for the Poisson equation. The associated weak 
formulation is:

For given ``f^0 \in L^2 \Lambda^n (\Omega)``, find ``\phi^0 \in H^1_0 \Lambda^n (\Omega)`` such that 
```math
\int_{\Omega} d \phi^0 \wedge \star d \varphi^0 = -\int_{\Omega} f^0 \wedge \star \varphi^0 \quad \forall \ \varphi^0 \in H^1_0 \Lambda^0 (\Omega)
```

# Arguments
- `inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}`: weak form setup. Requires one test and one trial space for 0-forms.
- `elem_id`: element for which to compute the contribution.
"""
function poisson_non_mixed(inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    # The inner product will be between the exterior derivative of the 
    # trial zero form with the exterior derivative of the test zero 
    # form, so we compute those first.
    dtrial = Forms.exterior_derivative(inputs.space_trial)
    dtest = Forms.exterior_derivative(inputs.space_test)

    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(dtest, dtrial, element_id, inputs.quad_rule)

    # The linear form is the inner product between the trial form and 
    # the forcing function which is a form of an appropriate rank.
    b_row_idx, b_col_idx, b_elem = Forms.evaluate_inner_product(inputs.space_test, inputs.forcing, element_id, inputs.quad_rule)
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end


@doc raw"""
    l2_weak_form(inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Bilinear form for the computation of the ``L^2``-projection on the given element.

The associated weak formulation is:

For given ``f^k \in L^2 \Lambda^k (\Omega)``, find ``\phi^k \in L^2 \Lambda^k (\Omega)`` such that 
```math
\int_{\Omega} \phi^k \wedge \star \varphi^k = -\int_{\Omega} f^k \wedge \star \varphi^k \quad \forall \ \varphi^k \in L^2 \Lambda^k (\Omega)
```

# Arguments
- `inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}`: weak form setup. Requires one test and one trial space for 0-forms.
- `elem_id`: element for which to compute the contribution.
"""
function l2_weak_form(inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    # The l.h.s. is the inner product between the test and trial functions.
    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(inputs.space_test, inputs.space_trial, element_id, inputs.quad_rule)

    # The r.h.s. is the inner product between the test and forcing functions.
    b_row_idx, b_col_idx, b_elem = Forms.evaluate_inner_product(inputs.space_test, inputs.forcing, element_id, inputs.quad_rule)
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # column_indices, values for the vector part.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end










@doc raw"""
    struct WeakFormInputsMixed{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2} <: AbstractInputs

Contains the required data for a two variable (mixed) FEM that uses forms.

# Fields
- `forcing::Frhs <: Forms.AbstractFormExpression{manifold_dim, form_rank, G}`: Forcing function defined as a form.
- `space_trial_u_1::Ttrial1 <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}`: Trial (Solution) space for the (n-1)-form.
- `space_trial_phi_2::Ttrial2 <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}`: Trial (Solution) space for the n-form.
- `space_test_eps_1::Ttest1 <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}`: Test space for the (n-1)-forms.
- `space_test_eps_2::Ttest1 <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}`: Test space for the n-forms.
- `quad_rule::Quadrature.QuadratureRule{manifold_dim}`: Quadrature rule.
"""
struct WeakFormInputsMixed{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2} <: AbstractInputs
    forcing::Frhs

    space_trial_u_1::Ttrial1
    space_trial_phi_2::Ttrial2

    space_test_eps_1::Ttest1
    space_test_eps_2::Ttest2

    quad_rule::Quadrature.QuadratureRule{manifold_dim}

    function WeakFormInputsMixed(forcing::Frhs, 
                            space_trial_1::Ttrial1,
                            space_trial_2::Ttrial2, 
                            space_test_1::Ttest1,
                            space_test_2::Ttest2,
                            quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, form_rank, 
                            G <: Geometry.AbstractGeometry{manifold_dim},
                            Frhs <: Forms.AbstractFormExpression{manifold_dim, manifold_dim, G},
                            Ttrial1 <: Forms.AbstractFormSpace{manifold_dim, form_rank, G},
                            Ttrial2 <: Forms.AbstractFormSpace{manifold_dim, manifold_dim, G},
                            Ttest1 <: Forms.AbstractFormSpace{manifold_dim, form_rank, G},
                            Ttest2 <: Forms.AbstractFormSpace{manifold_dim, manifold_dim, G}}
        # Check that form_rank == manifold_dim - 1
        # We may not want to do this here, as this struct is more general than only the mixed Poisson problem.
        new{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2}(forcing, space_trial_1, space_trial_2, space_test_1, space_test_2, quad_rule)
    end
end

# Every bilinear form will need the functions defined below. These are 
# used by the global assembler to set up the problem.
function get_num_elements(wf::WeakFormInputsMixed)
    geo = Forms.get_geometry(wf.space_trial_phi_2)
    return Geometry.get_num_elements(geo)
end

function get_problem_size(wf::WeakFormInputsMixed)
    return Forms.get_num_basis(wf.space_trial_u_1) + Forms.get_num_basis(wf.space_trial_phi_2), Forms.get_num_basis(wf.space_test_eps_1) + Forms.get_num_basis(wf.space_test_eps_2)
end

function get_estimated_nnz_per_elem(wf::WeakFormInputsMixed)
    return (Forms.get_max_local_dim(wf.space_trial_u_1) + Forms.get_max_local_dim(wf.space_trial_phi_2)) * (Forms.get_max_local_dim(wf.space_test_eps_1) + Forms.get_max_local_dim(wf.space_test_eps_2)), Forms.get_max_local_dim(wf.space_test_eps_1) + Forms.get_max_local_dim(wf.space_test_eps_2)
end


@doc raw"""
    poisson_mixed(inputs::WeakFormInputsMixed{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2}, element_id) where {manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2}

Bilinear form for the computation of the mixed Poisson equation on the given element.

This function computes the contribution of the given element of both the 
bilinear and linear form for the Poisson equation. The associated weak 
formulation is:

For given ``f^n \in L^2 \Lambda^n (\Omega)``, find ``u^{n-1} \in H(div, \Omega) \Lambda^{n-1} (\Omega)`` and ``\phi^n \in L^2 \Lambda^n (\Omega)`` such that 
```math
\langle \varepsilon^{n-1}, u^{n-1} \rangle - \langle d \varepsilon^{n-1}, \phi^n \rangle = 0 \quad \forall \ \varepsilon^{n-1} \in H(div, \Omega) \Lambda^{n-1} (\Omega) \\
\langle \varepsilon^n, d u^{n-1} \rangle = -\langle \varepsilon^n f^n \rangle \quad \forall \ \varepsilon^n \in L^2 \Lambda^n (\Omega)
```

# Arguments
- `inputs::WeakFormInputs{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2}`: weak form setup. Must have a test and a trial space for both n- and (n-1)-forms.
- `element_id`: element for which to compute the contribution.
"""
function poisson_mixed(inputs::WeakFormInputsMixed{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2}, element_id) where {manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2}
    # Left hand side.
    # <ε¹, u¹>
    A_row_idx_11, A_col_idx_11, A_elem_11 = Forms.evaluate_inner_product(inputs.space_test_eps_1, inputs.space_trial_u_1, element_id, inputs.quad_rule)

    # <dε¹, ϕ²>
    A_row_idx_12, A_col_idx_12, A_elem_12 = Forms.evaluate_inner_product(Forms.exterior_derivative(inputs.space_test_eps_1), inputs.space_trial_phi_2, element_id, inputs.quad_rule)

    # <ε², du¹>
    A_row_idx_21, A_col_idx_21, A_elem_21 = Forms.evaluate_inner_product(inputs.space_test_eps_2, Forms.exterior_derivative(inputs.space_trial_u_1), element_id, inputs.quad_rule)

    # The remain term, A22, is zero, so not computed.

    # Add offsets.
    A_row_idx_21 .+= Forms.get_num_basis(inputs.space_test_eps_1)

    A_col_idx_12 .+= Forms.get_num_basis(inputs.space_trial_u_1)

    # Put all variables together.
    A_row_idx = vcat(A_row_idx_11, A_row_idx_12, A_row_idx_21)
    A_col_idx = vcat(A_col_idx_11, A_col_idx_12, A_col_idx_21)
    A_elem = vcat(A_elem_11, -A_elem_12, A_elem_21)


    # Right hand side. Only the second part is non-zero.
    # <ε², f²>
    b_row_idx, _, b_elem = Forms.evaluate_inner_product(inputs.space_test_eps_2, inputs.forcing, element_id, inputs.quad_rule)
    b_elem .*= -1.0
    
    b_row_idx .+= Forms.get_num_basis(inputs.space_test_eps_1)

    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end






