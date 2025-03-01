
@doc raw"""
    struct WeakFormInputs{manifold_dim, Frhs, Ttrial, Ttest, G} <: AbstractInputs

# Fields
- `forcing::Frhs`: The forcing `Forms.MixedFormField` of the weak form.
- `space_trial::Ttrial`: The trial `Forms.MixedFormSpace` of the weak form.
- `space_test::Ttest`: The test `Forms.MixedFormSpace` of the weak form.
- `quad_rule::Quadrature.QuadratureRule{manifold_dim}`: The quadrature rule used to evaluate the weak form.
"""
struct WeakFormInputs{manifold_dim, num_forms, Frhs, Ttrial, Ttest} <: AbstractInputs
    forcing::Frhs

    space_trial::Ttrial
    space_test::Ttest

    quad_rule::Quadrature.QuadratureRule{manifold_dim}

    function WeakFormInputs(forcing::Frhs, 
                            space_trial::Ttrial, 
                            space_test::Ttest,
                            quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, num_forms, F1, F2, F3,
                            Frhs <: Forms.MixedFormField{num_forms, F1},
                            Ttrial <: Forms.MixedFormSpace{num_forms, F2},
                            Ttest <: Forms.MixedFormSpace{num_forms, F3}}
        
        new{manifold_dim, num_forms, Frhs, Ttrial, Ttest}(forcing, space_trial, space_test, quad_rule)
    end

    # Convenience constructor for the Galerkin case, i.e., when the trial and test spaces are the same.
    function WeakFormInputs(forcing::Frhs, 
            space::T, 
            quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, num_forms, F1, F2,
            Frhs <: Forms.MixedFormField{num_forms, F1},
            T <: Forms.MixedFormSpace{num_forms, F2}}

        WeakFormInputs(forcing, space, space, quad_rule)
    end

    # Convenience constructor for non-mixed problems.
    function WeakFormInputs(forcing::Frhs, 
            space_trial::Ttrial, 
            space_test::Ttest, 
            quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, form_rank, expression_rank,
            G <: Geometry.AbstractGeometry{manifold_dim},
            Frhs <: Forms.AbstractFormField{manifold_dim, form_rank, expression_rank, G},
            Ttrial <: Forms.AbstractFormSpace{manifold_dim, form_rank, G},
            Ttest <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}}

        WeakFormInputs(forcing, Forms.MixedFormSpace((space_trial,)), Forms.MixedFormSpace((space_test,)), quad_rule)
    end

    # Convenience constructor for non-mixed Galerkin problems
    function WeakFormInputs(forcing::Frhs, 
            space::T, 
            quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, form_rank, expression_rank,
            G <: Geometry.AbstractGeometry{manifold_dim},
            Frhs <: Forms.AbstractFormField{manifold_dim, form_rank, expression_rank, G},
            T <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}}

        WeakFormInputs(Forms.MixedFormField((forcing,)), Forms.MixedFormSpace((space,)), Forms.MixedFormSpace((space,)), quad_rule)
    end
end

# Every bilinear form will need the functions defined below. These are 
# used by the global assembler to set up the problem.
function get_num_elements(wf::WeakFormInputs)
    return Forms.get_num_elements(wf.space_trial)
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

############################################################################################
#                                    Maxwell Eigenvalue                                    #
############################################################################################

struct EigenvalueWeakFormInputs{manifold_dim, Ttrial, Ttest} <: AbstractInputs
    space_trial::Ttrial
    space_test::Ttest

    quad_rule::Quadrature.QuadratureRule{manifold_dim}

    function EigenvalueWeakFormInputs(space_trial::Ttrial, 
                            space_test::Ttest,
                            quad_rule::Quadrature.QuadratureRule{manifold_dim}) where {manifold_dim, form_rank, 
                            G <: Geometry.AbstractGeometry{manifold_dim},
                            Ttrial <: Forms.AbstractFormSpace{manifold_dim, form_rank, G},
                            Ttest <: Forms.AbstractFormSpace{manifold_dim, form_rank, G}}
        
        new{manifold_dim, Ttrial, Ttest}(space_trial, space_test, quad_rule)
    end
end

function get_num_elements(wf::EigenvalueWeakFormInputs)
    geo = Forms.get_geometry(wf.space_trial)
    return Geometry.get_num_elements(geo)
end

function get_problem_size(wf::EigenvalueWeakFormInputs)
    return Forms.get_num_basis(wf.space_trial), Forms.get_num_basis(wf.space_test)
end

function get_estimated_nnz_per_elem(wf::EigenvalueWeakFormInputs)
    return Forms.get_max_local_dim(wf.space_trial) * Forms.get_max_local_dim(wf.space_test), Forms.get_max_local_dim(wf.space_trial) * Forms.get_max_local_dim(wf.space_test)
end

function maxwell_eigenvalue(inputs::EigenvalueWeakFormInputs{manifold_dim, Ttrial, Ttest}, element_id) where {manifold_dim, Ttrial, Ttest}
    # The weak form in question is:
    #  ⟨curl u, curl v⟩ = ω²⟨u,v⟩, ∀v ∈ H(curl; Ω).   

    # Left-hand side
    # A term = ⟨curl u, curl v⟩
    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(Forms.exterior_derivative(inputs.space_trial), Forms.exterior_derivative(inputs.space_test), element_id, inputs.quad_rule)
    
    # Right-hand side
    # B term = ⟨u,v⟩
    B_row_idx, B_col_idx, B_elem = Forms.evaluate_inner_product(inputs.space_trial, inputs.space_test, element_id, inputs.quad_rule)
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (B_row_idx, B_col_idx, B_elem)
end
