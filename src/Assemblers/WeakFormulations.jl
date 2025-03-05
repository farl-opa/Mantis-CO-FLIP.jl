
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
#                                       Mixed Inputs                                       #
############################################################################################

"""
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
struct WeakFormInputsMixed{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2} <:
       AbstractInputs
    forcing::Frhs

    space_trial_u_1::Ttrial1
    space_trial_phi_2::Ttrial2

    space_test_eps_1::Ttest1
    space_test_eps_2::Ttest2

    quad_rule::Quadrature.QuadratureRule{manifold_dim}

    function WeakFormInputsMixed(
        forcing::Frhs,
        space_trial_1::Ttrial1,
        space_trial_2::Ttrial2,
        space_test_1::Ttest1,
        space_test_2::Ttest2,
        quad_rule::Quadrature.QuadratureRule{manifold_dim},
    ) where {
        manifold_dim,
        form_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        Frhs <: Forms.AbstractFormExpression{manifold_dim, manifold_dim, G},
        Ttrial1 <: Forms.AbstractFormSpace{manifold_dim, form_rank, G},
        Ttrial2 <: Forms.AbstractFormSpace{manifold_dim, manifold_dim, G},
        Ttest1 <: Forms.AbstractFormSpace{manifold_dim, form_rank, G},
        Ttest2 <: Forms.AbstractFormSpace{manifold_dim, manifold_dim, G},
    }
        # Check that form_rank == manifold_dim - 1
        # We may not want to do this here, as this struct is more general than only the mixed Poisson problem.
        return new{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2}(
            forcing, space_trial_1, space_trial_2, space_test_1, space_test_2, quad_rule
        )
    end

    # Constructor for 1-form mixed problem
    function WeakFormInputsMixed(
        forcing::Frhs,
        space_trial_1::Ttrial1,
        space_trial_2::Ttrial2,
        space_test_1::Ttest1,
        space_test_2::Ttest2,
        quad_rule::Quadrature.QuadratureRule{manifold_dim},
    ) where {
        manifold_dim,
        G <: Geometry.AbstractGeometry{manifold_dim},
        Frhs <: Forms.AbstractFormExpression{manifold_dim, 1, G},
        Ttrial1 <: Forms.AbstractFormSpace{manifold_dim, 0, G},
        Ttrial2 <: Forms.AbstractFormSpace{manifold_dim, 1, G},
        Ttest1 <: Forms.AbstractFormSpace{manifold_dim, 0, G},
        Ttest2 <: Forms.AbstractFormSpace{manifold_dim, 1, G},
    }
        # We may not want to do this here, as this struct is more general than only the mixed Poisson problem.
        return new{manifold_dim, Frhs, Ttrial1, Ttrial2, Ttest1, Ttest2}(
            forcing, space_trial_1, space_trial_2, space_test_1, space_test_2, quad_rule
        )
    end
end

# Every bilinear form will need the functions defined below. These are 
# used by the global assembler to set up the problem.
function get_num_elements(wf::WeakFormInputsMixed)
    geo = Forms.get_geometry(wf.space_trial_phi_2)
    return Geometry.get_num_elements(geo)
end

function get_problem_size(wf::WeakFormInputsMixed)
    return Forms.get_num_basis(wf.space_trial_u_1) +
           Forms.get_num_basis(wf.space_trial_phi_2),
    Forms.get_num_basis(wf.space_test_eps_1) + Forms.get_num_basis(wf.space_test_eps_2)
end

function get_estimated_nnz_per_elem(wf::WeakFormInputsMixed)
    return (
        Forms.get_max_local_dim(wf.space_trial_u_1) +
        Forms.get_max_local_dim(wf.space_trial_phi_2)
    ) * (
        Forms.get_max_local_dim(wf.space_test_eps_1) +
        Forms.get_max_local_dim(wf.space_test_eps_2)
    ),
    Forms.get_max_local_dim(wf.space_test_eps_1) +
    Forms.get_max_local_dim(wf.space_test_eps_2)
end

############################################################################################
#                                        Eigenvalue                                        #
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
