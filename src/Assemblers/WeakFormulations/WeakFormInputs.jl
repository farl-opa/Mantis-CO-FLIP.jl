struct WeakFormInputs{manifold_dim, Q, TrF, TeF, F} <: AbstractInputs
    quad_rule::Q
    trial_forms::TrF
    test_forms::TeF
    forcing::F

    function WeakFormInputs(
        quad_rule::Q,
        trial_forms::TrF,
        test_forms::TeF,
        forcing::F,
    ) where {
        manifold_dim,
        num_TrF,
        num_TeF,
        num_F,
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        TeF <: NTuple{num_TeF, Forms.AbstractFormSpace{manifold_dim}},
        F <: NTuple{num_F, Forms.AbstractFormField{manifold_dim}},
    }
        return new{manifold_dim, Q, TrF, TeF, F}(quad_rule, trial_forms, test_forms, forcing)
    end

    function WeakFormInputs(
        quad_rule::Q,
        trial_forms::TrF,
        test_forms::TeF,
    ) where {
        manifold_dim,
        num_TrF,
        num_TeF,
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        TeF <: NTuple{num_TeF, Forms.AbstractFormSpace{manifold_dim}},
    }
        return new{manifold_dim, Q, TrF, TeF, Tuple{Nothing}}(
            quad_rule, trial_forms, test_forms, (nothing,)
        )
    end

    # Convenience constructor for single trial, test and forcing forms..
    function WeakFormInputs(
        quad_rule::Q,
        trial_forms::TrF,
        test_forms::TeF,
        forcing::F,
    ) where {
        manifold_dim,
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        TeF <: Forms.AbstractFormSpace{manifold_dim},
        F <: Forms.AbstractFormField{manifold_dim}
    }
        return WeakFormInputs(quad_rule, (trial_forms,), (test_forms,), (forcing,))
    end

    # Convenience constructor for single trial, test and forcing forms..
    function WeakFormInputs(
        quad_rule::Q,
        trial_forms::TrF,
        test_forms::TeF,
    ) where {
        manifold_dim,
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        TeF <: Forms.AbstractFormSpace{manifold_dim},
    }
        return WeakFormInputs(quad_rule, (trial_forms,), (test_forms,))
    end

    # Convenience constructor for Galerking methods: trial and test spaces are the same.
    function WeakFormInputs(
        quad_rule::Q,
        forms::TrF,
        forcing::F,
    ) where {
        manifold_dim,
        num_TrF,
        num_F,
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        F <: NTuple{num_F, Forms.AbstractFormField{manifold_dim}},
    }
        return WeakFormInputs(quad_rule, forms, forms, forcing)
    end

    # Convenience constructor for Galerking methods: trial and test spaces are the same.
    function WeakFormInputs(
        quad_rule::Q,
        forms::TrF,
    ) where {
        manifold_dim,
        num_TrF,
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
    }
        return WeakFormInputs(quad_rule, forms, forms)
    end

    # Convenience constructor for single-space Galerking methods: trial and test spaces are
    # the same.
    function WeakFormInputs(
        quad_rule::Q,
        forms::TrF,
        forcing::F
    ) where {
        manifold_dim,
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        F <: Forms.AbstractFormField{manifold_dim}
    }
        return WeakFormInputs(quad_rule, (forms,), (forms,), (forcing,))
    end

    # Convenience constructor for single-space Galerking methods: trial and test spaces are
    # the same.
    function WeakFormInputs(
        quad_rule::Q,
        forms::TrF,
    ) where {
        manifold_dim,
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
        TrF <: Forms.AbstractFormSpace{manifold_dim},
    }
        return WeakFormInputs(quad_rule, forms, forms)
    end
end

get_global_quadrature_rule(wf_inputs::WeakFormInputs) = wf_inputs.quad_rule

function get_trial_forms(wf_inputs::WeakFormInputs)
    return wf_inputs.trial_forms
end

function get_test_forms(wf_inputs::WeakFormInputs)
    return wf_inputs.test_forms
end

function get_forcing(wf_inputs::WeakFormInputs)
    return wf_inputs.forcing
end

function get_num_elements(wf_inputs::WeakFormInputs)
    geometry = Forms.get_geometry(get_trial_forms(wf_inputs)[1])
    return Geometry.get_num_elements(geometry)
end

function get_num_quadrature_elements(wf_inputs::WeakFormInputs)
    return Quadrature.get_num_elements(get_global_quadrature_rule(wf_inputs))
end

function get_trial_size(wf_inputs::WeakFormInputs)
    trial_size = 0
    for form in get_trial_forms(wf_inputs)
        trial_size += Forms.get_num_basis(form)
    end

    return trial_size
end

function get_test_size(wf_inputs::WeakFormInputs)
    test_size = 0
    for form in get_test_forms(wf_inputs)
        test_size += Forms.get_num_basis(form)
    end

    return test_size
end

function get_problem_size(wf_inputs::WeakFormInputs)
    return get_trial_size(wf_inputs), get_test_size(wf_inputs)
end

function get_trial_max_local_dim(wf_inputs::WeakFormInputs)
    trial_max_local_dim = 0
    for form in get_trial_forms(wf_inputs)
        trial_max_local_dim += Forms.get_max_local_dim(form)
    end

    return trial_max_local_dim
end

function get_test_max_local_dim(wf_inputs::WeakFormInputs)
    test_max_local_dim = 0
    for form in get_test_forms(wf_inputs)
        test_max_local_dim += Forms.get_max_local_dim(form)
    end

    return test_max_local_dim
end

function get_estimated_nnz_per_elem(
    wf_inputs::WeakFormInputs{manifold_dim, Q, TrF, TeF, F}
) where {manifold_dim, Q, TrF, TeF, F}
    trial_max_local_dim = get_trial_max_local_dim(wf_inputs)
    test_max_local_dim = get_test_max_local_dim(wf_inputs)
    left_hand_nnz = trial_max_local_dim * test_max_local_dim
    right_hand_nnz = test_max_local_dim
    if F == Nothing
        right_hand_nnz *= trial_max_local_dim
    end

    return left_hand_nnz, right_hand_nnz
end
