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

get_quadrature_rule(wf_inputs::WeakFormInputs) = wf_inputs.quad_rule
get_trial_forms(wf_inputs::WeakFormInputs) = wf_inputs.trial_forms
get_test_forms(wf_inputs::WeakFormInputs) = wf_inputs.test_forms
get_forcing(wf_inputs::WeakFormInputs) = wf_inputs.forcing
