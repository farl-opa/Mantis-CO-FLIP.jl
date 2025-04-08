struct WeakFormInputs{manifold_dim, TrF, TeF, F} <: AbstractInputs
    trial_forms::TrF
    test_forms::TeF
    forcing::F

    function WeakFormInputs(
        trial_forms::TrF,
        test_forms::TeF,
        forcing::F,
    ) where {
        manifold_dim,
        num_TrF,
        num_TeF,
        num_F,
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        TeF <: NTuple{num_TeF, Forms.AbstractFormSpace{manifold_dim}},
        F <: NTuple{num_F, Forms.AbstractFormField{manifold_dim}},
    }
        return new{manifold_dim, TrF, TeF, F}(trial_forms, test_forms, forcing)
    end

    function WeakFormInputs(
        trial_forms::TrF,
        test_forms::TeF,
    ) where {
        manifold_dim,
        num_TrF,
        num_TeF,
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        TeF <: NTuple{num_TeF, Forms.AbstractFormSpace{manifold_dim}},
    }
        return new{manifold_dim, TrF, TeF, Tuple{Nothing}}(
            trial_forms, test_forms, (nothing,)
        )
    end

    # Convenience constructor for single trial, test and forcing forms..
    function WeakFormInputs(
        trial_forms::TrF,
        test_forms::TeF,
        forcing::F,
    ) where {
        manifold_dim,
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        TeF <: Forms.AbstractFormSpace{manifold_dim},
        F <: Forms.AbstractFormField{manifold_dim}
    }
        return WeakFormInputs((trial_forms,), (test_forms,), (forcing,)) 
    end

    # Convenience constructor for single trial, test and forcing forms..
    function WeakFormInputs(
        trial_forms::TrF,
        test_forms::TeF,
    ) where {
        manifold_dim,
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        TeF <: Forms.AbstractFormSpace{manifold_dim},
    }
        return WeakFormInputs((trial_forms,), (test_forms,)) 
    end

    # Convenience constructor for Galerking methods: trial and test spaces are the same.
    function WeakFormInputs(
        forms::TrF,
        forcing::F,
    ) where {
        manifold_dim,
        num_TrF,
        num_F,
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
        F <: NTuple{num_F, Forms.AbstractFormField{manifold_dim}},
    }
        return WeakFormInputs(forms, forms, forcing)
    end

    # Convenience constructor for Galerking methods: trial and test spaces are the same.
    function WeakFormInputs(
        forms::TrF,
    ) where {
        manifold_dim,
        num_TrF,
        TrF <: NTuple{num_TrF, Forms.AbstractFormSpace{manifold_dim}},
    }
        return WeakFormInputs(forms, forms)
    end

    # Convenience constructor for single-space Galerking methods: trial and test spaces are
    # the same.
    function WeakFormInputs(
        forms::TrF,
        forcing::F
    ) where {
        manifold_dim,
        TrF <: Forms.AbstractFormSpace{manifold_dim},
        F <: Forms.AbstractFormField{manifold_dim}
    }
        return WeakFormInputs((forms,), (forms,), (forcing,))
    end

    # Convenience constructor for single-space Galerking methods: trial and test spaces are
    # the same.
    function WeakFormInputs(
        forms::TrF,
    ) where {
        manifold_dim,
        TrF <: Forms.AbstractFormSpace{manifold_dim},
    }
        return WeakFormInputs(forms, forms)
    end
end

get_trial_forms(wf_inputs::WeakFormInputs) = wf_inputs.trial_forms
get_test_forms(wf_inputs::WeakFormInputs) = wf_inputs.test_forms
get_forcing(wf_inputs::WeakFormInputs) = wf_inputs.forcing
