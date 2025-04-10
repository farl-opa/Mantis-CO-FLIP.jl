############################################################################################
#                              1-form Hodge Laplacian                                      #
############################################################################################

function one_form_hodge_laplacian(inputs::AbstractInputs, element_id::Int)

    ∫ = Assemblers.get_global_quadrature_rule(inputs)
    trial_forms = Assemblers.get_trial_forms(inputs)
    test_forms = Assemblers.get_test_forms(inputs)
    forcing = Assemblers.get_forcing(inputs)

    # The mixed weak form in question is:
    #  (τ, σ) - (dτ, u) = 0, ∀τ ∈ V⁰,
    #  (v, dσ) + (dv, du) = (v, f), ∀v ∈ V¹.

    # Left-hand side

    # A11 term = (τ, σ)
    A_11 = Forms.Wedge(test_forms[1], Forms.Hodge(trial_forms[1]))
    # A12 term = -(dτ, u)
    A_12 = Forms.Wedge(Forms.ExteriorDerivative(test_forms[1]), Forms.Hodge(trial_forms[2]))
    # A21 term = (v, dσ)
    A_21 = Forms.Wedge(test_forms[2], Forms.Hodge(Forms.ExteriorDerivative(trial_forms[1])))
    # A22 term = (dv, du)
    A_22 = Forms.Wedge(
        Forms.ExteriorDerivative(test_forms[2]),
        Forms.Hodge(Forms.ExteriorDerivative(trial_forms[2]))
    )
    # b1 term = 0
    # b2 term = (v, f)
    b_2 = Forms.Wedge(test_forms[2], Forms.Hodge(forcing[1]))

    # integrate terms
    A_elem_11, A_idx_11 = Analysis.integrate(
        ∫, element_id, A_11
    )
    A_elem_12, A_idx_12 = Analysis.integrate(
        ∫, element_id, A_12
    )
    A_elem_21, A_idx_21 = Analysis.integrate(
        ∫, element_id, A_21
    )
    A_elem_22, A_idx_22 = Analysis.integrate(
        ∫, element_id, A_22
    )
    b_elem_2, b_idx_2 = Analysis.integrate(
        ∫, element_id, b_2
    )

    # Add offsets.
    trial_offset = Forms.get_num_basis(trial_forms[1])
    test_offset = Forms.get_num_basis(test_forms[1])
    A_idx_21[1] .+= test_offset
    A_idx_12[2] .+= trial_offset
    A_idx_22[1] .+= test_offset
    A_idx_22[2] .+= trial_offset
    b_idx_2[1] .+= test_offset

    # Put all variables together.
    A_row_idx = vcat(A_idx_11[1], A_idx_12[1], A_idx_21[1], A_idx_22[1]) #A_row_idx_12, A_row_idx_21, A_row_idx_22)
    A_col_idx = vcat(A_idx_11[2], A_idx_12[2], A_idx_21[2], A_idx_22[2])
    A_elem = vcat(A_elem_11, -A_elem_12, A_elem_21, A_elem_22)

    return (A_row_idx, A_col_idx, A_elem), (b_idx_2[1], b_elem_2)
end

function solve_one_form_hodge_laplacian(zero_form, one_form, qr_assembly, forcing)
    weak_form_inputs = Assemblers.WeakFormInputs(
        qr_assembly, (zero_form, one_form), (zero_form, one_form), (forcing,)
    )
    weak_form = one_form_hodge_laplacian
    A, b = Assemblers.assemble(weak_form, weak_form_inputs, Dict{Int, Float64}())
    sol = A \ b
    # Create solutions as forms and return.
    compt_zero_form, compt_one_form = Forms.build_form_fields(
        (zero_form, one_form), sol; labels=("δuₕ", "uₕ")
    )

    return compt_one_form, compt_zero_form, LinearAlgebra.cond(Array(A))
end

function solve_one_form_hodge_laplacian(
    complex::C,
    forcing_function::Function,
    num_steps::Int,
    dorfler_parameter::Float64,
    Lchains::Bool,
    ∫_assembly::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    ∫_error::Quadrature.AbstractGlobalQuadratureRule{manifold_dim};
    verbose::Bool=false,
) where {manifold_dim, num_forms, C <: NTuple{num_forms, Forms.AbstractFormSpace}}
    if verbose
        println("Solving the problem on initial step...")
    end

    # Exact solution on initial step
    exact_one_form, exact_zero_form, forcing = forcing_function(
        1, Forms.get_geometry(complex[1])
    )

    # Solve problem on initial step
    compt_one_form, compt_zero_form = solve_one_form_hodge_laplacian(
        complex[1], complex[2], ∫_assembly, forcing
    )

    # Calculate element-wise error
    err_per_element = Analysis.compute_error_per_element(
        compt_one_form, exact_one_form, ∫_error
    )

    for step in 1:num_steps
        if verbose
            println("Solving the problem on step $step...")
        end

        zero_form_space = FunctionSpaces.get_component_spaces(complex[1].fem_space)[1]

        L = FunctionSpaces.get_num_levels(zero_form_space)

        new_operator, new_space = FunctionSpaces.build_two_scale_operator(
            FunctionSpaces.get_space(zero_form_space, L),
            FunctionSpaces.get_num_subdivisions(zero_form_space),
        )

        dorfler_marking = FunctionSpaces.get_dorfler_marking(
            err_per_element, dorfler_parameter
        )

        # Get domains to be refined in current step
        marked_elements_per_level = FunctionSpaces.get_padding_per_level(
            zero_form_space, dorfler_marking
        )

        # Add Lchains if needed
        if Lchains
            FunctionSpaces.add_Lchains_supports!(
                marked_elements_per_level, zero_form_space, new_operator
            )
        end
        # Get children of marked elements
        refinement_domains = FunctionSpaces.get_refinement_domains(
            zero_form_space, marked_elements_per_level, new_operator
        )

        # Update the hierarchical complex based on the refinement domains and the 0-form space
        complex = Forms.update_hierarchical_de_rham_complex(
            complex, refinement_domains, new_operator, new_space
        )

        # Update exact solution
        exact_one_form, exact_zero_form, forcing = forcing_function(
            1, Forms.get_geometry(complex[1])
        )

        # Solve problem on current step
        compt_one_form, compt_zero_form = solve_one_form_hodge_laplacian(
            complex[1], complex[2], ∫_assembly, forcing
        )

        # Calculate element-wise error
        err_per_element = Analysis.compute_error_per_element(
            compt_one_form, exact_one_form, ∫_error
        )
    end

    return compt_one_form, compt_zero_form
end
