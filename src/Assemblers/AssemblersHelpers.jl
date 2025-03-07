
############################################################################################
#                                    Maxwell Eigenvalue                                    #
############################################################################################

function maxwell_eigenvalue(
    inputs::WeakFormInputs{manifold_dim, TrF, TeF}, element_id
) where {manifold_dim, TrF, TeF}
    # The weak form in question is:
    #  ⟨curl u, curl v⟩ = ω²⟨u,v⟩, ∀v ∈ H(curl; Ω).   

    # Left-hand side
    # A term = ⟨curl u, curl v⟩
    trial_forms = get_trial_forms(inputs)
    test_forms = get_test_forms(inputs)
    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(
        Forms.exterior_derivative(trial_forms[1]),
        Forms.exterior_derivative(test_forms[1]),
        element_id,
        inputs.quad_rule,
    )

    # Right-hand side
    # B term = ⟨u,v⟩
    B_row_idx, B_col_idx, B_elem = Forms.evaluate_inner_product(
        trial_forms[1], test_forms[1], element_id, inputs.quad_rule
    )

    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (B_row_idx, B_col_idx, B_elem)
end

"""
    analytical_maxwell_eigenfunction(
        m::Int, n::Int, scale_factors::NTuple{2, Float64}, x::Matrix{Float64}
    )

Evaluates the analytical Maxwell eigenfunction for the eigenmode `(m, n)` at points `x`. 

# Arguments
- `m::Int`: Eigenmode component x.
- `n::Int`: Eigenmode component y.
- `scale_factors::NTuple{2, Int}`: Scaling factors based on the size of the domain.
- `x::Matrix{Float64}`: Evaluation points. 

# Returns
- `NTuple{2, Vector{Float64}}`: The evaluated Maxwell eigenfunction at points `x`.
"""
function analytical_maxwell_eigenfunction(
    m::Int, n::Int, scale_factors::NTuple{2, Float64}, x::Matrix{Float64}
)
    num_points = size(x, 1)

    x_component = Vector{Float64}(undef, num_points)
    y_component = Vector{Float64}(undef, num_points)

    for point in axes(x, 1)
        x_component[point] =
            cos(m * scale_factors[1] * x[point, 1]) *
            sin(n * scale_factors[2] * x[point, 2])
        y_component[point] =
            sin(m * scale_factors[1] * x[point, 1]) *
            cos(n * scale_factors[2] * x[point, 2])
    end

    return [x_component, y_component]
end

"""
    get_maxwell_eig(num_eig::Int, geom::G) where {G <: Geometry.AbstractGeometry{2}}

Returns the first `num_eig` eigenvalues and 1-form eigenfunctions on the geometry `geom`.

# Arguments
- `num_eig::Int`: Number of eigenvalues and eigenfunctions to compute.
- `geom::Geometry.AbstractGeometry{2}`: The two-dimensional geometry.

# Returns
- `Vector{Float64}`: The first `num_eig` analytical eigenvalues.
- `Vector{Forms.AnalyticalFormField{2, 1, G}`: The first `num_eig` analytical
    eigenfunctions.
"""
function get_analytical_maxwell_eig(
    num_eig::Int, geom::G, scale_factors::NTuple{2, Float64}
) where {G <: Geometry.AbstractGeometry{2}}
    eig_vals = Vector{Float64}(undef, (num_eig+1)^2)
    eig_funcs = Vector{Forms.AnalyticalFormField{2, 1, typeof(geom)}}(
        undef, (num_eig+1)^2
    )

    eig_count = 1
    for m in 0:num_eig
        for n in 0:num_eig
            curr_val = (scale_factors[1] * m)^2 + (scale_factors[2] * n)^2
            eig_vals[eig_count] = curr_val
            eig_func_expr = x -> analytical_maxwell_eigenfunction(m, n, scale_factors, x)
            eig_funcs[eig_count] = Forms.AnalyticalFormField(
                1, eig_func_expr, geom, "u"
            )
            eig_count += 1
        end
    end

    sort_inds = sortperm(eig_vals)
    eig_vals = (eig_vals[sort_inds])[2:num_eig+1]
    eig_funcs = (eig_funcs[sort_inds])[2:num_eig+1]

    return eig_vals, eig_funcs
end

function solve_maxwell_eig(
    W::Forms.AbstractFormSpace{2, 0, G},
    X::Forms.AbstractFormSpace{2, 1, G},
    q_rule::Quadrature.AbstractQuadratureRule,
    num_eig::Int;
    verbose::Bool=false,
) where {G}
    # H₀(curl; Ω) boundary conditions
    bc_H_zero_curl = Forms.null_tangential_boundary_conditions(X)

    # Assemble matrices
    weak_form = maxwell_eigenvalue
    weak_form_inputs = WeakFormInputs(q_rule, X, X)
    A, B = assemble_eigenvalue(
        weak_form, weak_form_inputs, bc_H_zero_curl
    )

    ωₕ², eig_vecs = LinearAlgebra.eigen(A, B)
    ωₕ² = real.(ωₕ²)
    sort_ids = sortperm(ωₕ²)
    ωₕ² = ωₕ²[sort_ids]
    eig_vecs = eig_vecs[:, sort_ids]

    nullspace_offset = Forms.get_num_basis(W) - length(bc_H_zero_curl)
    if verbose
        println("""
            The nullspace offset is:
            \t$(nullspace_offset) = dim(ℜ⁰) - (dim(ℜ¹) - dim(ℜ¹ ∩ H₀(curl; Ω)) .
            """
        )
    end
    ωₕ² = (ωₕ²[(nullspace_offset + 1):end])[1:num_eig]

    uₕ = Vector{Forms.FormField{2, 1, G}}(undef, num_eig)
    non_boundary_rows_cols = setdiff(1:Forms.get_num_basis(X), keys(bc_H_zero_curl))
    for eig_id in 1:num_eig
        subscript_str = join(Char(0x2080 + d) for d in reverse(digits(eig_id)))
        uₕ[eig_id] = Forms.FormField(X, "uₕ" * subscript_str)
        uₕ[eig_id].coefficients[non_boundary_rows_cols] .=
            real.(eig_vecs[:, nullspace_offset + eig_id])
    end

    return ωₕ², uₕ
end

function solve_maxwell_eig(
    complex::C,
    num_steps::Int,
    dorfler_parameter::Float64,
    Lchains::Bool,
    q_rule_assembly::Quadrature.AbstractQuadratureRule{manifold_dim},
    q_rule_error::Quadrature.AbstractQuadratureRule{manifold_dim},
    eigenfunction::Int,
    num_eig::Int,
    scale_factors::NTuple{manifold_dim, Float64};
    verbose::Bool=false,
) where {manifold_dim, num_forms, C <: NTuple{num_forms, Forms.AbstractFormSpace}}
    if verbose
        println("Solving the problem on initial step...")
    end

    # Exact solution on initial step
    exact_eigvals, exact_eigfuncs = get_analytical_maxwell_eig(
        num_eig, Forms.get_geometry(complex[1]), scale_factors
    )

    # Solve problem on initial step
    compt_eigvals, compt_eigfuncs = solve_maxwell_eig(
        complex[1], complex[2], q_rule_assembly, num_eig
    )

    # Calculate element-wise error
    err_per_element = Analysis.compute_error_per_element(
        compt_eigfuncs[eigenfunction], exact_eigfuncs[eigenfunction], q_rule_error
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
        exact_eigvals, exact_eigfuncs = get_analytical_maxwell_eig(
            num_eig, Forms.get_geometry(complex[1]), scale_factors
        )

        # Solve problem on current step
        compt_eigvals, compt_eigfuncs = solve_maxwell_eig(
            complex[1], complex[2], q_rule_assembly, num_eig; verbose
        )

        err_per_element = Analysis._compute_square_error_per_element(
            compt_eigfuncs[eigenfunction], exact_eigfuncs[eigenfunction], q_rule_error
        )

    end

    return compt_eigvals, compt_eigfuncs
end

############################################################################################
#                                     Hodge Laplacian                                      #
############################################################################################
function one_form_hodge_laplacian(inputs::WeakFormInputs, element_id::Int)

    q_rule = Assemblers.get_quadrature_rule(inputs)
    trial_forms = Assemblers.get_trial_forms(inputs)
    test_forms = Assemblers.get_test_forms(inputs)
    forcing = Assemblers.get_forcing(inputs)

    # The mixed weak form in question is:
    #  ⟨τ, σ⟩ - ⟨dτ, u⟩ = 0, ∀τ ∈ V⁰,   
    #  ⟨v, dσ⟩ + ⟨dv, du⟩ = ⟨v, f⟩, ∀v ∈ V¹.

    # Left-hand side 

    # A11 term = ⟨τ, σ⟩
    A_row_idx_11, A_col_idx_11, A_elem_11 = Forms.evaluate_inner_product(
        test_forms[1], trial_forms[1], element_id, q_rule
    )

    # A12 term = -⟨dτ, u⟩
    A_row_idx_12, A_col_idx_12, A_elem_12 = Forms.evaluate_inner_product(
        Forms.exterior_derivative(test_forms[1]), trial_forms[2], element_id, q_rule
    )

    # A21 term = ⟨v, dσ⟩
    A_row_idx_21, A_col_idx_21, A_elem_21 = Forms.evaluate_inner_product(
        test_forms[2], Forms.exterior_derivative(trial_forms[1]), element_id, q_rule
    )

    # A22 term = ⟨dv, du⟩
    A_row_idx_22, A_col_idx_22, A_elem_22 = Forms.evaluate_inner_product(
        Forms.exterior_derivative(test_forms[2]),
        Forms.exterior_derivative(trial_forms[2]),
        element_id,
        q_rule,
    )

    # Add offsets.
    trial_offset = Forms.get_num_basis(trial_forms[1])
    test_offset = Forms.get_num_basis(test_forms[1])
    A_row_idx_21 .+= test_offset
    A_col_idx_12 .+= trial_offset
    A_row_idx_22 .+= test_offset
    A_col_idx_22 .+= trial_offset

    # Put all variables together.
    A_row_idx = vcat(A_row_idx_11, A_row_idx_12, A_row_idx_21, A_row_idx_22)
    A_col_idx = vcat(A_col_idx_11, A_col_idx_12, A_col_idx_21, A_col_idx_22)
    A_elem = vcat(A_elem_11, -A_elem_12, A_elem_21, A_elem_22)

    # Right-hand side 

    # b1 term = 0

    # b2 term = ⟨v, f⟩
    b_row_idx_2, _, b_elem_2 = Forms.evaluate_inner_product(
        test_forms[2], forcing[1], element_id, q_rule
    )
    b_row_idx_2 .+= test_offset

    return (A_row_idx, A_col_idx, A_elem), (b_row_idx_2, b_elem_2)
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

    return compt_one_form, compt_zero_form
end

function solve_one_form_hodge_laplacian(
    complex::C,
    forcing_function::Function,
    num_steps::Int,
    dorfler_parameter::Float64,
    Lchains::Bool,
    q_rule_assembly::Quadrature.AbstractQuadratureRule{manifold_dim},
    q_rule_error::Quadrature.AbstractQuadratureRule{manifold_dim};
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
        complex[1], complex[2], q_rule_assembly, forcing
    )

    # Calculate element-wise error
    err_per_element = Analysis.compute_error_per_element(
        compt_one_form, exact_one_form, q_rule_error
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
            complex[1], complex[2], q_rule_assembly, forcing
        )

        # Calculate element-wise error
        err_per_element = Analysis.compute_error_per_element(
            compt_one_form, exact_one_form, q_rule_error
        )
    end

    return compt_one_form, compt_zero_form
end
