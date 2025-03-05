
############################################################################################
#                                    Maxwell Eigenvalue                                    #
############################################################################################

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
    weak_form_inputs = EigenvalueWeakFormInputs(X, X, q_rule)
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

    # COMPUTE ERROR -------------------------------------------------------------------
    if verbose
        println("Computing error...")
        error_u = Analysis.L2_norm(
            compt_eigfuncs[eigenfunction] - exact_eigfuncs[eigenfunction], q_rule_error
        )
        println("Error in eigenfunction $(eigenfunction): ", error_u)
    end

    for step in 1:num_steps
        println("Solving the problem on step $step...")

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
        # marked_elements_per_level = FunctionSpaces.get_padding_per_level(
        #     zero_form_space, dorfler_marking
        # )
        basis_support = [
            1,
            2,
            3,
            4,
            5,
            16,
            17,
            18,
            19,
            20,
            31,
            32,
            33,
            34,
            35,
            46,
            47,
            48,
            49,
            50,
            61,
            62,
            63,
            64,
            65,
        ]
        marked_elements_per_level = [
            vcat(
                basis_support .+ (5 + 15 * 2),
                basis_support .+ (8 + 15 * 5),
                basis_support .+ (2 + 15 * 5),
                basis_support .+ (5 + 15 * 8),
            ),
        ]

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

        # COMPUTE ERROR -------------------------------------------------------------------
        if verbose
            println("Computing error...")
            error_u = Analysis.L2_norm(
                compt_eigfuncs[eigenfunction] - exact_eigfuncs[eigenfunction], q_rule_error
            )
            println("Error in eigenfunction $(eigenfunction): ", error_u)
        end
    end

    return compt_eigvals, compt_eigfuncs
end
