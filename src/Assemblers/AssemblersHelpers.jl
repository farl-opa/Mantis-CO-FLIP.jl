
############################################################################################
#                                    Maxwell Eigenvalue                                    #
############################################################################################

function maxwell_eigen_function(
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

function get_maxwell_eig(
    num_eig::Int, geom::Geometry.AbstractGeometry{2}, box_size::NTuple{2, Float64}
)

    eig_vals = Vector{Float64}(undef, (num_eig+1)^2)
    eig_funcs = Vector{Forms.AnalyticalFormField{2, 1, typeof(geom)}}(
        undef, (num_eig+1)^2
    )
    eig_count = 1
    scale_factors = (pi / box_size[1], pi / box_size[2])
    for m in 0:num_eig
        for n in 0:num_eig
            curr_val = (scale_factors[1] * m)^2 + (scale_factors[2] * n)^2
            eig_vals[eig_count] = curr_val
            eig_func_expr = x -> maxwell_eigen_function(m, n, scale_factors, x)
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
