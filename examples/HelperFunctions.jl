using LinearAlgebra
using Printf

############################################################################################
#                                        Variables                                         #
############################################################################################
fpi = Float64(π)

############################################################################################
#                                     General methods                                      #
############################################################################################

function unit_square_biquadratic(num_ref::Int=0)
    deg = (2, 2)
    TP = tensor_product_bsplines(deg, (1, 1), (-1, -1), (1.0, 1.0))
    geom_coeffs = [
        0.0 0.0
        0.5 0.0
        1.0 0.0
        0.0 0.5
        0.95 0.95
        1.0 0.5
        0.0 1.0
        0.5 1.0
        1.0 1.0
    ]
    if num_ref == 0
        return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
    else
        ts_TP, TP_refined = Mantis.FunctionSpaces.build_two_scale_operator(
            TP, (2^num_ref, 2^num_ref)
        )
        geom_coeffs_refined = ts_TP.global_subdiv_matrix * geom_coeffs
        return Mantis.Geometry.FEMGeometry(TP_refined, geom_coeffs_refined)
    end
end

function bilinear_wedge(num_ref::Int=0)
    deg = (1, 1)
    TP = tensor_product_bsplines(deg, (1, 1), (-1, -1), (1.0, 1.0))
    geom_coeffs = [
        0.0 0.0
        1.0 0.0
        0.0 1.0
        1.5 1.75
    ]
    if num_ref == 0
        return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
    else
        ts_TP, TP_refined = Mantis.FunctionSpaces.build_two_scale_operator(
            TP, (2^num_ref, 2^num_ref)
        )
        geom_coeffs_refined = ts_TP.global_subdiv_matrix * geom_coeffs
        return Mantis.Geometry.FEMGeometry(TP_refined, geom_coeffs_refined)
    end
end

function quarter_annulus_nurbs(radius_inner::Float64, radius_outer::Float64)
    deg = 2
    num_el = 1
    # B-spline space - angular direction
    Bθ = univariate_bsplines(deg, num_el, -1)
    # NURBS space - angular direction
    Rθ = Mantis.FunctionSpaces.RationalFiniteElementSpace(Bθ, [1, 1 / sqrt(2), 1])
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, num_el, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Rθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector = [
        0.0 1.0
        1.0 1.0
        1.0 0.0
    ]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg + 1)
    geom_coeffs = vcat([geom_coeffs_sector .* radii[i] for i in eachindex(radii)]...)

    return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
end

function quarter_annulus_trigonometric_bspline(radius_inner::Float64, radius_outer::Float64)
    deg = 2
    Wt = pi / 2
    num_el = 1
    # TB-spline space - angular direction
    Bθ = univariate_bsplines(
        Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt), num_el, -1
    )
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, 1, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Bθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector = [
        0.0 1.0
        1.0 1.0
        1.0 0.0
    ]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg + 1)
    geom_coeffs = vcat([geom_coeffs_sector .* radii[i] for i in eachindex(radii)]...)

    return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
end

function quarter_bulging_annulus_trigonometric_bspline(
    radius_inner::Float64, radius_outer::Float64
)
    deg = 2
    Wt = pi / 2
    num_el = 1
    # TB-spline space - angular direction
    Bθ = univariate_bsplines(
        Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt), num_el, -1
    )
    # B-spline space - radial direction
    Br = univariate_bsplines(deg, 1, -1)
    # tensor-product space
    TP = Mantis.FunctionSpaces.TensorProductSpace(Bθ, Br)
    # control points for quarter-circle
    geom_coeffs_sector = [
        0.0 1.0
        1.0 1.0
        1.0 0.0
    ]
    # control points for quarter annulus
    radii = LinRange(radius_inner, radius_outer, deg + 1)
    geom_coeffs = vcat([geom_coeffs_sector .* radii[i] for i in eachindex(radii)]...)

    return Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
end

function get_quadrature_rules(
    nq_assembly::NTuple{manifold_dim, Int},
    nq_error::NTuple{manifold_dim, Int},
    q_rule::Function = Mantis.Quadrature.gauss_legendre,
) where {manifold_dim}
    q_assembly = Mantis.Quadrature.tensor_product_rule(nq_assembly, q_rule)
    q_error = Mantis.Quadrature.tensor_product_rule(nq_error, q_rule)

    return q_assembly, q_error
end

############################################################################################
#                                   Boundary conditions                                    #
############################################################################################

function null_tangential_boundary_conditions(X)
    # H_zero(curl; Ω) boundary conditions
    dof_partition = Mantis.FunctionSpaces.get_dof_partition(X.fem_space)
    bc_H_zero_curl_1 = Dict{Int, Float64}(
        i => 0.0 for j in [1, 2, 3, 7, 8, 9] for i in dof_partition[1][1][j]
    )
    bc_H_zero_curl_2 = Dict{Int, Float64}(
        i => 0.0 for j in [1, 3, 4, 6, 7, 9] for i in dof_partition[2][1][j]
    )
    bc_H_zero_curl = merge(bc_H_zero_curl_1, bc_H_zero_curl_2)

    return bc_H_zero_curl
end

function null_normal_boundary_conditions(X)
    # H_zero(div; Ω) boundary conditions
    dof_partition = Mantis.FunctionSpaces.get_dof_partition(X.fem_space)
    bc_H_zero_div_1 = Dict{Int, Float64}(
        i => 0.0 for j in [1, 3, 4, 6, 7, 9] for i in dof_partition[1][1][j]
    )
    bc_H_zero_div_2 = Dict{Int, Float64}(
        i => 0.0 for j in [1, 2, 3, 7, 8, 9] for i in dof_partition[2][1][j]
    )
    bc_H_zero_div = merge(bc_H_zero_div_1, bc_H_zero_div_2)

    return bc_H_zero_div
end


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
    num_eig::Int, geom::Mantis.Geometry.AbstractGeometry{2}, box_size::NTuple{2, Float64}
)

    eig_vals = Vector{Float64}(undef, (num_eig+1)^2)
    eig_funcs = Vector{Mantis.Forms.AnalyticalFormField{2, 1, typeof(geom)}}(
        undef, (num_eig+1)^2
    )
    eig_count = 1
    scale_factors = (fpi / box_size[1], fpi / box_size[2])
    for m in 0:num_eig
        for n in 0:num_eig
            curr_val = (scale_factors[1] * m)^2 + (scale_factors[2] * n)^2
            eig_vals[eig_count] = curr_val
            eig_func_expr = x -> maxwell_eigen_function(m, n, scale_factors, x)
            eig_funcs[eig_count] = Mantis.Forms.AnalyticalFormField(
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
    W::Mantis.Forms.AbstractFormSpace{2, 0, G},
    X::Mantis.Forms.AbstractFormSpace{2, 1, G},
    q_rule::Mantis.Quadrature.AbstractQuadratureRule,
    num_eig::Int;
    verbose::Bool=false,
) where {G}
    # H₀(curl; Ω) boundary conditions
    bc_H_zero_curl = null_tangential_boundary_conditions(X)

    # Assemble matrices
    weak_form = Mantis.Assemblers.maxwell_eigenvalue
    weak_form_inputs = Mantis.Assemblers.EigenvalueWeakFormInputs(X, X, q_rule)
    A, B = Mantis.Assemblers.assemble_eigenvalue(
        weak_form, weak_form_inputs, bc_H_zero_curl
    )

    ωₕ², eig_vecs = LinearAlgebra.eigen(A, B)
    ωₕ² = real.(ωₕ²)
    sort_ids = sortperm(ωₕ²)
    ωₕ² = ωₕ²[sort_ids]
    eig_vecs = eig_vecs[:, sort_ids]

    nullspace_offset = Mantis.Forms.get_num_basis(W) - length(bc_H_zero_curl)
    if verbose
        println("""
            The nullspace offset is:
            \t$(nullspace_offset) = dim(ℜ⁰) - (dim(ℜ¹) - dim(ℜ¹ ∩ H₀(curl; Ω)) .
            """
        )
    end
    ωₕ² = (ωₕ²[(nullspace_offset + 1):end])[1:num_eig]

    uₕ = Vector{Mantis.Forms.FormField{2, 1, G}}(undef, num_eig)
    non_boundary_rows_cols = setdiff(1:Mantis.Forms.get_num_basis(X), keys(bc_H_zero_curl))
    for eig_id in 1:num_eig
        subscript_str = join(Char(0x2080 + d) for d in reverse(digits(eig_id)))
        uₕ[eig_id] = Mantis.Forms.FormField(X, "uₕ" * subscript_str)
        uₕ[eig_id].coefficients[non_boundary_rows_cols] .=
            real.(eig_vecs[:, nullspace_offset + eig_id])
    end

    return ωₕ², uₕ
end
