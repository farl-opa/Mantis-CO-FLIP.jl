import Mantis

using Test
using LinearAlgebra
using SparseArrays

# PROBLEM PARAMETERS -------------------------------------------------------------------
# manifold dimensions
manifold_dim = 2
# number of elements in radial direction
num_el_r = 5
# number of elements in angular direction
num_el_θ = 15
# radius of the domain
R = 1.0
# polynomial degrees of the zero-form finite element spaces to be used
p⁰ = [2, 3]
# type of section spaces to use
θ = 2*pi
α = 5.0
section_space_type = [Mantis.FunctionSpaces.GeneralizedTrigonometric, Mantis.FunctionSpaces.GeneralizedExponential, Mantis.FunctionSpaces.Bernstein]
# print info?
verbose = true
# tolerance for zero values
zero_tol = 1e-12
# tolerance for convergence rates
rate_tol = 2e-1

# number of refinement levels to run
num_ref_levels = 4

# exact solution for the problem
function sinusoidal_solution(form_rank::Int, geo::Mantis.Geometry.AbstractGeometry{manifold_dim}) where {manifold_dim}
    n_form_components = binomial(manifold_dim, form_rank)
    ω = 1.0
    function my_sol(x::Matrix{Float64})
        # ∀i ∈ {1, 2, ..., n}: uᵢ = sin(ωx¹)sin(ωx²)...sin(ωxⁿ)
        y = @. sin(ω * x)
        return repeat([vec(prod(y, dims=2))], n_form_components)
    end
    return Mantis.Forms.AnalyticalFormField(form_rank, my_sol, geo, "f")
end

# RUN L2 PROJECTION PROBLEM -------------------------------------------------------------------
errors = zeros(Float64, num_ref_levels+1, length(p⁰), length(section_space_type), 1+manifold_dim)
for (p_idx, p) in enumerate(p⁰)
    for (ss_idx, section_space) in enumerate(section_space_type)

        if verbose
            @info("Running L2 projection tests for p = $p, section_space = $section_space")
        end

        # section space degrees
        degree = (p, p)

        # function space regularities
        regularities = degree .- 1
        if section_space == Mantis.FunctionSpaces.LobattoLegendre
            regularities = tuple([0 for _ in 1:manifold_dim]...)
        end

        # initialize geometry coefficients
        geom_coeffs_tp = nothing

        # number of elements at the coarsest refinement level
        num_elements = (num_el_θ, num_el_r)

        for ref_lev = 0:num_ref_levels

            if verbose
                println("Refinement level = $ref_lev ------------------------------------")
            end

            # section spaces
            if section_space == Mantis.FunctionSpaces.GeneralizedTrigonometric
                section_spaces = map(section_space, degree, (θ, θ), 1 ./ num_elements)
                dq⁰ = 2 .* degree
            elseif section_space == Mantis.FunctionSpaces.GeneralizedExponential
                section_spaces = map(section_space, degree, (α, α), 1 ./ num_elements)
                dq⁰ = 3 .* degree
            else
                section_spaces = map(section_space, degree)
                dq⁰ = (2, 2)
            end

            # create (and refine) polar spline complex
            X, _, geom_coeffs_tp = Mantis.Forms.create_polar_spline_de_rham_complex(num_elements, section_spaces, regularities, R; refine = ref_lev>0, geom_coeffs_tp = geom_coeffs_tp)

            # update number of elements
            num_elements = (num_el_θ, num_el_r) .* (2^ref_lev)

            # retrieve geometry underlying the form spaces
            geometry = Mantis.Forms.get_geometry(X[1])

            # quadrature rule
            canonical_qrule = Mantis.Quadrature.tensor_product_rule(degree .+ dq⁰, Mantis.Quadrature.gauss_legendre)
            # global quadrature rule
            ∫ = Mantis.Quadrature.StandardQuadrature(canonical_qrule, Mantis.Geometry.get_num_elements(geometry))

            for form_rank in 0:manifold_dim
                n_dofs = Mantis.Forms.get_num_basis(X[form_rank+1])
                if verbose
                    display("   Form rank = $form_rank, n_dofs = $n_dofs")
                end
                # exact solution for the problem
                fₑ = sinusoidal_solution(form_rank, geometry)

                # solve the problem
                fₕ = Mantis.Assemblers.solve_L2_projection(∫, X[form_rank+1], fₑ)

                # compute error
                error = Mantis.Analysis.L2_norm(∫, fₕ - fₑ)
                errors[ref_lev+1, p_idx, ss_idx, form_rank+1] = error

                if verbose; display("   Error: $error"); end
            end
            if verbose; println("...done!"); end
        end
    end
end

# compute orders of convergence
error_rates = log.(Ref(2), errors[1:end-1,:,:,:]./errors[2:end,:,:,:])
if verbose
    println("Error convergence rates:")
    display(error_rates)
end
for (p_idx, p) in enumerate(p⁰)
    for (ss_idx, section_space) in enumerate(section_space_type)
        if isapprox(errors[end, p_idx, ss_idx, 1], 0.0, atol=zero_tol)
            continue
        else
            # expected 0-form convergence: p+1
            @test isapprox(error_rates[end, p_idx, ss_idx, 1], p+1, atol=rate_tol)
        end
        for form_rank in 1:manifold_dim
            if isapprox(errors[end, p_idx, ss_idx, form_rank+1], 0.0, atol=zero_tol)
                continue
            else
                # expected k-form convergence for k>0: p
                @test isapprox(error_rates[end, p_idx, ss_idx, form_rank+1], p, atol=rate_tol)
            end
        end
    end
end
