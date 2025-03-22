module NFormTensorProductMixedLaplacianTests

import Mantis

using Test
using LinearAlgebra
using SparseArrays

# PROBLEM PARAMETERS -------------------------------------------------------------------
# manifold dimensions
manifold_dim = 2
# mesh types to be used
mesh_type = ["cartesian"]#, "curvilinear"]
# number of elements in each direction at the coarsest level of refinement
num_el_0 = 2
# origin of the parametric domain in each direction
origin = (0.0, 0.0)
# length of the domain in each direction
L = (1.0, 1.0)
# polynomial degrees of the zero-form finite element spaces to be used
p⁰ = [3]
# type of section spaces to use
θ = 2*pi
α = 10.0
section_space_type = [Mantis.FunctionSpaces.Bernstein]#Mantis.FunctionSpaces.Bernstein]#, Mantis.FunctionSpaces.LobattoLegendre, Mantis.FunctionSpaces.GeneralizedTrigonometric, Mantis.FunctionSpaces.GeneralizedExponential]
# print info?
verbose = true

# number of refinement levels to run
num_ref_levels = 4

# exact solution for the 0-form problem
function sinusoidal_solution(geo::Mantis.Geometry.AbstractGeometry)
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # [u] = [sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = @. sin(ω * x)
        return [vec(prod(y, dims=2))]
    end

    function grad_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = [ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        y = sin.(ω .* x)
        z = ω .* cos.(ω .* x)
        w = Vector{Vector{Float64}}(undef, size(x, 2))
        for i ∈ 1:size(x,2)
            w[i] = z[:,i] .* prod(y[:,setdiff(1:size(x,2), i)], dims=2)[:,1]
        end
        return w
    end

    function flux_my_sol(x::Matrix{Float64})
        # [u₁, u₂, ...] = ⋆[ω*cos(ωx¹)sin(ωx²)...sin(ωxⁿ), ω*sin(ωx¹)cos(ωx²)...sin(ωxⁿ), ...]
        w = grad_my_sol(x)
        if size(x,2) == 1
            # (a) -> (a)
            return [w[1]]

        elseif size(x,2) == 2
            # (a, b) -> (b, -a)
            w = [w[2], -w[1]]
            return w

        elseif size(x,2) == 3
            # (a, b, c) -> (a, b, c)
            return [w[1], w[2], w[3]]

        else
            throw(ArgumentError("Dimension not supported."))
        end
    end

    function laplace_my_sol(x::Matrix{Float64})
        # [-(u₁₁+u₂₂+...uₙₙ)] = [2ω²*sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = prod(sin.(ω * x), dims=2)
        y = @. 2 * ω * ω * y
        return [vec(y)]
    end

    ϕ² = Mantis.Forms.AnalyticalFormField(2, my_sol, geo, "ϕ")
    δϕ² = Mantis.Forms.AnalyticalFormField(1, flux_my_sol, geo, "δϕ")
    f² = Mantis.Forms.AnalyticalFormField(2, laplace_my_sol, geo, "f")

    return ϕ², δϕ², f²
end

# RUN POISSON PROBLEM -------------------------------------------------------------------
errors = zeros(Float64, num_ref_levels+1, length(p⁰), length(section_space_type), length(mesh_type), 2)
for (mesh_idx, mesh) in enumerate(mesh_type)
    for (p_idx, p) in enumerate(p⁰)
        for (ss_idx, section_space) in enumerate(section_space_type)

            if verbose
                @info("Running volume-form Hodge-Laplace tests for p = $p, section_space = $section_space, mesh = $mesh")
            end

            # section space degrees
            degree = (p, p)

            # function space regularities
            regularities = degree .- 1
            if section_space == Mantis.FunctionSpaces.LobattoLegendre
                regularities = tuple([0 for _ in 1:manifold_dim]...)
            end

            for ref_lev = 0:num_ref_levels

                if verbose
                    println("Refinement level = $ref_lev ------------------------------------")
                end

                # update number of elements
                num_elements = (num_el_0 * (2^ref_lev)) .* tuple([1 for _ in 1:manifold_dim]...)

                # geometry
                if mesh == "cartesian"
                    geometry = Mantis.Geometry.create_cartesian_box(origin, L, num_elements)
                else
                    geometry = Mantis.Geometry.create_curvilinear_square(origin, L, num_elements)
                end

                # section spaces
                if section_space == Mantis.FunctionSpaces.GeneralizedTrigonometric
                    section_spaces = map(section_space, degree, (θ, θ), L ./ num_elements)
                    dq⁰ = 2 .* degree
                elseif section_space == Mantis.FunctionSpaces.GeneralizedExponential
                    section_spaces = map(section_space, degree, (α, α), L ./ num_elements)
                    dq⁰ = 3 .* degree
                else
                    section_spaces = map(section_space, degree)
                    dq⁰ = (2, 2)
                end

                # quadrature rule
                canonical_qrule = Mantis.Quadrature.tensor_product_rule(degree .+ dq⁰, Mantis.Quadrature.gauss_legendre)
                ∫ = Mantis.Quadrature.StandardQuadrature(
                    canonical_qrule, prod(num_elements)
                )

                # create tensor-product B-spline complex
                X = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(origin, L, num_elements, section_spaces, regularities, geometry)

                # number of dofs
                n_dofs = Mantis.Forms.get_num_basis(X[2]) + Mantis.Forms.get_num_basis(X[3])
                if verbose
                    display("   n_dofs = $n_dofs")
                end
                # exact solution for the problem
                ϕₑ, δϕₑ, fₑ = sinusoidal_solution(geometry)

                # solve the problem
                uₕ, ϕₕ = Mantis.Assemblers.solve_volume_form_hodge_laplacian(∫, X[2], X[3], fₑ)
                # if verbose
                #     display("   cond_num = $cond_num")
                # end
                # display([n_dofs cond_num])

                # compute error
                error = Mantis.Analysis.L2_norm(∫, ϕₕ - ϕₑ)
                δerror = Mantis.Analysis.L2_norm(∫, uₕ - δϕₑ)
                errors[ref_lev+1, p_idx, ss_idx, mesh_idx, 1] = error
                errors[ref_lev+1, p_idx, ss_idx, mesh_idx, 2] = δerror

                if verbose
                    display("   L2-Error in ϕ: $error")
                    display("   L2-Error in δϕ: $δerror")
                end
                if verbose; println("...done!"); end
            end
        end
    end
end

# compute orders of convergence
error_rates = log.(Ref(2), errors[1:end-1,:,:,:,:]./errors[2:end,:,:,:,:])
if verbose
    println("Error convergence rates:")
    display(error_rates)
end
for (p_idx, p) in enumerate(p⁰)
    for (ss_idx, section_space) in enumerate(section_space_type)
        for (mesh_idx, mesh) in enumerate(mesh_type)
            if isapprox(errors[end, p_idx, ss_idx, mesh_idx, 1], 0.0, atol=1e-12)
                continue
            else
                # expected (n-1)-form convergence: p
                @test isapprox(error_rates[end, p_idx, ss_idx, mesh_idx, 1], p, atol=5e-2)
                if isapprox(errors[end, p_idx, ss_idx, mesh_idx, 2], 0.0, atol=1e-12)
                    continue
                else
                    # expected n-form convergence: p
                    @test isapprox(error_rates[end, p_idx, ss_idx, mesh_idx, 2], p, atol=5e-2)
                end
            end
        end
    end
end

end
