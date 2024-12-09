import Mantis

using Test
using LinearAlgebra
using SparseArrays

@doc raw"""
    volume_form_hodge_laplacian(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 2, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Function for assembling the weak form of the n-form Hodge Laplacian on the given element. The associated weak formulation is:

Given ``f^n \in L^2 \Lambda^n (\Omega)``, find ``u^{n-1}_h \in X^{n-1}`` and ``\phi^n \in X^n`` such that 
```math
\begin{gather}
\langle \varepsilon^{n-1}_h, u^{n-1}_h \rangle - \langle d \varepsilon^{n-1}_h, \phi^n_h \rangle = 0 \quad \forall \ \varepsilon^{n-1}_h \in X^{n-1} \\
\langle \varepsilon^n_h, d u^{n-1}_h \rangle = -\langle \varepsilon^n_h, f^n \rangle \quad \forall \ \varepsilon^n_h \in X^n
\end{gather}
```
"""
function volume_form_hodge_laplacian(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 2, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    Forms = Mantis.Forms

    # Left hand side.
    # <ε¹, u¹>
    A_row_idx_11, A_col_idx_11, A_elem_11 = Forms.evaluate_inner_product(inputs.space_test[1], inputs.space_trial[1], element_id, inputs.quad_rule)
    
    # <dε¹, ϕ²>
    A_row_idx_12, A_col_idx_12, A_elem_12 = Forms.evaluate_inner_product(Forms.exterior_derivative(inputs.space_test[1]), inputs.space_trial[2], element_id, inputs.quad_rule)
    
    # <ε², du¹>
    A_row_idx_21, A_col_idx_21, A_elem_21 = Forms.evaluate_inner_product(inputs.space_test[2], Forms.exterior_derivative(inputs.space_trial[1]), element_id, inputs.quad_rule)

    # The remain term, A22, is zero, so not computed.

    # Add offsets.
    A_row_idx_21 .+= Forms.get_num_basis(inputs.space_test[1])

    A_col_idx_12 .+= Forms.get_num_basis(inputs.space_trial[1])

    # Put all variables together.
    A_row_idx = vcat(A_row_idx_11, A_row_idx_12, A_row_idx_21)
    A_col_idx = vcat(A_col_idx_11, A_col_idx_12, A_col_idx_21)
    A_elem = vcat(A_elem_11, A_elem_12, A_elem_21)


    # Right hand side. Only the second part is non-zero.
    # <ε², f²>
    b_row_idx, _, b_elem = Forms.evaluate_inner_product(inputs.space_test[2], inputs.forcing[2], element_id, inputs.quad_rule)
    b_elem .*= -1.0
    
    b_row_idx .+= Forms.get_num_basis(inputs.space_test[1])

    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end

function volume_form_hodge_laplacian(fₑ, Xⁿ⁻¹, Xⁿ, ∫)
    # create mixed form space
    V = Mantis.Forms.MixedFormSpace((Xⁿ⁻¹, Xⁿ))
    F = Mantis.Forms.MixedFormField((nothing, fₑ))

    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(F, V, ∫)

    # assemble all matrices
    A, b = Mantis.Assemblers.assemble(volume_form_hodge_laplacian, weak_form_inputs, Dict{Int, Float64}())

    # solve for coefficients of solution
    sol = A \ b

    # create solution as form fields and return
    uₕ, ϕₕ = Mantis.Forms.build_form_fields(V, sol; labels=("uh", "ϕh"))
    
    # return the field
    return uₕ, ϕₕ
end

# PROBLEM PARAMETERS -------------------------------------------------------------------
# manifold dimensions
manifold_dim = 2
# mesh types to be used
mesh_type = ["cartesian", "curvilinear"]
# number of elements in each direction at the coarsest level of refinement
num_el_0 = 4
# origin of the parametric domain in each direction
origin = (0.0, 0.0)
# length of the domain in each direction
L = (1.0, 1.0)
# polynomial degrees of the zero-form finite element spaces to be used
p⁰ = [2, 3]
# type of section spaces to use
θ = 2*pi
α = 10.0
section_space_type = [Mantis.FunctionSpaces.Bernstein, Mantis.FunctionSpaces.LobattoLegendre, Mantis.FunctionSpaces.GeneralizedTrigonometric, Mantis.FunctionSpaces.GeneralizedExponential]
# print info?
verbose = false

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
            # (a, b) -> (-b, a)
            w = [-w[2], w[1]]
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
for ref_lev = 0:num_ref_levels
    num_elements = (num_el_0 * (2^ref_lev)) .* tuple([1 for _ in 1:manifold_dim]...)
    for (mesh_idx, mesh) in enumerate(mesh_type)
        if mesh == "cartesian"
            geometry = Mantis.Geometry.create_cartesian_box(origin, L, num_elements)
        else
            geometry = Mantis.Geometry.create_curvilinear_square(origin, L, num_elements)
        end
        for (p_idx, p) in enumerate(p⁰)
            for (ss_idx, section_space) in enumerate(section_space_type)
                if verbose
                    @info("Running L2 projection for p = $p, section_space = $section_space, mesh = $mesh, ref_lev = $ref_lev")
                end
                
                # section spaces
                degree = (p, p)
                if section_space == Mantis.FunctionSpaces.GeneralizedTrigonometric
                    section_spaces = map(section_space, degree, θ ./ num_elements)
                    dq⁰ = 2 .* degree
                elseif section_space == Mantis.FunctionSpaces.GeneralizedExponential
                    section_spaces = map(section_space, degree, α ./ num_elements)
                    dq⁰ = 3 .* degree
                else
                    section_spaces = map(section_space, degree)
                    dq⁰ = (2, 2)
                end

                # quadrature rule
                ∫ = Mantis.Quadrature.tensor_product_rule(degree .+ dq⁰, Mantis.Quadrature.gauss_legendre)

                # function spaces
                regularities = degree .- 1
                if section_space == Mantis.FunctionSpaces.LobattoLegendre
                    regularities = tuple([0 for _ in 1:manifold_dim]...)
                end
                X = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(origin, L, num_elements, section_spaces, regularities, geometry)

                # number of dofs
                n_dofs = Mantis.Forms.get_num_basis(X[2]) + Mantis.Forms.get_num_basis(X[3])
                if verbose
                    display("   n_dofs = $n_dofs")
                end
                # exact solution for the problem
                ϕₑ, δϕₑ, fₑ = sinusoidal_solution(geometry)

                # solve the problem
                uₕ, ϕₕ = volume_form_hodge_laplacian(fₑ, X[2], X[3], ∫)
                
                # compute error
                error = L2_norm(ϕₕ - ϕₑ, ∫)
                δerror = L2_norm(uₕ - δϕₑ, ∫)
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