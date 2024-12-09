import Mantis

using Test
using LinearAlgebra
using SparseArrays

@doc raw"""
    zero_form_hodge_laplacian(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Function for assembling the weak form of the 0-form Hodge Laplacian on the given element. The associated weak formulation is:

For given ``f^0 \in L^2 \Lambda^0 (\Omega)``, find ``\phi^0_h \in X^0`` such that 
```math
\int_{\Omega} d \phi^0_h \wedge \star d \varphi^0_h = -\int_{\Omega} f^0 \wedge \star \varphi^0_h \quad \forall \ \varphi^0_h \in X^0\;,
```
where ``X`` is the discrete de Rham complex, and such that ``\phi^0_h`` satisfies zero Dirichlet boundary conditions.
"""
function zero_form_hodge_laplacian(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    Forms = Mantis.Forms

    # The inner product will be between the exterior derivative of the 
    # trial zero form with the exterior derivative of the test zero 
    # form, so we compute those first.
    dtrial = Forms.exterior_derivative(inputs.space_trial[1])
    dtest = Forms.exterior_derivative(inputs.space_test[1])

    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(dtest, dtrial, element_id, inputs.quad_rule)
    
    # The linear form is the inner product between the trial form and 
    # the forcing function which is a form of an appropriate rank.
    b_row_idx, _, b_elem = Forms.evaluate_inner_product(inputs.space_test[1], inputs.forcing[1], element_id, inputs.quad_rule)
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # row_indices, values for the vector part. For this case, no shifts 
    # or offsets are needed.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end

function zero_form_hodge_laplacian(fₑ, X⁰, ∫)
    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(fₑ, X⁰, ∫)

    # boundary conditions: all entries of the dof_partition contribute except for the interior (=5th index)
    dof_partition = Mantis.FunctionSpaces.get_component_dof_partition(X⁰.fem_space, 1)
    bc_inds = vcat([dof_partition[1][i] for i in setdiff(1:9, 5)]...)
    bc_vals = zeros(Float64,length(bc_inds))

    # assemble all matrices
    A, b = Mantis.Assemblers.assemble(zero_form_hodge_laplacian, weak_form_inputs, Dict(zip(bc_inds, bc_vals)))

    # solve for coefficients of solution
    sol = A \ b

    # create the form field from the solution coefficients
    uₕ = Mantis.Forms.build_form_fields(weak_form_inputs.space_trial, sol; labels=("uh",))[1]
    
    # return the field
    return uₕ
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
function sinusoidal_solution(geo::Mantis.Geometry.AbstractGeometry{manifold_dim}) where {manifold_dim}
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # u = [sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
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
    function laplace_my_sol(x::Matrix{Float64})
        # [-(u₁₁+u₂₂+...uₙₙ)] = [2ω²*sin(ωx¹)sin(ωx²)...sin(ωxⁿ)]
        y = prod(sin.(ω * x), dims=2)
        y = @. 2 * ω * ω * y
        return [vec(y)]
    end
    u⁰ = Mantis.Forms.AnalyticalFormField(0, my_sol, geo, "u")
    du⁰ = Mantis.Forms.AnalyticalFormField(1, grad_my_sol, geo, "du")
    f⁰ = Mantis.Forms.AnalyticalFormField(0, laplace_my_sol, geo, "f")
    return u⁰, du⁰, f⁰
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
                n_dofs = Mantis.Forms.get_num_basis(X[1])
                if verbose
                    display("   n_dofs = $n_dofs")
                end
                # exact solution for the problem
                uₑ, duₑ, fₑ = sinusoidal_solution(geometry)

                # solve the problem
                uₕ = zero_form_hodge_laplacian(fₑ, X[1], ∫)
                
                # compute error
                error = L2_norm(uₕ - uₑ, ∫)
                derror = L2_norm(Mantis.Forms.exterior_derivative(uₕ) - duₑ, ∫)
                errors[ref_lev+1, p_idx, ss_idx, mesh_idx, 1] = error
                errors[ref_lev+1, p_idx, ss_idx, mesh_idx, 2] = derror

                if verbose
                    display("   L2-Error: $error")
                    display("   H1-Error: $derror")
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
                # expected 0-form convergence: p+1
                @test isapprox(error_rates[end, p_idx, ss_idx, mesh_idx, 1], p+1, atol=1.5e-1)
                if isapprox(errors[end, p_idx, ss_idx, mesh_idx, 2], 0.0, atol=1e-12)
                    continue
                else
                    # expected k-form convergence for k>0: p
                    @test isapprox(error_rates[end, p_idx, ss_idx, mesh_idx, 2], p, atol=1.5e-1)
                end
            end
        end
    end
end