import Mantis

using Test
using LinearAlgebra
using SparseArrays

@doc raw"""
    L2_projection(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Weak form for the computation of the ``L^2``-projection on the given element. The associated weak formulation is:

For given ``f^k \in L^2 \Lambda^k (\Omega)``, find ``\phi^k_h \in X^k`` such that 
```math
\int_{\Omega} \phi^k_h \wedge \star \varphi^k_h = -\int_{\Omega} f^k \wedge \star \varphi^k_h \quad \forall \ \varphi^k_h \in X^k\;,
```
where ``X`` is the discrete de Rham complex.
"""
function L2_projection(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}
    Forms = Mantis.Forms

    # The l.h.s. is the inner product between the test and trial functions.
    A_row_idx, A_col_idx, A_elem = Forms.evaluate_inner_product(inputs.space_test[1], inputs.space_trial[1], element_id, inputs.quad_rule)

    # The r.h.s. is the inner product between the test and forcing functions.
    b_row_idx, _, b_elem = Forms.evaluate_inner_product(inputs.space_test[1], inputs.forcing[1], element_id, inputs.quad_rule)
    
    # The output should be the contribution to the left-hand-side matrix 
    # A and right-hand-side vector b. The outputs are tuples of 
    # row_indices, column_indices, values for the matrix part and 
    # column_indices, values for the vector part.
    return (A_row_idx, A_col_idx, A_elem), (b_row_idx, b_elem)
    
end

function L2_projection(fₑ, Xᵏ, ∫)
    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(fₑ, Xᵏ, ∫)

    # assemble all matrices
    A, b = Mantis.Assemblers.assemble(L2_projection, weak_form_inputs, Dict{Int, Float64}())

    # solve for coefficients of solution
    sol = A \ b

    # create the form field from the solution coefficients
    fₕ = Mantis.Forms.build_form_fields(weak_form_inputs.space_trial, sol; labels=("fh",))[1]
    
    # return the field
    return fₕ
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
num_ref_levels = 5

# exact solution for the problem
function sinusoidal_solution(form_rank::Int, geo::Mantis.Geometry.AbstractGeometry{manifold_dim}) where {manifold_dim}
    n_form_components = binomial(manifold_dim, form_rank)
    ω = 2.0 * pi
    function my_sol(x::Matrix{Float64})
        # ∀i ∈ {1, 2, ..., n}: uᵢ = sin(ωx¹)sin(ωx²)...sin(ωxⁿ) 
        y = @. sin(ω * x)
        return repeat([vec(prod(y, dims=2))], n_form_components)
    end
    return Mantis.Forms.AnalyticalFormField(form_rank, my_sol, geo, "f")
end

# RUN L2 PROJECTION PROBLEM -------------------------------------------------------------------
errors = zeros(Float64, num_ref_levels+1, length(p⁰), length(section_space_type), length(mesh_type), 1+manifold_dim)
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

                for form_rank in 0:manifold_dim
                    n_dofs = Mantis.Forms.get_num_basis(X[form_rank+1])
                    if verbose
                        display("   Form rank = $form_rank, n_dofs = $n_dofs")
                    end
                    # exact solution for the problem
                    fₑ = sinusoidal_solution(form_rank, geometry)

                    # solve the problem
                    fₕ = L2_projection(fₑ, X[form_rank+1], ∫)
                    
                    # compute error
                    error = L2_norm(fₕ - fₑ, ∫)
                    errors[ref_lev+1, p_idx, ss_idx, mesh_idx, form_rank+1] = error

                    if verbose; display("   Error: $error"); end
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
            if isapprox(errors[end, p_idx, ss_idx, mesh_idx, 1], 0.0, atol=1e-14)
                continue
            else
                # expected 0-form convergence: p+1
                @test isapprox(error_rates[end, p_idx, ss_idx, mesh_idx, 1], p+1, atol=1.5e-1)
            end
            for form_rank in 1:manifold_dim
                if isapprox(errors[end, p_idx, ss_idx, mesh_idx, form_rank+1], 0.0, atol=1e-14)
                    continue
                else
                    # expected k-form convergence for k>0: p
                    @test isapprox(error_rates[end, p_idx, ss_idx, mesh_idx, form_rank+1], p, atol=1.5e-1)
                end
            end
        end
    end
end