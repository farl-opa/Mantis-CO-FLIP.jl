import Mantis

using Test
using LinearAlgebra
using SparseArrays

@doc raw"""
    L2_projection(inputs::Mantis.Assemblers.WeakFormInputs{manifold_dim, 1, Frhs, Ttrial, Ttest}, element_id) where {manifold_dim, Frhs, Ttrial, Ttest}

Weak form for the computation of the ``L^2``-projection on the given element. The associated weak formulation is:

For given ``f^k \in L^2 \Lambda^k (\Omega)``, find ``\phi^k_h \in L^2 \Lambda^k_h (\Omega)`` such that 
```math
\int_{\Omega} \phi^k_h \wedge \star \varphi^k_h = -\int_{\Omega} f^k \wedge \star \varphi^k_h \quad \forall \ \varphi^k_h \in L^2 \Lambda^k_h (\Omega)
```
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

function L2_projection(fₑ, X, ∫)
    # inputs for the mixed weak form
    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(fₑ, X, ∫)

    # assemble all matrices
    A, b = Mantis.Assemblers.assemble(L2_projection, weak_form_inputs, Dict{Int, Float64}())

    # solve for coefficients of solution
    sol = A \ b

    # create solution as form fields and return
    V_fields = Mantis.Forms.build_form_fields(weak_form_inputs.space_trial, sol; labels=("fh",))
    
    return V_fields[1]
end

function L2_norm(u, ∫)
    norm = 0.0
    for el_id ∈ 1:Mantis.Geometry.get_num_elements(u.geometry)
        inner_prod = SparseArrays.sparse(Mantis.Forms.evaluate_inner_product(u, u, el_id, ∫)...)
        norm += inner_prod[1,1]
    end
    return sqrt(norm)
end

# PROBLEM PARAMETERS -------------------------------------------------------------------
# manifold dimensions
const manifold_dim = 2
# mesh types to be used
mesh_type = ["cartesian", "curvilinear"]
# number of elements in each direction at the coarsest level of refinement
num_el_0 = (2, 2)
# origin of the parametric domain in each direction
origin = (0.0, 0.0)
# length of the domain in each direction
L = (1.0, 1.0)
# polynomial degrees of the zero-form finite element spaces to be used
p⁰ = [2, 3, 4]
# type of section spaces to use
section_space_type = [Mantis.FunctionSpaces.Bernstein]#, "trigonometric", "legendre"]
θ = 2*pi ./ num_el_0
# extra quadrature points compared to degree
dq⁰ = (2, 2)

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

# verbose output
verbose = true

# RUN L2 PROJECTION PROBLEM -------------------------------------------------------------------
errors = zeros(Float64, num_ref_levels+1, length(p⁰), length(section_space_type), length(mesh_type), 1+manifold_dim)
for ref_lev = 0:num_ref_levels
    num_elements = num_el_0 .* (2 .^ ref_lev)
    for (mesh_idx, mesh) in enumerate(mesh_type)
        if mesh == "cartesian"
            geometry = Mantis.Geometry.create_cartesian_box(origin, L, num_elements)
        else
            geometry = Mantis.Geometry.create_curvilinear_square(origin, L, num_elements)
        end
        for (p_idx, p) in enumerate(p⁰)
            for (ss_idx, section_space) in enumerate(section_space_type)
                @info("Running L2 projection for p = $p, section_space = $section_space, mesh = $mesh, ref_lev = $ref_lev")
                
                # section spaces
                degree = (p, p)
                section_spaces = map(section_space, degree)

                # quadrature rule
                ∫ = Mantis.Quadrature.tensor_product_rule(degree .+ dq⁰, Mantis.Quadrature.gauss_legendre)

                # function spaces
                Λ = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(origin, L, num_elements, section_spaces, degree .- 1, geometry)

                for form_rank in 0:manifold_dim
                    n_dofs = Mantis.Forms.get_num_basis(Λ[form_rank+1])
                    display("   Form rank = $form_rank, n_dofs = $n_dofs")
                    # exact solution for the problem
                    fₑ = sinusoidal_solution(form_rank, geometry)

                    # solve the problem
                    fₕ = L2_projection(fₑ, Λ[form_rank+1], ∫)
                    
                    # compute error
                    error = 0#L2_norm(fₕ - fₑ, ∫)
                    errors[ref_lev+1, p_idx, ss_idx, mesh_idx, form_rank+1] = error

                    display("   Error: $error")
                end
                println("...done!")
            end
        end
    end
end

# # GEOMETRY, SPACES & QUADRATURE -------------------------------------------------------------------
# @info("Setting up geometry, function spaces and quadrature rules")
# # geometry
# if mesh_type == "cartesian"
#     □ = unit_cube_cartesian(num_el)
# else
#     □ = unit_square_curvilinear(num_el)
# end

# # function spaces
# W, _, _, ∫ₐ, ∫ₑ = tensor_product_de_rham_complex(□, p⁰, num_el, L, section_space_type, θ)

# # EXACT SOLUTION -------------------------------------------------------------------
# # exact solution for the problem
# uₑ, _, _ = sinusoidal_solution(0, □)

# # SOLVE PROBLEM -------------------------------------------------------------------
# println("Solving the problem...")
# uₕ = run_L2_projection(W, ∫ₐ, uₑ)

# # COMPUTE ERROR -------------------------------------------------------------------
# println("Computing error...")
# error_u = L2_norm(uₕ - uₑ, ∫ₑ)
# println("Error in u: ", error_u)

# # VISUALIZE SOLUTION -------------------------------------------------------------------
# # println("Visualizing the solution...")
# # visualize_solution((uₕ, uₑ), ("uh", "ue"), "L2projection_$section_space_type _$mesh_type", □, 1, 4)

# println("...done!")

# import Mantis

# using Test
# using LinearAlgebra

# # This is how MANTIS can be called to solve a problem.
# function fe_run(weak_form_inputs, weak_form, bc_dirichlet, verbose)

#     # Setup the global assembler.

#     if verbose
#         println("Assembling ...")
#     end
#     A, b = Mantis.Assemblers.assemble(weak_form, weak_form_inputs, bc_dirichlet)

#     # Solve & add bcs.
#     if verbose
#         println("Solving ",size(A,1)," x ",size(A,2)," sized system ...")
#     end
#     sol = A \ b

#     return sol
# end

# function write_form_sol_to_file(form_sols, var_names, geom, p, k, case, n, verbose)
    
#     # This is for the plotting.
#     for (form_sol, var_name) in zip(form_sols, var_names)
#         if verbose
#             println("Writing form '$var_name' to file ...")
#         end
        
#         output_filename = "L2-Projection-p$p-k$k-case-"*case*"-var_$var_name.vtu"
#         #output_filename_error = "Poisson-Forms-$n-D-p$p-k$k-m$msave-case-"*case*"-error.vtu"

#         output_file = joinpath(output_data_folder, output_filename)
#         #output_file_error = joinpath(output_data_folder, output_filename_error)

#         if n == 1
#             out_deg = maximum([1, p])
#         else
#             out_deg = maximum([1, maximum(p)])
#         end
#         Mantis.Plot.plot(form_sol; vtk_filename = output_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
#     end
# end

# # Compute base directories for data input and output
# Mantis_folder = dirname(dirname(pathof(Mantis)))
# data_folder = joinpath(Mantis_folder, "test", "data")
# output_data_folder = joinpath(data_folder, "output", "Convergence") # Create this folder first if you haven't done so yet

# # Choose whether to write the output to a file, run the tests, and/or 
# # print progress statements. Make sure they are set as indicated when 
# # committing and that the grid is not much larger than 10x10
# write_to_output_file = false # false
# run_tests = true             # true
# verbose = false           # false

# # Setup the form spaces
# # First the 2D information for first step
# starting_point_2d = (0.0, 0.0)
# box_size_2d = (1.0, 1.0)
# num_elements_2d = (5, 5)
# degree_2d = (2, 2)
# regularity_2d = degree_2d .- 1

# # Quadrature rule
# q_rule_2d = Mantis.Quadrature.tensor_product_rule(degree_2d .+ 1, Mantis.Quadrature.gauss_legendre)

# # Forcing functions 
# function forcing_function_sine_2d(x::Matrix{Float64})
#     return [@. sinpi(x[:,1]) * sinpi(x[:,2])]
# end

# # Boundary conditions 
# bc_dirichlet_2d_empty = Dict{Int, Float64}()

# # Refinement conditions 
# n_subdivs_2d = (2, 2)
# h_steps = 4 # Steps used for h-refinement

# case = "sine_2d"

# run_tests ? h_errors = Vector{Float64}(undef, h_steps+1) : nothing

# for step ∈ 0:h_steps
#     verbose ? println("Step $step") : nothing
#     curr_nels = num_elements_2d .* (n_subdivs_2d .^ step)

#     geo_cart_2d = Mantis.Geometry.create_cartesian_geometry(starting_point_2d, box_size_2d, curr_nels)
#     tp_space_2d = Mantis.FunctionSpaces.create_bspline_space(starting_point_2d, box_size_2d, curr_nels, degree_2d, regularity_2d)

#     zero_form_space_2d = Mantis.Forms.FormSpace(0, geo_cart_2d, (tp_space_2d,), "α⁰")

#     f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, geo_cart_2d, "f")

#     weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
#     coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.l2_weak_form, bc_dirichlet_2d_empty, verbose)

#     # Assign coefficients to a form field
#     α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
#     α⁰.coefficients .= coeffs
#     l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_sine_2d, q_rule_2d, "L2")

#     run_tests ? h_errors[step+1] = l2_error : nothing
    
#     if verbose
#         print("Total L2 error 0-form: ")
#         println(l2_error)
#     end
#     if write_to_output_file
#         write_form_sol_to_file([α⁰, f⁰_sine_2d, α⁰-f⁰_sine_2d], ["zero_form_step-$step", "exact_solution_step-$step", "diff_step-$step"], geo_cart_2d, degree_2d, regularity_2d, case, 2, verbose)
#     end
# end
# if run_tests
#     error_rates = log.(Ref(2), h_errors[1:end-1]./h_errors[2:end])
#     if verbose
#         println("L2 error convergence rates: ")
#         println(error_rates)
#     end
#     @test isapprox(error_rates[end], minimum(degree_2d)+1, atol=5e-2)
# end