import Mantis

using Test
using LinearAlgebra

# This is how MANTIS can be called to solve a problem.
function fe_run(weak_form_inputs, weak_form, bc_dirichlet, case, test, verbose)

    # Setup the global assembler.

    if verbose
        println("Assembling ...")
    end
    A, b = Mantis.Assemblers.assemble(weak_form, weak_form_inputs, bc_dirichlet)

    # Solve & add bcs.
    if verbose
        println("Solving ",size(A,1)," x ",size(A,2)," sized system ...")
    end
    sol = A \ b

    return sol
end

function write_form_sol_to_file(form_sols, var_names, geom, p, k, case, n, verbose)
    
    # This is for the plotting.
    for (form_sol, var_name) in zip(form_sols, var_names)
        if verbose
            println("Writing form '$var_name' to file ...")
        end
        
        output_filename = "Poisson-Projection-p$p-k$k-case-"*case*"-var_$var_name.vtu"
        #output_filename_error = "Poisson-Forms-$n-D-p$p-k$k-m$msave-case-"*case*"-error.vtu"

        output_file = joinpath(output_data_folder, output_filename)
        #output_file_error = joinpath(output_data_folder, output_filename_error)

        if n == 1
            out_deg = maximum([1, p])
        else
            out_deg = maximum([1, maximum(p)])
        end
        Mantis.Plot.plot(form_sol; vtk_filename = output_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
    end
end

# Compute base directories for data input and output
Mantis_folder = dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Convergence") # Create this folder first if you haven't done so yet

# Choose whether to write the output to a file, run the tests, and/or 
# print progress statements. Make sure they are set as indicated when 
# committing and that the grid is not much larger than 10x10
write_to_output_file = false  # false
run_tests = true              # true
verbose = false             # false

# Setup the form spaces
# First the 2D information for first step
starting_point_2d = (0.0, 0.0)
box_size_2d = (1.0, 1.0)
num_elements_2d = (5, 5)
degree_2d = (2, 2)
regularity_2d = degree_2d .- 1

# Quadrature rule
q_rule_2d = Mantis.Quadrature.tensor_product_rule(degree_2d .+ 1, Mantis.Quadrature.gauss_legendre)

# Create the forcing function
function forcing_function_sine_2d(x::Matrix{Float64})
    return [@. 8.0 * pi^2 * sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2])]
end
# Create the exact_solution
function exact_sol_sine_2d(x::Matrix{Float64})
    return [@. sinpi(2.0 * x[:,1]) * sinpi(2.0 * x[:,2])]
end

# Refinement conditions 
n_subdivs_2d = (2, 2)
h_steps = 4 # Steps used for h-refinement
H_steps = 6 # Steps used for hierarchical refinements 
dorfler_parameter = 0.2 # Adaptivity parameter 

# Use this to choose the cases to run. One at a time is recommended.
# Available: ["sine2d-Dirichlet"] 
case = "sine2d-Dirichlet"
# Refinement schemes
#Available: ["h", "HB", "THB", "THB-Lchain"]
refinemenent_schemes = ["h"] # Only h gives proper rates. The other schemes should be used for visual checks only.

if verbose
    println("Running case "*case*" ...")
end
for scheme ∈ refinemenent_schemes
    if scheme == "h"
        if verbose
            println("Running refinement scheme h ...")
        end
        run_tests ? h_errors = Vector{Float64}(undef, h_steps+1) : nothing
        for step ∈ 0:h_steps
            verbose ? println("Step $step") : nothing
            curr_nels = num_elements_2d .* (n_subdivs_2d .^ step)

            geo_cart_2d = Mantis.Geometry.create_cartesian_geometry(starting_point_2d, box_size_2d, curr_nels)
            tp_space_2d = Mantis.FunctionSpaces.create_bspline_space(starting_point_2d, box_size_2d, curr_nels, degree_2d, regularity_2d)

            zero_form_space_2d = Mantis.Forms.FormSpace(0, geo_cart_2d, (tp_space_2d,), "α⁰")

            f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, geo_cart_2d, "f")
            f⁰_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, geo_cart_2d, "u")

            bc_dirichlet = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in tp_space_2d.dof_partition[1][j])
            weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
            coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet, case, run_tests, verbose)

            # Assign coefficients to a form field
            α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
            α⁰.coefficients .= coeffs
            l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_exact_sol, q_rule_2d, "L2")

            run_tests ? h_errors[step+1] = l2_error : nothing
            
            if verbose
                print("Total L2 error 0-form: ")
                println(l2_error)
            end
            if write_to_output_file
                write_form_sol_to_file([α⁰, f⁰_exact_sol, α⁰-f⁰_exact_sol], ["zero_form_scheme-$(scheme)_step-$step", "exact_solution_scheme-$(scheme)_step-$step", "diff_scheme-$(scheme)_step-$step"], geo_cart_2d, degree_2d, regularity_2d, case, 2, verbose)
            end
        end
        if run_tests
            error_rates = log.(Ref(2), h_errors[1:end-1]./h_errors[2:end])
            if verbose
                println("L2 error convergence rates: ")
                println(error_rates)
            end
            @test isapprox(error_rates[end], minimum(degree_2d)+1, atol=5e-2)
        end
    elseif scheme == "HB"
        if verbose
            println("Running refinement scheme HB ...")
        end
        run_tests ? H_errors = Vector{Float64}(undef, H_steps+1) : nothing

        # Initial space
        geo_cart_2d = Mantis.Geometry.create_cartesian_geometry(starting_point_2d, box_size_2d, num_elements_2d)
        tp_space_2d = Mantis.FunctionSpaces.create_bspline_space(starting_point_2d, box_size_2d, num_elements_2d, degree_2d, regularity_2d)
        
        HB_spaces = [tp_space_2d]
        HB_operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]

        HB_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(HB_spaces, HB_operators, [Int[]])
        HB_geo = Mantis.Geometry.get_parametric_geometry(HB_space)

        zero_form_space_2d = Mantis.Forms.FormSpace(0, HB_geo, (HB_space,), "α⁰")

        f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, HB_geo, "f")
        f⁰_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, HB_geo, "u")

        bc_dirichlet = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in HB_space.dof_partition[1][j])
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
        coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet, case, run_tests, verbose)

        # Assign coefficients to a form field
        α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
        α⁰.coefficients .= coeffs

        l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_exact_sol, q_rule_2d, "L2")
        run_tests ? H_errors[1] = l2_error : nothing

        if verbose
            print("Total L2 error 0-form: ")
            println(l2_error)
        end
        if write_to_output_file
            write_form_sol_to_file([α⁰, f⁰_exact_sol, α⁰-f⁰_exact_sol], ["zero_form_scheme-$(scheme)_step-0", "exact_solution_scheme-$(scheme)_step-0", "diff_scheme-$(scheme)_step-0"], geo_cart_2d, degree_2d, regularity_2d, case, 2, verbose)
        end

        err_per_element = Mantis.Assemblers.compute_error_per_element(α⁰, f⁰_exact_sol, q_rule_2d)
        
        for step ∈ 1:H_steps
            L = Mantis.FunctionSpaces.get_num_levels(HB_space)
            new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(HB_space.spaces[L], n_subdivs_2d)

            dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
            marked_domains = Mantis.FunctionSpaces.get_marked_domains(HB_space, dorfler_marking, new_operator, false)

            if length(marked_domains) > L
                push!(HB_spaces, new_space)
                push!(HB_operators, new_operator)
            end

            HB_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(HB_spaces, HB_operators, marked_domains)
            HB_geo = Mantis.Geometry.get_parametric_geometry(HB_space)

            zero_form_space_2d = Mantis.Forms.FormSpace(0, HB_geo, (HB_space,), "α⁰")

            f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, HB_geo, "f")
            f⁰_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, HB_geo, "u")

            bc_dirichlet = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in HB_space.dof_partition[1][j])
            weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
            coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet, case, run_tests, verbose)

            # Assign coefficients to a form field
            α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
            α⁰.coefficients .= coeffs

            l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_exact_sol, q_rule_2d, "L2")
            run_tests ? H_errors[step+1] = l2_error : nothing
            if verbose
                print("Total L2 error 0-form: ")
                println(l2_error)
            end
            if write_to_output_file
                write_form_sol_to_file([α⁰, f⁰_exact_sol, α⁰-f⁰_exact_sol], ["zero_form_scheme-$(scheme)_step-$step", "exact_solution_scheme-$(scheme)_step-$step", "diff_scheme-$(scheme)_step-$step"], HB_geo, degree_2d, regularity_2d, case, 2, verbose)
            end

            err_per_element = Mantis.Assemblers.compute_error_per_element(α⁰, f⁰_exact_sol, q_rule_2d)
        end
        if run_tests
            error_rates = log.(Ref(2), H_errors[1:end-1]./H_errors[2:end])
            if verbose
                println("L2 error convergence rates: ")
                println(error_rates)
            end
            #@test isapprox(error_rates[end], minimum(degree_2d)+1, atol=5e-2)
        end
    elseif scheme == "THB"
        if verbose
            println("Running refinement scheme HB ...")
        end
        run_tests ? H_errors = Vector{Float64}(undef, H_steps+1) : nothing

        # Initial space
        geo_cart_2d = Mantis.Geometry.create_cartesian_geometry(starting_point_2d, box_size_2d, num_elements_2d)
        tp_space_2d = Mantis.FunctionSpaces.create_bspline_space(starting_point_2d, box_size_2d, num_elements_2d, degree_2d, regularity_2d)
        
        THB_spaces = [tp_space_2d]
        THB_operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]

        THB_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(THB_spaces, THB_operators, [Int[]], true)
        THB_geo = Mantis.Geometry.get_parametric_geometry(THB_space)

        zero_form_space_2d = Mantis.Forms.FormSpace(0, THB_geo, (THB_space,), "α⁰")

        f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, THB_geo, "f")
        f⁰_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, THB_geo, "u")

        bc_dirichlet = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in THB_space.dof_partition[1][j])
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
        coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet, case, run_tests, verbose)

        # Assign coefficients to a form field
        α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
        α⁰.coefficients .= coeffs

        l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_exact_sol, q_rule_2d, "L2")
        run_tests ? H_errors[1] = l2_error : nothing

        if verbose
            print("Total L2 error 0-form: ")
            println(l2_error)
        end
        if write_to_output_file
            write_form_sol_to_file([α⁰, f⁰_exact_sol, α⁰-f⁰_exact_sol], ["zero_form_scheme-$(scheme)_step-0", "exact_solution_scheme-$(scheme)_step-0", "diff_scheme-$(scheme)_step-0"], geo_cart_2d, degree_2d, regularity_2d, case, 2, verbose)
        end

        err_per_element = Mantis.Assemblers.compute_error_per_element(α⁰, f⁰_exact_sol, q_rule_2d)
        
        for step ∈ 1:H_steps
            L = Mantis.FunctionSpaces.get_num_levels(THB_space)
            new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(THB_space.spaces[L], n_subdivs_2d)

            dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
            marked_domains = Mantis.FunctionSpaces.get_marked_domains(THB_space, dorfler_marking, new_operator, false)

            if length(marked_domains) > L
                push!(THB_spaces, new_space)
                push!(THB_operators, new_operator)
            end

            THB_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(THB_spaces, THB_operators, marked_domains, true)
            THB_geo = Mantis.Geometry.get_parametric_geometry(THB_space)

            zero_form_space_2d = Mantis.Forms.FormSpace(0, THB_geo, (THB_space,), "α⁰")

            f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, THB_geo, "f")
            f⁰_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, THB_geo, "u")

            bc_dirichlet = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in THB_space.dof_partition[1][j])
            weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
            coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet, case, run_tests, verbose)

            # Assign coefficients to a form field
            α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
            α⁰.coefficients .= coeffs

            l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_exact_sol, q_rule_2d, "L2")
            run_tests ? H_errors[step+1] = l2_error : nothing
            if verbose
                print("Total L2 error 0-form: ")
                println(l2_error)
            end
            if write_to_output_file
                write_form_sol_to_file([α⁰, f⁰_exact_sol, α⁰-f⁰_exact_sol], ["zero_form_scheme-$(scheme)_step-$step", "exact_solution_scheme-$(scheme)_step-$step", "diff_scheme-$(scheme)_step-$step"], THB_geo, degree_2d, regularity_2d, case, 2, verbose)
            end

            err_per_element = Mantis.Assemblers.compute_error_per_element(α⁰, f⁰_exact_sol, q_rule_2d)
        end
        if run_tests
            error_rates = log.(Ref(2), H_errors[1:end-1]./H_errors[2:end])
            if verbose
                println("L2 error convergence rates: ")
                println(error_rates)
            end
            #@test isapprox(error_rates[end], minimum(degree_2d)+1, atol=5e-2)
        end
    elseif scheme == "THB-Lchain"
        if verbose
            println("Running refinement scheme HB ...")
        end
        run_tests ? H_errors = Vector{Float64}(undef, H_steps+1) : nothing

        # Initial space
        geo_cart_2d = Mantis.Geometry.create_cartesian_geometry(starting_point_2d, box_size_2d, num_elements_2d)
        tp_space_2d = Mantis.FunctionSpaces.create_bspline_space(starting_point_2d, box_size_2d, num_elements_2d, degree_2d, regularity_2d)
        
        THB_spaces = [tp_space_2d]
        THB_operators = Mantis.FunctionSpaces.AbstractTwoScaleOperator[]

        THB_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(THB_spaces, THB_operators, [Int[]], true)
        THB_geo = Mantis.Geometry.get_parametric_geometry(THB_space)

        zero_form_space_2d = Mantis.Forms.FormSpace(0, THB_geo, (THB_space,), "α⁰")

        f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, THB_geo, "f")
        f⁰_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, THB_geo, "u")

        bc_dirichlet = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in THB_space.dof_partition[1][j])
        weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
        coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet, case, run_tests, verbose)

        # Assign coefficients to a form field
        α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
        α⁰.coefficients .= coeffs

        l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_exact_sol, q_rule_2d, "L2")
        run_tests ? H_errors[1] = l2_error : nothing

        if verbose
            print("Total L2 error 0-form: ")
            println(l2_error)
        end
        if write_to_output_file
            write_form_sol_to_file([α⁰, f⁰_exact_sol, α⁰-f⁰_exact_sol], ["zero_form_scheme-$(scheme)_step-0", "exact_solution_scheme-$(scheme)_step-0", "diff_scheme-$(scheme)_step-0"], geo_cart_2d, degree_2d, regularity_2d, case, 2, verbose)
        end

        err_per_element = Mantis.Assemblers.compute_error_per_element(α⁰, f⁰_exact_sol, q_rule_2d)
        
        for step ∈ 1:H_steps
            L = Mantis.FunctionSpaces.get_num_levels(THB_space)
            new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(THB_space.spaces[L], n_subdivs_2d)

            dorfler_marking = Mantis.FunctionSpaces.get_dorfler_marking(err_per_element, dorfler_parameter)
            marked_domains = Mantis.FunctionSpaces.get_marked_domains(THB_space, dorfler_marking, new_operator, true)

            if length(marked_domains) > L
                push!(THB_spaces, new_space)
                push!(THB_operators, new_operator)
            end

            THB_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(THB_spaces, THB_operators, marked_domains, true)
            THB_geo = Mantis.Geometry.get_parametric_geometry(THB_space)

            zero_form_space_2d = Mantis.Forms.FormSpace(0, THB_geo, (THB_space,), "α⁰")

            f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, THB_geo, "f")
            f⁰_exact_sol = Mantis.Forms.AnalyticalFormField(0, exact_sol_sine_2d, THB_geo, "u")

            bc_dirichlet = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in THB_space.dof_partition[1][j])
            weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
            coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet, case, run_tests, verbose)

            # Assign coefficients to a form field
            α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
            α⁰.coefficients .= coeffs

            l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_exact_sol, q_rule_2d, "L2")
            run_tests ? H_errors[step+1] = l2_error : nothing
            if verbose
                print("Total L2 error 0-form: ")
                println(l2_error)
            end
            if write_to_output_file
                write_form_sol_to_file([α⁰, f⁰_exact_sol, α⁰-f⁰_exact_sol], ["zero_form_scheme-$(scheme)_step-$step", "exact_solution_scheme-$(scheme)_step-$step", "diff_scheme-$(scheme)_step-$step"], THB_geo, degree_2d, regularity_2d, case, 2, verbose)
            end

            err_per_element = Mantis.Assemblers.compute_error_per_element(α⁰, f⁰_exact_sol, q_rule_2d)
        end
        if run_tests
            error_rates = log.(Ref(2), H_errors[1:end-1]./H_errors[2:end])
            if verbose
                println("L2 error convergence rates: ")
                println(error_rates)
            end
            #@test isapprox(error_rates[end], minimum(degree_2d)+1, atol=5e-2)
        end
    end
end
