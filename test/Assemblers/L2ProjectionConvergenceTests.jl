import Mantis

using Test
using LinearAlgebra

# This is how MANTIS can be called to solve a problem.
function fe_run(weak_form_inputs, weak_form, bc_dirichlet, verbose)

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
        
        output_filename = "L2-Projection-p$p-k$k-case-"*case*"-var_$var_name.vtu"
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
write_to_output_file = false # false
run_tests = true             # true
verbose = false           # false

# Setup the form spaces
# First the 2D information for first step
starting_point_2d = (0.0, 0.0)
box_size_2d = (1.0, 1.0)
num_elements_2d = (5, 5)
degree_2d = (2, 2)
regularity_2d = degree_2d .- 1

# Quadrature rule
q_rule_2d = Mantis.Quadrature.tensor_product_rule(degree_2d .+ 1, Mantis.Quadrature.gauss_legendre)

# Forcing functions 
function forcing_function_sine_2d(x::Matrix{Float64})
    return [@. sinpi(x[:,1]) * sinpi(x[:,2])]
end

# Boundary conditions 
bc_dirichlet_2d_empty = Dict{Int, Float64}()

# Refinement conditions 
n_subdivs_2d = (2, 2)
h_steps = 4 # Steps used for h-refinement

case = "sine_2d"

run_tests ? h_errors = Vector{Float64}(undef, h_steps+1) : nothing

for step ∈ 0:h_steps
    verbose ? println("Step $step") : nothing
    curr_nels = num_elements_2d .* (n_subdivs_2d .^ step)

    geo_cart_2d = Mantis.Geometry.create_cartesian_geometry(starting_point_2d, box_size_2d, curr_nels)
    tp_space_2d = Mantis.FunctionSpaces.create_bspline_space(starting_point_2d, box_size_2d, curr_nels, degree_2d, regularity_2d)

    zero_form_space_2d = Mantis.Forms.FormSpace(0, geo_cart_2d, (tp_space_2d,), "α⁰")

    f⁰_sine_2d = Mantis.Forms.AnalyticalFormField(0, forcing_function_sine_2d, geo_cart_2d, "f")

    weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰_sine_2d, zero_form_space_2d, zero_form_space_2d, q_rule_2d)
    coeffs = fe_run(weak_form_inputs, Mantis.Assemblers.l2_weak_form, bc_dirichlet_2d_empty, verbose)

    # Assign coefficients to a form field
    α⁰ = Mantis.Forms.FormField(zero_form_space_2d, "α⁰")
    α⁰.coefficients .= coeffs
    l2_error = Mantis.Assemblers.compute_error_total(α⁰, f⁰_sine_2d, q_rule_2d, "L2")

    run_tests ? h_errors[step+1] = l2_error : nothing
    
    if verbose
        print("Total L2 error 0-form: ")
        println(l2_error)
    end
    if write_to_output_file
        write_form_sol_to_file([α⁰, f⁰_sine_2d, α⁰-f⁰_sine_2d], ["zero_form_step-$step", "exact_solution_step-$step", "diff_step-$step"], geo_cart_2d, degree_2d, regularity_2d, case, 2, verbose)
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