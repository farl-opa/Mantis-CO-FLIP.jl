

import Mantis

using Test
using LinearAlgebra



function create_bspline_space(x_left, x_right, n_elements, p, k)
    breakpoints = collect(LinRange(x_left, x_right, n_elements+1))
    patch = Mantis.Mesh.Patch1D(breakpoints)
    
    kvec = fill(k, (n_elements+1,))
    kvec[1] = -1 # Open knot vector
    kvec[end] = -1

    return Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)
end




# This is how MANTIS can be called to solve a problem.
function fe_run(weak_form_inputs, weak_form, bc_dirichlet, geom, p, k, case, n, output_to_file, test, verbose)
    if verbose
        println("Running case "*case*" ...")
    end

    # Setup the global assembler.
    global_assembler = Mantis.Assemblers.Assembler(bc_dirichlet)

    if verbose
        println("Assembling ...")
    end
    A, b = global_assembler(weak_form, weak_form_inputs)

    # if n > 1 && isempty(bc_dirichlet)
    #     # Add the average = 0 condition for Neumann b.c. (derivatives are 
    #     # assumed to be zero!). Note that this only sums the coefficients.
    #     A = vcat(A, ones((1,size(A)[2])))
    #     A = hcat(A, vcat(ones(size(A)[1]-1), 0.0))
    #     b = vcat(b, 0.0)
    # end

    if test
        if verbose
            println("Running tests ...")
        end
        #@test isapprox(A, A', rtol=1e-12)  # Full system matrices need not be symmetric due to the boundary conditions.
        @test isempty(nullspace(Matrix(A)))  # Only works on dense matrices!
        @test LinearAlgebra.cond(Matrix(A)) < 1e10
    end

    # Solve & add bcs.
    if verbose
        println("Solving ",size(A,1)," x ",size(A,2)," sized system ...")
    end
    sol = A \ b

    # if n > 1 && isempty(bc_dirichlet)
    #     sol_rsh = reshape(sol[1:end-1], :, 1)
    # else
    sol_rsh = reshape(sol, :, 1)
    #end

    # This is for the plotting.
    # WARNING: This has to be updated to use the plotting for forms once available.
    if output_to_file
        if verbose
            println("Writing to file ...")
        end
        
        msave = Mantis.Geometry.get_num_elements(geom)
        output_filename = "Poisson-Forms-$n-D-p$p-k$k-elements$msave-case-"*case*".vtu"
        #output_filename_error = "Poisson-Forms-$n-D-p$p-k$k-m$msave-case-"*case*"-error.vtu"

        output_file = joinpath(output_data_folder, output_filename)
        #output_file_error = joinpath(output_data_folder, output_filename_error)
        if occursin("mixed", case)
            field = Mantis.Fields.FEMField(weak_form_inputs.space_trial_phi_2.fem_space[1], reshape(sol_rsh[Mantis.Forms.get_num_basis(weak_form_inputs.space_trial_u_1)+1:end], (:,1)))
        else
            field = Mantis.Fields.FEMField(weak_form_inputs.space_trial.fem_space[1], sol_rsh)
        end

        #field = Mantis.Fields.FEMField(trial_space, sol_rsh)
        if n == 1
            out_deg = maximum([1, p])
        else
            out_deg = maximum([1, maximum(p)])
        end
        Mantis.Plot.plot(geom, field; vtk_filename = output_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
        #Mantis.Plot.plot(geom, field, exact_sol; vtk_filename = output_file_error, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
    end

    # # Compute error
    # if verbose
    #     println("Computing L^2 error w.r.t. exact solution ...")
    # end
    # err_assembler = Mantis.Assemblers.AssemblerError(q_rule)
    # err = err_assembler(trial_space, sol_rsh, geom, exact_sol)
    # if verbose
    #     println("The L^2 error is: ",err)
    # end

    # if test
    #     if case == "const1d"
    #         if verbose
    #             println("Error tests ...")
    #         end
    #         @test err < 1e12
    #     end
    # end

    return sol
end







using InteractiveUtils
# Here we setup and specify the inputs to the FE problem.

# Compute base directories for data input and output
Mantis_folder = dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Poisson") # Create this folder first if you haven't done so yet.

# Choose whether to write the output to a file, run the tests, and/or 
# print progress statements. Make sure they are set as indicated when 
# committing and that the grid is not much larger than 10x10
write_to_output_file = false  # false
run_tests = true              # true
verbose = false               # false





# ########################################################################
# ## Test cases for the 1D Poisson problem.                             ##
# ########################################################################

if verbose
    println("Creating 1D Geometry and spaces ...")
end

# Dimension
n_1d = 1
# Number of elements.
m_1d = 5
# polynomial degree and inter-element continuity.
p_1d = 3
k_1d = 2
# Domain. The length of the domain is chosen so that the normal 
# derivatives of the exact solution are zero at the boundary. This is 
# the only Neumann b.c. that we can specify at the moment. These are 
# specified as constants to make sure that the forcing function can use 
# them while remaining type stable.
const Lleft_1d = 0.0
const Lright_1d = 2.0


function exact_sol_sine_1d(x::Float64)
    return sinpi(x)
end

# Create function spaces (b-splines here).
trial_space_1d = create_bspline_space(Lleft_1d, Lright_1d, m_1d, p_1d, k_1d)
test_space_1d = create_bspline_space(Lleft_1d, Lright_1d, m_1d, p_1d, k_1d)

trial_space_1d_pm1 = create_bspline_space(Lleft_1d, Lright_1d, m_1d, p_1d-1, k_1d-1)
test_space_1d_pm1 = create_bspline_space(Lleft_1d, Lright_1d, m_1d, p_1d-1, k_1d-1)

# Set Dirichlet boundary conditions.
# bc_sine_1d = Dict{Int, Float64}(i == 1 ? i => 0.0 : i => 1.0 for i in Mantis.FunctionSpaces.get_boundary_dof_indices(trial_space_1d))
# bc_const_1d = Dict{Int, Float64}(i => 0.0 for i in Mantis.FunctionSpaces.get_boundary_dof_indices(trial_space_1d))
bc_sine_1d = Dict{Int, Float64}(1 => 0.0, Mantis.FunctionSpaces.get_num_basis(trial_space_1d) => 1.0)
bc_const_1d = Dict{Int, Float64}(1 => 0.0, Mantis.FunctionSpaces.get_num_basis(trial_space_1d) => 0.0)
bc_const_1d_empty = Dict{Int, Float64}()

# Create the geometry.
brk_1d = collect(LinRange(Lleft_1d, Lright_1d, m_1d+1))
geom_1d = Mantis.Geometry.CartesianGeometry((brk_1d,))

# Setup the quadrature rule.
q_rule_1d = Mantis.Quadrature.tensor_product_rule((p_1d + 1,), Mantis.Quadrature.gauss_legendre)

# Create form spaces (both test and trial)
zero_form_space_trial_1d = Mantis.Forms.FormSpace(0, geom_1d, (trial_space_1d,), "φ")
zero_form_space_test_1d = Mantis.Forms.FormSpace(0, geom_1d, (test_space_1d,), "ϕ")
one_form_space_trial_1d = Mantis.Forms.FormSpace(1, geom_1d, (trial_space_1d_pm1,), "φ")
one_form_space_test_1d = Mantis.Forms.FormSpace(1, geom_1d, (test_space_1d_pm1,), "ϕ")

top_form_space_trial_1d = Mantis.Forms.FormSpace(n_1d, geom_1d, (trial_space_1d,), "φ")
top_form_space_test_1d = Mantis.Forms.FormSpace(n_1d, geom_1d, (test_space_1d,), "ϕ")

# Forcing forms
# Constant forcing
f⁰_const = Mantis.Forms.FormField(zero_form_space_trial_1d, "f")
f⁰_const.coefficients .= 1.0
f¹_const = Mantis.Forms.FormField(one_form_space_trial_1d, "f")
f¹_const.coefficients .= 1.0
# Sine forcing
# Define the function that we want.
function forcing_sine_1d(x::Float64)
    return pi^2 * sinpi(x) + (1.0 - sinpi(Lright_1d))*x
end
# This part requires the analytical form field so that we don't have to 
# do this manually. This also uses the old inner product function 
# because that can be used without forms.
# Find the required coefficients by L2 projection.
# weak_form_inputs = Mantis.Assemblers.WeakFormInputs(f⁰, zero_form_space_trial_1d, zero_form_space_test_1d, q_rule_1d)
# Af, bf = global_assembler(Mantis.Assemblers.l2_weak_form, weak_form_inputs)
# coeffs_forcing_sine_1d = Af \ bf
# create the forcing form.
# f⁰_sine = Mantis.Forms.FormField(zero_form_space_trial_1d, "f")
# f⁰_sine.coefficients .= coeffs_forcing_sine_1d





# ########################################################################
# ## Test cases for the 2D Poisson problem.                             ##
# ########################################################################

if verbose
    println("Creating 2D Geometry and spaces ...")
end

# Dimension
n_2d = 2
# Number of elements.
m_x = 4
m_y = 4
# polynomial degree and inter-element continuity.
p_2d = (8, 8)
k_2d = (7, 7)
# Domain. The length of the domain is chosen so that the normal 
# derivatives of the exact solution are zero at the boundary. This is 
# the only Neumann b.c. that we can specify at the moment.
const Lleft = 0.0#0.25
const Lright = 2.0#0.75
const Lbottom = 0.0#0.25
const Ltop = 2.0#0.75


function forcing_sine_2d(x::Float64, y::Float64)
    return 8.0 * pi^2 * sinpi(2.0 * x) * sinpi(2.0 * y)
end

function exact_sol_sine_2d(x::Float64, y::Float64)
    return sinpi(2.0 * x) * sinpi(2.0 * y)
end


# Create function spaces (b-splines here).
trial_space_x = create_bspline_space(Lleft, Lright, m_x, p_2d[1], k_2d[1])
trial_space_y = create_bspline_space(Lbottom, Ltop, m_y, p_2d[2], k_2d[2])
trial_space_x_pm1 = create_bspline_space(Lleft, Lright, m_x, p_2d[1]-1, k_2d[1]-1)
trial_space_y_pm1 = create_bspline_space(Lbottom, Ltop, m_y, p_2d[2]-1, k_2d[2]-1)

test_space_x = create_bspline_space(Lleft, Lright, m_x, p_2d[1], k_2d[1])
test_space_y = create_bspline_space(Lbottom, Ltop, m_y, p_2d[2], k_2d[2])
test_space_x_pm1 = create_bspline_space(Lleft, Lright, m_x, p_2d[1]-1, k_2d[1]-1)
test_space_y_pm1 = create_bspline_space(Lbottom, Ltop, m_y, p_2d[2]-1, k_2d[2]-1)

trial_space_2d_volume = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x_pm1, trial_space_y_pm1)
test_space_2d_volume = Mantis.FunctionSpaces.TensorProductSpace(test_space_x_pm1, test_space_y_pm1)

trial_space_2d_1_form_x = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x_pm1, trial_space_y)
trial_space_2d_1_form_y = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x, trial_space_y_pm1)
test_space_2d_1_form_x = Mantis.FunctionSpaces.TensorProductSpace(test_space_x_pm1, test_space_y)
test_space_2d_1_form_y = Mantis.FunctionSpaces.TensorProductSpace(test_space_x, test_space_y_pm1)

trial_space_2d = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x, trial_space_y)
test_space_2d = Mantis.FunctionSpaces.TensorProductSpace(test_space_x, test_space_y)

# Set Dirichlet boundary conditions to zero.
bc_dirichlet_2d = Dict{Int, Float64}(i => 0.0 for j in [1, 2, 3, 4, 6, 7, 8, 9] for i in trial_space_2d.dof_partition[1][j])
bc_dirichlet_2d_empty = Dict{Int, Float64}()

# Create the geometry.
brk_2d_x = collect(LinRange(Lleft, Lright, m_x+1))
brk_2d_y = collect(LinRange(Lbottom, Ltop, m_y+1))
geom_cartesian = Mantis.Geometry.CartesianGeometry((brk_2d_x, brk_2d_y))

const crazy_c = 0.3
function mapping(x::Vector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [x[1] + ((Lright-Lleft)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new), x[2] + ((Ltop-Lbottom)/2.0)*crazy_c*sinpi(x1_new)*sinpi(x2_new)]
end
function dmapping(x::Vector{Float64})
    x1_new = (2.0/(Lright-Lleft))*x[1] - 2.0*Lleft/(Lright-Lleft) - 1.0
    x2_new = (2.0/(Ltop-Lbottom))*x[2] - 2.0*Lbottom/(Ltop-Lbottom) - 1.0
    return [1.0 + pi*crazy_c*cospi(x1_new)*sinpi(x2_new) ((Lright-Lleft)/(Ltop-Lbottom))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new); ((Ltop-Lbottom)/(Lright-Lleft))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0 + pi*crazy_c*sinpi(x1_new)*cospi(x2_new)]
end
dimension = (n_2d, n_2d)
curved_mapping = Mantis.Geometry.Mapping(dimension, mapping, dmapping)
geom_crazy = Mantis.Geometry.MappedGeometry(geom_cartesian, curved_mapping)


# Setup the quadrature rule.
q_rule_2d = Mantis.Quadrature.tensor_product_rule((p_2d[1] + 1, p_2d[2] + 1), Mantis.Quadrature.gauss_legendre)

# Create form spaces (both test and trial)
# Cartesian mesh
zero_form_space_trial_2d_cart = Mantis.Forms.FormSpace(0, geom_cartesian, (trial_space_2d,), "φ")
zero_form_space_test_2d_cart = Mantis.Forms.FormSpace(0, geom_cartesian, (test_space_2d,), "ϕ")

two_form_space_trial_2d_cart = Mantis.Forms.FormSpace(2, geom_cartesian, (trial_space_2d_volume,), "φ")
two_form_space_test_2d_cart = Mantis.Forms.FormSpace(2, geom_cartesian, (test_space_2d_volume,), "ϕ")

one_form_space_trial_2d_cart = Mantis.Forms.FormSpace(1, geom_cartesian, (trial_space_2d_1_form_x, trial_space_2d_1_form_y), "u")
one_form_space_test_2d_cart = Mantis.Forms.FormSpace(1, geom_cartesian, (test_space_2d_1_form_x, test_space_2d_1_form_y), "q")

# Crazy mesh
zero_form_space_trial_2d_crazy = Mantis.Forms.FormSpace(0, geom_crazy, (trial_space_2d,), "φ")
zero_form_space_test_2d_crazy = Mantis.Forms.FormSpace(0, geom_crazy, (test_space_2d,), "ϕ")

two_form_space_trial_2d_crazy = Mantis.Forms.FormSpace(2, geom_crazy, (trial_space_2d_volume,), "φ")
two_form_space_test_2d_crazy = Mantis.Forms.FormSpace(2, geom_crazy, (test_space_2d_volume,), "ϕ")

one_form_space_trial_2d_crazy = Mantis.Forms.FormSpace(1, geom_crazy, (trial_space_2d_1_form_x, trial_space_2d_1_form_y), "u")
one_form_space_test_2d_crazy = Mantis.Forms.FormSpace(1, geom_crazy, (test_space_2d_1_form_x, test_space_2d_1_form_y), "q")

# Create the forcing forms
f⁰_cart = Mantis.Forms.FormField(zero_form_space_trial_2d_cart, "f")  # Forcing function
f⁰_cart.coefficients .= 1.0

f⁰_crazy = Mantis.Forms.FormField(zero_form_space_trial_2d_crazy, "f")  # Forcing function
f⁰_crazy.coefficients .= 1.0

f²_cart = Mantis.Forms.FormField(two_form_space_trial_2d_cart, "f")  # Forcing function
f²_cart.coefficients .= 1.0

f²_crazy = Mantis.Forms.FormField(two_form_space_trial_2d_crazy, "f")  # Forcing function
f²_crazy.coefficients .= 1.0





# ########################################################################
# ## Test cases for the 3D Poisson problem.                             ##
# ########################################################################

# if verbose
#     println("Creating 3D Geometry and spaces ...")
# end

# # Dimension
# n_3d = 3
# # Number of elements.
# m_3d_x = 5
# m_3d_y = 5
# m_3d_z = 5
# # polynomial degree and inter-element continuity.
# p_3d = (3, 4, 1)
# k_3d = (2, 2, 0)
# # Domain. The length of the domain is chosen so that the normal 
# # derivatives of the exact solution are zero at the boundary. This is 
# # the only Neumann b.c. that we can specify at the moment.
# const Lx1 = 0.0
# const Lx2 = 1.0
# const Ly1 = 0.0
# const Ly2 = 1.0
# const Lz1 = 0.0
# const Lz2 = 1.0


# function forcing_sine_3d(x::Float64, y::Float64, z::Float64)
#     return 12.0 * pi^2 * sinpi(2.0 * x) * sinpi(2.0 * y) * sinpi(2.0 * z)
# end

# function exact_sol_sine_3d(x::Float64, y::Float64, z::Float64)
#     return sinpi(2.0 * x) * sinpi(2.0 * y) * sinpi(2.0 * z)
# end



# # Tensor product b-spline case on a Cartesian geometry.
# # Create Patch.
# brk_3d_x = collect(LinRange(Lx1, Lx2, m_3d_x+1))
# brk_3d_y = collect(LinRange(Ly1, Ly2, m_3d_y+1))
# brk_3d_z = collect(LinRange(Lz1, Lz2, m_3d_z+1))
# patch_3d_x = Mantis.Mesh.Patch1D(brk_3d_x)
# patch_3d_y = Mantis.Mesh.Patch1D(brk_3d_y)
# patch_3d_z = Mantis.Mesh.Patch1D(brk_3d_z)
# # Continuity vector for OPEN knot vector.
# kvec_3d_x = fill(k_3d[1], (m_3d_x+1,))
# kvec_3d_x[1] = -1
# kvec_3d_x[end] = -1
# kvec_3d_y = fill(k_3d[2], (m_3d_y+1,))
# kvec_3d_y[1] = -1
# kvec_3d_y[end] = -1
# kvec_3d_z = fill(k_3d[3], (m_3d_z+1,))
# kvec_3d_z[1] = -1
# kvec_3d_z[end] = -1
# # Create function spaces (b-splines here).
# trial_space_3d_x = Mantis.FunctionSpaces.BSplineSpace(patch_3d_x, p_3d[1], kvec_3d_x)
# test_space_3d_x = Mantis.FunctionSpaces.BSplineSpace(patch_3d_x, p_3d[1], kvec_3d_x)
# trial_space_3d_y = Mantis.FunctionSpaces.BSplineSpace(patch_3d_y, p_3d[2], kvec_3d_y)
# test_space_3d_y = Mantis.FunctionSpaces.BSplineSpace(patch_3d_y, p_3d[2], kvec_3d_y)
# trial_space_3d_z = Mantis.FunctionSpaces.BSplineSpace(patch_3d_z, p_3d[3], kvec_3d_z)
# test_space_3d_z = Mantis.FunctionSpaces.BSplineSpace(patch_3d_z, p_3d[3], kvec_3d_z)

# trial_space_3d_xy = Mantis.FunctionSpaces.TensorProductSpace(trial_space_3d_x, trial_space_3d_y)
# test_space_3d_xy = Mantis.FunctionSpaces.TensorProductSpace(test_space_3d_x, test_space_3d_y)

# trial_space_3d = Mantis.FunctionSpaces.TensorProductSpace(trial_space_3d_xy, trial_space_3d_z)
# test_space_3d = Mantis.FunctionSpaces.TensorProductSpace(test_space_3d_xy, test_space_3d_z)

# # Set Dirichlet boundary conditions to zero.
# bc_dirichlet_3d = Dict{Int, Float64}(i => 0.0 for i in Mantis.FunctionSpaces.get_boundary_dof_indices(trial_space_3d))

# # Create the geometry.
# geom_3d_cartesian = Mantis.Geometry.CartesianGeometry((brk_3d_x, brk_3d_y, brk_3d_z))

# # Setup the quadrature rule.
# q_nodes_3d, q_weights_3d = Mantis.Quadrature.tensor_product_rule(p_3d .+ 1, Mantis.Quadrature.gauss_legendre)




# Running all testcases.
println()
#cases = ["sine1d", "const1d"]#, "sine2d-Dirichlet", "sine2d-Neumann", "sine2d-crazy-Dirichlet", "sine2d-crazy-Neumann", "sine2dH-Dirichlet", "sine2dH-Neumann", "sine3d-Dirichlet"]
cases = ["const1d-Dirichlet", "const1d-Dirichlet-mixed", "const2d-Dirichlet", "const2d-Dirichlet-crazy", "const2d-Dirichlet-mixed", "const2d-Dirichlet-mixed-crazy"]
for case in cases

    if case == "const1d-Dirichlet"
        weak_form_inputs_const1dD = Mantis.Assemblers.WeakFormInputs(f⁰_const, zero_form_space_trial_1d, zero_form_space_test_1d, q_rule_1d)
        fe_run(weak_form_inputs_const1dD, Mantis.Assemblers.poisson_non_mixed, bc_const_1d, geom_1d, p_1d, k_1d, case, n_1d, write_to_output_file, run_tests, verbose)
    elseif case == "const1d-Dirichlet-mixed"
        weak_form_inputs_const1dDm = Mantis.Assemblers.WeakFormInputsMixed(f¹_const, zero_form_space_trial_1d, one_form_space_trial_1d, zero_form_space_test_1d, one_form_space_test_1d, q_rule_1d)
        fe_run(weak_form_inputs_const1dDm, Mantis.Assemblers.poisson_mixed, bc_const_1d_empty, geom_1d, p_1d, k_1d, case, n_1d, write_to_output_file, run_tests, verbose)
    elseif case == "const2d-Dirichlet"
        weak_form_inputs_const2dD = Mantis.Assemblers.WeakFormInputs(f⁰_cart, zero_form_space_trial_2d_cart, zero_form_space_test_2d_cart, q_rule_2d)
        fe_run(weak_form_inputs_const2dD, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d, geom_cartesian, p_2d, k_2d, case, n_2d, write_to_output_file, run_tests, verbose)
    elseif case == "const2d-Dirichlet-crazy"
        weak_form_inputs_const2dD = Mantis.Assemblers.WeakFormInputs(f⁰_crazy, zero_form_space_trial_2d_crazy, zero_form_space_test_2d_crazy, q_rule_2d)
        fe_run(weak_form_inputs_const2dD, Mantis.Assemblers.poisson_non_mixed, bc_dirichlet_2d, geom_crazy, p_2d, k_2d, case*"_crazy_c$crazy_c", n_2d, write_to_output_file, run_tests, verbose)
    elseif case == "const2d-Dirichlet-mixed"
        weak_form_inputs_const2dDm = Mantis.Assemblers.WeakFormInputsMixed(f²_cart, one_form_space_trial_2d_cart, two_form_space_trial_2d_cart, one_form_space_test_2d_cart, two_form_space_test_2d_cart, q_rule_2d)
        fe_run(weak_form_inputs_const2dDm, Mantis.Assemblers.poisson_mixed, bc_dirichlet_2d_empty, geom_cartesian, p_2d, k_2d, case, n_2d, write_to_output_file, run_tests, verbose)
    elseif case == "const2d-Dirichlet-mixed-crazy"
        weak_form_inputs_const2dDmc = Mantis.Assemblers.WeakFormInputsMixed(f²_crazy, one_form_space_trial_2d_crazy, two_form_space_trial_2d_crazy, one_form_space_test_2d_crazy, two_form_space_test_2d_crazy, q_rule_2d)
        fe_run(weak_form_inputs_const2dDmc, Mantis.Assemblers.poisson_mixed, bc_dirichlet_2d_empty, geom_crazy, p_2d, k_2d, case*"_crazy_c$crazy_c", n_2d, write_to_output_file, run_tests, verbose)
    else
        if verbose
            println("Warning: case '"*case*"' unknown. Skipping.") 
        end
    end
    if verbose
        println()  # Extra blank line to separate the different runs.
    end
end


