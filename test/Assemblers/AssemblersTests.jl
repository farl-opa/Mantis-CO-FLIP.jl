

import Mantis

using Test
using LinearAlgebra


# # This is how MANTIS is called to create a 1D mixed problem.
# function fe_run_1D(forcing_function, bilinear_function, linear_function, trial_space, 
#                    test_space, geom, q_nodes, q_weights, bc_left, bc_right, p, 
#                    k, case, output_to_file, tests=true)
#     # Setup the element assembler.
#     element_assembler = Mantis.Assemblers.PoissonBilinearForm1D(forcing_function,
#                                                                 bilinear_function,
#                                                                 linear_function,
#                                                                 trial_space,
#                                                                 test_space,
#                                                                 geom,
#                                                                 q_nodes,
#                                                                 q_weights)

#     # Setup the global assembler.
#     global_assembler = Mantis.Assemblers.Assembler(bc_left, bc_right)

#     # Set boundary conditions
#     #bc_assembler = Mantis.Assemblers.Dirichlet()

#     # Assemble.
#     A, b = global_assembler(element_assembler)#, bc_assembler)

#     # Solve & add bcs.
#     #sol = A \ Vector(b)
#     sol = [bc_left, (A \ Vector(b))..., bc_right]

#     if tests
#         @test isapprox(A, A', rtol=1e-12)
#         @test isempty(nullspace(Matrix(A)))

#         if case == "const" && p >= 2
#             # The exact solution is in the space, so we can test for 
#             # equality to the exact solution.
#             n_points = 10
#             m = Mantis.Geometry.get_num_elements(geom)
#             xi = collect(LinRange(0.0, 1.0, n_points))
#             exact_sol = x -> x.*(x.-m)
#             for element_id in 1:m
#                 x = Mantis.Geometry.evaluate(geom, element_id, (xi,))
#                 basis, active_bases = Mantis.FunctionSpaces.evaluate(trial_space, element_id, (xi,), 0)
#                 exact = exact_sol(x)
#                 #@test all(isapprox.(exact .- reshape(sol[active_bases]' * basis[0]', n_points), 0.0, atol=1e-12))
#             end
#         end
#     end


#     # This is for the plotting. If you want to output the data into vtu-
#     # files, uncomment these lines. You can visualise the solution in 
#     # Paraview, using the 'Plot over line'-filter.
#     if output_to_file
#         msave = Mantis.Geometry.get_num_elements(geom)
#         output_filename = "Poisson-1D-p$p-k$k-m$msave-0to1-case-"*case*".vtu"
#         output_file = joinpath(output_data_folder, output_filename)
#         field = Mantis.Fields.FEMField(trial_space, reshape(sol, :, 1))
#         Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = p, ascii = false, compress = false)
#     end
# end

# # Here we setup and specify the inputs to the FE problem.
# function forcing_sine(x)
#     return pi^2 * sinpi(x)
# end

# function forcing_const(x)
#     return -2.0
# end

# function bilinear_function(Ndi, Ndj)
#     return Ndi * Ndj
# end

# function linear_function(Ni, f)
#     return Ni * f
# end


# # Number of elements.
# m = 10
# # polynomial degree and inter-element continuity.
# p = 5
# k = 3
# # Domain.
# Lleft = 0.0
# Lright = 1.0

# # Create Patch.
# brk = collect(LinRange(Lleft, Lright, m+1))
# patch = Mantis.Mesh.Patch1D(brk)
# # Continuity vector for OPEN knot vector.
# kvec = fill(k, (m+1,))
# kvec[1] = -1
# kvec[end] = -1
# # Create function spaces (b-splines here).
# trial_space = Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)
# test_space = Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)

# # Create the geometry.
# geom_1d = Mantis.Geometry.CartesianGeometry((brk,))

# # Setup the quadrature rule.
# q_nodes, q_weights = Mantis.Quadrature.gauss_legendre(p+1)

# # Choose case ("sine" or "const")
# for case in ["sine", "const"]
#     if case == "sine"
#         # Boundary conditions (Dirichlet).
#         bc_left = 0.0
#         bc_right = 1.0
#         fe_run_1D(forcing_sine, bilinear_function, linear_function, trial_space, 
#                   test_space, geom_1d, q_nodes, q_weights, bc_left, bc_right, p, 
#                   k, case, write_to_output_file)
#     elseif case == "const"
#         # Boundary conditions (Dirichlet).
#         bc_left = 0.0
#         bc_right = 0.0
#         fe_run_1D(forcing_const, bilinear_function, linear_function, trial_space, 
#                   test_space, geom_1d, q_nodes, q_weights, bc_left, bc_right, p, 
#                   k, case, write_to_output_file)
#     else
#         error("Case: '",case,"' unknown.")
#     end
# end





# ########################################################################
# ## Test cases for the mixed 1D Poisson problem.                       ##
# ########################################################################

# # This is how MANTIS is called to create a 1D mixed problem.
# function fe_run_mixed(forcing_function_mixed, Fbilinear1, Fbilinear2, 
#                       Fbilinear3, Flinear, trial_space_phi, test_space_phi, 
#                       trial_space_sigma, test_space_sigma, geom, q_nodes, 
#                       q_weights, p_sigma, k_sigma, p_phi, k_phi, case, 
#                       output_to_file)
#     # Setup the element assembler.
#     element_assembler = Mantis.Assemblers.PoissonBilinearFormMixed1D(forcing_function_mixed,
#                                                                      Fbilinear1,
#                                                                      Fbilinear2,
#                                                                      Fbilinear3,
#                                                                      Flinear,
#                                                                      trial_space_phi,
#                                                                      test_space_phi,
#                                                                      trial_space_sigma,
#                                                                      test_space_sigma,
#                                                                      geom,
#                                                                      q_nodes,
#                                                                      q_weights)

#     # Setup the global assembler.
#     global_assembler = Mantis.Assemblers.Assembler()

#     # Assemble.
#     A, b = global_assembler(element_assembler)

#     # Solve & add bcs.
#     sol = A \ Vector(b)

#     # This is for the plotting. You can visualise the solution in 
#     # Paraview, using the 'Plot over line'-filter.
#     if output_to_file
#         ndofs_sigma = Mantis.FunctionSpaces.get_dim(trial_space_sigma)

#         msave = Mantis.Geometry.get_num_elements(geom)

#         output_filename = "Poisson-1D-Mixed-phi-p$p_sigma-k$k_sigma-m$msave-case-"*case*".vtu"
#         output_file = joinpath(output_data_folder, output_filename)
#         field = Mantis.Fields.FEMField(trial_space_phi, reshape(sol[ndofs_sigma+1:end], :, 1))
#         Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = maximum([1, p_phi]), ascii = false, compress = false)

#         output_filename = "Poisson-1D-Mixed-sigma-p$p_phi-k$k_phi-m$msave-case-"*case*".vtu"
#         output_file = joinpath(output_data_folder, output_filename)
#         field = Mantis.Fields.FEMField(trial_space_sigma, reshape(sol[begin:ndofs_sigma], :, 1))
#         Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = p_sigma, ascii = false, compress = false)
#     end

#     return sol
# end



# # Here we setup and specify the inputs to the FE problem.
# function function_der_function(Ndi, Nj)
#     return Ndi * Nj
# end

# function function_function(Ni, f)
#     return Ni * f
# end


# # Number of elements.
# m = 14
# # polynomial degree and inter-element continuity.
# p_phi = 0
# k_phi = -1
# p_sigma = 1
# k_sigma = 0
# # Domain.
# Lleft = 0.0
# Lright = float(m)


# function forcing_sine_mixed(x)
#     return (pi/Lright)^2 * sin(pi * x / Lright)
# end

# function forcing_const_mixed(x)
#     return -2.0
# end

# # Create Patch.
# brk = collect(LinRange(Lleft, Lright, m+1))
# patch = Mantis.Mesh.Patch1D(brk)
# # Continuity vector for OPEN knot vector.
# kvec_phi = fill(k_phi, (m+1,))
# kvec_phi[1] = -1
# kvec_phi[end] = -1
# kvec_sigma = fill(k_sigma, (m+1,))
# kvec_sigma[1] = -1
# kvec_sigma[end] = -1
# # Create function spaces (b-splines here).
# trial_space_phi = Mantis.FunctionSpaces.BSplineSpace(patch, p_phi, kvec_phi)
# test_space_phi = Mantis.FunctionSpaces.BSplineSpace(patch, p_phi, kvec_phi)
# trial_space_sigma = Mantis.FunctionSpaces.BSplineSpace(patch, p_sigma, kvec_sigma)
# test_space_sigma = Mantis.FunctionSpaces.BSplineSpace(patch, p_sigma, kvec_sigma)

# # Create the geometry.
# geom = Mantis.Geometry.CartesianGeometry((brk,))

# # Setup the quadrature rule.
# q_nodes, q_weights = Mantis.Quadrature.gauss_legendre(maximum([p_phi, p_sigma])+1)

# # Choose case ("sine" or "const")
# for case in ["sine", "const"]
    
#     if case == "sine"
#         fe_run_mixed(forcing_sine_mixed, function_function, function_der_function, 
#                      function_der_function, function_function, trial_space_phi, 
#                      test_space_phi, trial_space_sigma, test_space_sigma, geom, 
#                      q_nodes, q_weights, p_sigma, k_sigma, p_phi, k_phi, case, write_to_output_file)
#     elseif case == "const"
#         fe_run_mixed(forcing_const_mixed, function_function, function_der_function, 
#                      function_der_function, function_function, trial_space_phi, 
#                      test_space_phi, trial_space_sigma, test_space_sigma, geom, 
#                      q_nodes, q_weights, p_sigma, k_sigma, p_phi, k_phi, case, write_to_output_file)
#     else
#         error("Case: '",case,"' unknown.")
#     end
# end












# This is how MANTIS is called to solve a problem. The bc input is only for the 1D case.
function fe_run(forcing_function, trial_space, test_space, geom, q_nodes, 
                q_weights, exact_sol, p, k, case, n, output_to_file, test=true, 
                verbose=false, bc = (false, 0.0, 0.0))
    if verbose
        println("Starting setup for case "*case*" ...")
    end
    # Setup the element assembler.
    element_assembler = Mantis.Assemblers.PoissonBilinearForm(forcing_function,
                                                              trial_space,
                                                              test_space,
                                                              geom,
                                                              q_nodes,
                                                              q_weights)

    # Setup the global assembler.
    if bc[1] == false
        global_assembler = Mantis.Assemblers.Assembler()
    else
        global_assembler = Mantis.Assemblers.Assembler(bc[2], bc[3])
    end

    if verbose
        println("Assembling ...")
    end
    A, b = global_assembler(element_assembler)

    if n != 1
        # Add the average = 0 condition for Neumann b.c. (derivatives are 
        # assumed to be zero!)
        A = vcat(A, ones((1,size(A)[2])))
        A = hcat(A, vcat(ones(size(A)[1]-1), 0.0))
        b = vcat(b, 0.0)
    end

    #@code_warntype element_assembler(1)
    

    if test
        if verbose
            println("Running tests ...")
        end
        @test isapprox(A, A', rtol=1e-12)
        @test isempty(nullspace(Matrix(A)))  # Only works on dense matrices!
        @test LinearAlgebra.cond(Matrix(A)) < 1e10
    end

    # Solve & add bcs.
    if verbose
        println("Solving ",size(A,1)," x ",size(A,2)," sized system ...")
    end
    if n == 1
        sol = [bc[2], (A \ Vector(b))..., bc[3]]
    else
        sol = A \ Vector(b)
    end

    if n == 1
        sol_rsh = reshape(sol, :, 1)
    else
        sol_rsh = reshape(sol[1:end-1], :, 1)
    end


    # This is for the plotting. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.
    if output_to_file
        if verbose
            println("Writing to file ...")
        end
        
        msave = Mantis.Geometry.get_num_elements(geom)
        output_filename = "Poisson-$n-D-p$p-k$k-m$msave-case-"*case*".vtu"
        output_file = joinpath(output_data_folder, output_filename)
        field = Mantis.Fields.FEMField(trial_space, sol_rsh)
        if n == 1
            out_deg = maximum([1, p])
        else
            out_deg = maximum([1, maximum(p)])
        end
        Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = out_deg, ascii = false, compress = false)
    end

    # Compute error
    if verbose
        println("Computing L^2 error w.r.t. exact solution ...")
    end
    err_assembler = Mantis.Assemblers.AssemblerError(q_nodes, q_weights)
    err = err_assembler(trial_space, sol_rsh, geom, exact_sol)
    if verbose
        println("The L^2 error is: ",err)
        println()  # Extra blank line to separate the different runs.
    end

    # Extra check to test if the metric computation was correct.
    # err2 = err_assembler(trial_space, reshape(ones(length(sol)-1), :, 1), geom, (x,y) -> 0.0)
    # println(err2)

    return sol
end








# Here we setup and specify the inputs to the FE problem.

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Poisson") # Create this folder first if you haven't done so yet.

# Choose whether to write the output to a file, run the tests, and/or 
# print progress statements. Make sure they are set as indicated when 
# committing and that the grid is not much larger than 10x10
write_to_output_file = false  # false
run_tests = true              # true
verbose = false               # false






########################################################################
## Test cases for the 1D Poisson problem.                             ##
########################################################################

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
const Lright_1d = 1.0

bc_sine_1d = (true, 0.0, 1.0)
bc_const_1d = (true, 0.0, 0.0)

function forcing_sine_1d(x::Float64)
    return pi^2 * sinpi(x) + (1.0 - sinpi(Lright_1d))*x
end

function exact_sol_sine_1d(x::Float64)
    return sinpi(x)
end

function forcing_const_1d(x)
    return -2.0
end

function exact_sol_const_1d(x::Float64)
    return x * (x - Lright_1d)
end

# Create Patch.
brk_1d = collect(LinRange(Lleft_1d, Lright_1d, m_1d+1))
patch_1d = Mantis.Mesh.Patch1D(brk_1d)
# Continuity vector for OPEN knot vector.
kvec_1d = fill(k_1d, (m_1d+1,))
kvec_1d[1] = -1
kvec_1d[end] = -1
# Create function spaces (b-splines here).
trial_space_1d = Mantis.FunctionSpaces.BSplineSpace(patch_1d, p_1d, kvec_1d)
test_space_1d = Mantis.FunctionSpaces.BSplineSpace(patch_1d, p_1d, kvec_1d)

# Create the geometry.
geom_1d = Mantis.Geometry.CartesianGeometry((brk_1d,))

# Setup the quadrature rule.
q_nodes_1d, q_weights_1d = Mantis.Quadrature.gauss_legendre(p_1d+1)
q_weights_1d_all = Mantis.Quadrature.tensor_product_weights((q_weights_1d,)) # Simply returns q_weights_1d



########################################################################
## Test cases for the 2D Poisson problem.                             ##
########################################################################

# Dimension
n_2d = 2
# Number of elements.
m_x = 5
m_y = 5
# polynomial degree and inter-element continuity.
p_2d = (3, 2)
k_2d = (2, 0)
# Domain. The length of the domain is chosen so that the normal 
# derivatives of the exact solution are zero at the boundary. This is 
# the only Neumann b.c. that we can specify at the moment.
const Lleft = 0.25
const Lright = 0.75
const Lbottom = 0.25
const Ltop = 0.75


function forcing_sine_2d(x::Float64, y::Float64)
    return 8.0 * pi^2 * sinpi(2.0 * x) * sinpi(2.0 * y)
end

function exact_sol_sine_2d(x::Float64, y::Float64)
    return sinpi(2.0 * x) * sinpi(2.0 * y)
end



# Tensor product b-spline case on a Cartesian geometry.
# Create Patch.
brk_x = collect(LinRange(Lleft, Lright, m_x+1))
brk_y = collect(LinRange(Lbottom, Ltop, m_y+1))
patch_x = Mantis.Mesh.Patch1D(brk_x)
patch_y = Mantis.Mesh.Patch1D(brk_y)
# Continuity vector for OPEN knot vector.
kvec_x = fill(k_2d[1], (m_x+1,))
kvec_x[1] = -1
kvec_x[end] = -1
kvec_y = fill(k_2d[2], (m_y+1,))
kvec_y[1] = -1
kvec_y[end] = -1
# Create function spaces (b-splines here).
trial_space_x = Mantis.FunctionSpaces.BSplineSpace(patch_x, p_2d[1], kvec_x)
test_space_x = Mantis.FunctionSpaces.BSplineSpace(patch_x, p_2d[1], kvec_x)
trial_space_y = Mantis.FunctionSpaces.BSplineSpace(patch_y, p_2d[2], kvec_y)
test_space_y = Mantis.FunctionSpaces.BSplineSpace(patch_y, p_2d[2], kvec_y)

trial_space_2d = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x, trial_space_y)
test_space_2d = Mantis.FunctionSpaces.TensorProductSpace(test_space_x, test_space_y)

# Create the geometry.
geom_cartesian = Mantis.Geometry.CartesianGeometry((brk_x, brk_y))



# Hierarchical refinement on the same mesh as above.
# Create the space
CB1 = Mantis.FunctionSpaces.BSplineSpace(patch_x, p_2d[1], [-1; fill(p_2d[1]-1, m_x-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch_y, p_2d[2], [-1; fill(p_2d[2]-1, m_y-1); -1])

nsub1 = 2
nsub2 = 2

TS1,FB1 = Mantis.FunctionSpaces.subdivide_bspline(CB1, nsub1)
TS2, FB2 = Mantis.FunctionSpaces.subdivide_bspline(CB2, nsub2)

CTP = Mantis.FunctionSpaces.TensorProductSpace(CB1, CB2)
FTP = Mantis.FunctionSpaces.TensorProductSpace(FB1, FB2)
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

CTS = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(TS1,TS2)

coarse_elements_to_refine = [3,4,5,8,9,10]
refined_elements = vcat(Mantis.FunctionSpaces.get_finer_elements.((CTS,), coarse_elements_to_refine)...)

refined_domains = Mantis.FunctionSpaces.HierarchicalActiveInfo([1:CTP_num_els;refined_elements], [0, CTP_num_els, CTP_num_els + length(refined_elements)])
hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, [CTS], refined_domains)

# Test if projection in space is exact
nxi_per_dim = 3
nxi = nxi_per_dim^2
xi_per_dim = collect(range(0,1, nxi_per_dim))
xi = Matrix{Float64}(undef, nxi,2)

xi_eval = (xi_per_dim, xi_per_dim)

for (idx,x) ∈ enumerate(Iterators.product(xi_per_dim, xi_per_dim))
    xi[idx,:] = [x[1] x[2]]
end

xs = Matrix{Float64}(undef, Mantis.FunctionSpaces.get_num_elements(hspace)*nxi,2)
nx = size(xs)[1]

A = zeros(nx, Mantis.FunctionSpaces.get_dim(hspace))

for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)
    level = Mantis.FunctionSpaces.get_active_level(hspace.active_elements, el)
    element_id = Mantis.FunctionSpaces.get_active_id(hspace.active_elements, el)

    max_ind_els = Mantis.FunctionSpaces._get_num_elements_per_space(hspace.spaces[level])
    ordered_index = Mantis.FunctionSpaces.linear_to_ordered_index(element_id, max_ind_els)

    borders_x = Mantis.Mesh.get_element(hspace.spaces[level].function_space_1.knot_vector.patch_1d, ordered_index[1])
    borders_y = Mantis.Mesh.get_element(hspace.spaces[level].function_space_2.knot_vector.patch_1d, ordered_index[2])

    x = [(borders_x[1] .+ xi[:,1] .* (borders_x[2] - borders_x[1])) (borders_y[1] .+ xi[:,2] .* (borders_y[2] - borders_y[1]))]

    idx = (el-1)*nxi+1:el*nxi
    xs[idx,:] = x

    eval_space = Mantis.FunctionSpaces.evaluate(hspace, el, xi_eval, 0)

    A[idx, eval_space[2]] = eval_space[1][0,0]
end

coeffs = A \ xs

hierarchical_geo = Mantis.Geometry.FEMGeometry(hspace, coeffs)





# Setup the quadrature rule.
q_nodes_x, q_weights_x = Mantis.Quadrature.gauss_legendre(p_2d[1]+1)
q_nodes_y, q_weights_y = Mantis.Quadrature.gauss_legendre(p_2d[2]+1)
# This function computes the tensor product of the quadrature weights in 
# the reference domain. This ensures that this only need to be computed 
# once and makes the size compatible with our other outputs (it returns 
# a vector of the weights in the right order).
q_weights_all = Mantis.Quadrature.tensor_product_weights((q_weights_x, q_weights_y))

for case in ["sine1d", "const1d", "sine2d", "sine2dH"]

    if case == "sine1d"
        fe_run(forcing_sine_1d, trial_space_1d, test_space_1d, geom_1d, 
               (q_nodes_1d, ), q_weights_1d_all, exact_sol_sine_1d, p_1d, 
               k_1d, case, n_1d, write_to_output_file, run_tests, verbose, bc_sine_1d)
    elseif case == "const1d"
        fe_run(forcing_const_1d, trial_space_1d, test_space_1d, geom_1d, 
                (q_nodes_1d, ), q_weights_1d_all, exact_sol_const_1d, p_1d, 
                k_1d, case, n_1d, write_to_output_file, run_tests, verbose, bc_const_1d)
    elseif case == "sine2d"
        fe_run(forcing_sine_2d, trial_space_2d, test_space_2d, geom_cartesian, 
               (q_nodes_x, q_nodes_y), q_weights_all, exact_sol_sine_2d, p_2d, 
               k_2d, case, n_2d, write_to_output_file, run_tests, verbose)
    elseif case == "sine2dH"
        fe_run(forcing_sine_2d, hspace, hspace, hierarchical_geo, 
               (q_nodes_x, q_nodes_y), q_weights_all, exact_sol_sine_2d, p_2d, 
               k_2d, case, n_2d, write_to_output_file, run_tests, verbose)
    else
        println("Warning: case '"*case*"' unknown. Skipping.") 
    end
end


