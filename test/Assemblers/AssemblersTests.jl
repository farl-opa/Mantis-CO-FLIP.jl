

import Mantis

using Test
using LinearAlgebra

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Poisson")

write_to_output_file = false


# This is how MANTIS is called to create a 1D mixed problem.
function fe_run_1D(forcing_function, bilinear_function, linear_function, trial_space, 
                   test_space, geom, q_nodes, q_weights, bc_left, bc_right, p, 
                   k, case, output_to_file=false)
    # Setup the element assembler.
    element_assembler = Mantis.Assemblers.PoissonBilinearForm1D(forcing_function,
                                                                bilinear_function,
                                                                linear_function,
                                                                trial_space,
                                                                test_space,
                                                                geom,
                                                                q_nodes,
                                                                q_weights)

    # Setup the global assembler.
    global_assembler = Mantis.Assemblers.Assembler(bc_left, bc_right)

    # Assemble.
    A, b = global_assembler(element_assembler)

    # Solve & add bcs.
    sol = [bc_left, (A \ Vector(b))..., bc_right]

    @test issymmetric(A)
    @test isempty(nullspace(Matrix(A)))

    if case == "const" && p >= 2
        # The exact solution is in the space, so we can test for 
        # equality to the exact solution.
        xi = collect(LinRange(0.0, 1.0, 10))
        exact_sol = x -> x.*(x.-m)
        for element_id in 1:m
            x = Mantis.Geometry.evaluate(geom, element_id, (xi,))
            basis, active_bases = Mantis.FunctionSpaces.evaluate(trial_space, element_id, (xi,), 0)
            exact = exact_sol(x)
            @test all(isapprox.(exact .- sol[active_bases]' * basis[0]', 0.0, atol=1e-12))
        end
    end


    # This is for the plotting. If you want to output the data into vtu-
    # files, uncomment these lines. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.
    if output_to_file
        msave = Mantis.Geometry.get_num_elements(geom)
        output_filename = "Poisson-1D-p$p-k$k-m$msave-case-"*case*".vtu"
        output_file = joinpath(output_data_folder, output_filename)
        field = Mantis.Fields.FEMField(trial_space, reshape(sol, :, 1))
        Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = p, ascii = false, compress = false)
    end
end

# Here we setup and specify the inputs to the FE problem.
function forcing_sine(x)
    return pi^2 * sinpi(x)
end

function forcing_const(x)
    return -2.0
end

function bilinear_function(Ndi, Ndj)
    return Ndi * Ndj
end

function linear_function(Ni, f)
    return Ni * f
end


# Number of elements.
m = 10
# polynomial degree and inter-element continuity.
p = 5
k = 3
# Domain.
Lleft = 0.0
Lright = float(m)

# Create Patch.
brk = collect(LinRange(Lleft, Lright, m+1))
patch = Mantis.Mesh.Patch1D(brk)
# Continuity vector for OPEN knot vector.
kvec = fill(k, (m+1,))
kvec[1] = -1
kvec[end] = -1
# Create function spaces (b-splines here).
trial_space = Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)
test_space = Mantis.FunctionSpaces.BSplineSpace(patch, p, kvec)

# Create the geometry.
geom = Mantis.Geometry.CartesianGeometry((brk,))

# Setup the quadrature rule.
q_nodes, q_weights = Mantis.Quadrature.gauss_legendre(p+1)

# Choose case ("sine" or "const")
for case in ["sine", "const"]
    if case == "sine"
        # Boundary conditions (Dirichlet).
        bc_left = 0.0
        bc_right = 1.0
        fe_run_1D(forcing_sine, bilinear_function, linear_function, trial_space, 
                  test_space, geom, q_nodes, q_weights, bc_left, bc_right, p, k, case, write_to_output_file)
    elseif case == "const"
        # Boundary conditions (Dirichlet).
        bc_left = 0.0
        bc_right = 0.0
        fe_run_1D(forcing_const, bilinear_function, linear_function, trial_space, 
                  test_space, geom, q_nodes, q_weights, bc_left, bc_right, p, k, case, write_to_output_file)
    else
        error("Case: '",case,"' unknown.")
    end
end





########################################################################
## Test cases for the mixed 1D Poisson problem.                       ##
########################################################################

# This is how MANTIS is called to create a 1D mixed problem.
function fe_run_mixed(forcing_function_mixed, Fbilinear1, Fbilinear2, 
                      Fbilinear3, Flinear, trial_space_phi, test_space_phi, 
                      trial_space_sigma, test_space_sigma, geom, q_nodes, 
                      q_weights, p_sigma, k_sigma, p_phi, k_phi, case, 
                      output_to_file=false)
    # Setup the element assembler.
    element_assembler = Mantis.Assemblers.PoissonBilinearFormMixed1D(forcing_function_mixed,
                                                                     Fbilinear1,
                                                                     Fbilinear2,
                                                                     Fbilinear3,
                                                                     Flinear,
                                                                     trial_space_phi,
                                                                     test_space_phi,
                                                                     trial_space_sigma,
                                                                     test_space_sigma,
                                                                     geom,
                                                                     q_nodes,
                                                                     q_weights)

    # Setup the global assembler.
    global_assembler = Mantis.Assemblers.Assembler()

    # Assemble.
    A, b = global_assembler(element_assembler)

    # Solve & add bcs.
    sol = A \ Vector(b)

    # This is for the plotting. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.
    if output_to_file
        ndofs_sigma = Mantis.FunctionSpaces.get_dim(trial_space_sigma)

        msave = Mantis.Geometry.get_num_elements(geom)

        output_filename = "Poisson-1D-Mixed-phi-p$p_sigma-k$k_sigma-m$msave-case-"*case*".vtu"
        output_file = joinpath(output_data_folder, output_filename)
        field = Mantis.Fields.FEMField(trial_space_phi, reshape(sol[ndofs_sigma+1:end], :, 1))
        Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = maximum([1, p_phi]), ascii = false, compress = false)

        output_filename = "Poisson-1D-Mixed-sigma-p$p_phi-k$k_phi-m$msave-case-"*case*".vtu"
        output_file = joinpath(output_data_folder, output_filename)
        field = Mantis.Fields.FEMField(trial_space_sigma, reshape(sol[begin:ndofs_sigma], :, 1))
        Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = p_sigma, ascii = false, compress = false)
    end

    return sol
end



# Here we setup and specify the inputs to the FE problem.
function function_der_function(Ndi, Nj)
    return Ndi * Nj
end

function function_function(Ni, f)
    return Ni * f
end


# Number of elements.
m = 14
# polynomial degree and inter-element continuity.
p_phi = 0
k_phi = -1
p_sigma = 1
k_sigma = 0
# Domain.
Lleft = 0.0
Lright = float(m)


function forcing_sine_mixed(x)
    return (pi/Lright)^2 * sin(pi * x / Lright)
end

function forcing_const_mixed(x)
    return -2.0
end

# Create Patch.
brk = collect(LinRange(Lleft, Lright, m+1))
patch = Mantis.Mesh.Patch1D(brk)
# Continuity vector for OPEN knot vector.
kvec_phi = fill(k_phi, (m+1,))
kvec_phi[1] = -1
kvec_phi[end] = -1
kvec_sigma = fill(k_sigma, (m+1,))
kvec_sigma[1] = -1
kvec_sigma[end] = -1
# Create function spaces (b-splines here).
trial_space_phi = Mantis.FunctionSpaces.BSplineSpace(patch, p_phi, kvec_phi)
test_space_phi = Mantis.FunctionSpaces.BSplineSpace(patch, p_phi, kvec_phi)
trial_space_sigma = Mantis.FunctionSpaces.BSplineSpace(patch, p_sigma, kvec_sigma)
test_space_sigma = Mantis.FunctionSpaces.BSplineSpace(patch, p_sigma, kvec_sigma)

# Create the geometry.
geom = Mantis.Geometry.CartesianGeometry((brk,))

# Setup the quadrature rule.
q_nodes, q_weights = Mantis.Quadrature.gauss_legendre(maximum([p_phi, p_sigma])+1)

# Choose case ("sine" or "const")
for case in ["sine", "const"]
    
    if case == "sine"
        fe_run_mixed(forcing_sine_mixed, function_function, function_der_function, 
                     function_der_function, function_function, trial_space_phi, 
                     test_space_phi, trial_space_sigma, test_space_sigma, geom, 
                     q_nodes, q_weights, p_sigma, k_sigma, p_phi, k_phi, case, write_to_output_file)
    elseif case == "const"
        fe_run_mixed(forcing_const_mixed, function_function, function_der_function, 
                     function_der_function, function_function, trial_space_phi, 
                     test_space_phi, trial_space_sigma, test_space_sigma, geom, 
                     q_nodes, q_weights, p_sigma, k_sigma, p_phi, k_phi, case, write_to_output_file)
    else
        error("Case: '",case,"' unknown.")
    end
end









########################################################################
## Test cases for the 2D Poisson problem.                             ##
########################################################################

# This is how MANTIS is called to create a 2D problem.
function fe_run_2D(forcing_function_2d, trial_space, test_space, geom, q_nodes, 
                   q_weights, p_2d, k_2d, case, output_to_file=false)
    # Setup the element assembler.
    element_assembler = Mantis.Assemblers.PoissonBilinearForm(forcing_function_2d,
                                                              trial_space,
                                                              test_space,
                                                              geom,
                                                              q_nodes,
                                                              q_weights)

    # Setup the global assembler.
    global_assembler = Mantis.Assemblers.Assembler()

    # Assemble.
    A, b = global_assembler(element_assembler)

    # Solve & add bcs.
    sol = A \ Vector(b)

    # This is for the plotting. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.
    if output_to_file
        ndofs_sigma = Mantis.FunctionSpaces.get_dim(trial_space_sigma)

        msave = Mantis.Geometry.get_num_elements(geom)
        output_filename = "Poisson-2D-phi-p$p_2d-k$k_2d-m$msave-case-"*case*".vtu"
        output_file = joinpath(output_data_folder, output_filename)
        field = Mantis.Fields.FEMField(trial_space_phi, reshape(sol[ndofs_sigma+1:end], :, 1))
        Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = maximum([1, p_phi]), ascii = false, compress = false)
    end

    return sol
end



# Here we setup and specify the inputs to the FE problem.

# Number of elements.
m_x = 14
m_y = 14
# polynomial degree and inter-element continuity.
p_2d = (1, 2)
k_2d = (-1, 1)
# Domain.
Lleft = 0.0
Lright = float(m_x)
Lbottom = 0.0
Ltop = float(m_y)


function forcing_sine_2d(x)
    return 8.0 * pi^2 * sinpi(2.0 * x / Lbottom) * sinpi(2.0 * y / Ltop)
end

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

data = Dict{Int, Int}() # Why do we need this? (I made up the kv types)
trial_space_2d = Mantis.FunctionSpaces.TensorProductSpace(trial_space_x, trial_space_y, data)
test_space_2d = Mantis.FunctionSpaces.TensorProductSpace(test_space_x, test_space_y, data)

# Create the geometry.
geom = Mantis.Geometry.CartesianGeometry((brk_x, brk_y))

# Setup the quadrature rule.
q_nodes_x, q_weights_x = Mantis.Quadrature.gauss_legendre(p_2d[1]+1)
q_nodes_y, q_weights_y = Mantis.Quadrature.gauss_legendre(p_2d[2]+1)

# # Choose case ("sine")
for case in ["sine"]

    if case == "sine"
        fe_run_2D(forcing_sine_2d, trial_space_2d, test_space_2d, geom, 
                  (q_nodes_x, q_nodes_y), (q_weights_x, q_weights_y), p_2d, 
                  k_2d, case, write_to_output_file)
    else
        error("Case: '",case,"' unknown.") 
    end
end


