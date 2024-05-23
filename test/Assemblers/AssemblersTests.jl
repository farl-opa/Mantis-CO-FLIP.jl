

import Mantis

using Test
using LinearAlgebra

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
        forcing_function = forcing_sine
    elseif case == "const"
        # Boundary conditions (Dirichlet).
        bc_left = 0.0
        bc_right = 0.0
        forcing_function = forcing_const
    else
        error("Case: '",case,"' unknown.")
    end

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

    # # Compute base directories for data input and output
    # Mantis_folder =  dirname(dirname(pathof(Mantis)))
    # data_folder = joinpath(Mantis_folder, "test", "data")
    # output_data_folder = joinpath(data_folder, "output", "Poisson")

    # output_filename = "Poisson-1D-p$p-k$k-m$m-case-"*case*".vtu"
    # output_file = joinpath(output_data_folder, output_filename)
    # field = Mantis.Fields.FEMField(trial_space, reshape(sol, :, 1))
    # Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = p, ascii = false, compress = false)
end




########################################################################
## Test cases for the mixed 1D Poisson problem.                       ##
########################################################################
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
        forcing_function_mixed = forcing_sine_mixed
    elseif case == "const"
        forcing_function_mixed = forcing_const_mixed
    else
        error("Case: '",case,"' unknown.")
    end

    # Setup the element assembler.
    element_assembler = Mantis.Assemblers.PoissonBilinearFormMixed1D(forcing_function_mixed,
                                                                     function_function,
                                                                     function_der_function,
                                                                     function_der_function,
                                                                     function_function,
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

    # if case == "const" && p >= 2
    #     # The exact solution is in the space, so we can test for 
    #     # equality to the exact solution.
    #     xi = collect(LinRange(0.0, 1.0, 10))
    #     exact_sol = x -> x.*(x.-m)
    #     for element_id in 1:m
    #         x = Mantis.Geometry.evaluate(geom, element_id, (xi,))
    #         basis, active_bases = Mantis.FunctionSpaces.evaluate(trial_space, element_id, (xi,), 0)
    #         exact = exact_sol(x)
    #         @test all(isapprox.(exact .- sol[active_bases]' * basis[0]', 0.0, atol=1e-12))
    #     end
    # end


    # This is for the plotting. If you want to output the data into vtu-
    # files, uncomment these lines. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.

    # # Compute base directories for data input and output
    # Mantis_folder =  dirname(dirname(pathof(Mantis)))
    # data_folder = joinpath(Mantis_folder, "test", "data")
    # output_data_folder = joinpath(data_folder, "output", "Poisson")

    # ndofs_sigma = Mantis.FunctionSpaces.get_dim(trial_space_sigma)

    # output_filename = "Mixed-Poisson-1D-phi-p$p_phi-k$k_phi-m$m-case-"*case*".vtu"
    # output_file = joinpath(output_data_folder, output_filename)
    # field = Mantis.Fields.FEMField(trial_space_phi, reshape(sol[ndofs_sigma+1:end], :, 1))
    # Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = maximum([1, p_phi]), ascii = false, compress = false)

    # output_filename = "Mixed-Poisson-1D-sigma-p$p_sigma-k$k_sigma-m$m-case-"*case*".vtu"
    # output_file = joinpath(output_data_folder, output_filename)
    # field = Mantis.Fields.FEMField(trial_space_sigma, reshape(sol[begin:ndofs_sigma], :, 1))
    # Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = p_sigma, ascii = false, compress = false)
end


