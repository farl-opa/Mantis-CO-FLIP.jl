

import Mantis

using Test
using LinearAlgebra




# This is how MANTIS is called to solve a problem. The bc input is only for the 1D case.
function fe_run(forcing_function, trial_space, test_space, geom, q_nodes, 
                q_weights, exact_sol, p, k, case, n, output_to_file, test=true, 
                verbose=false, bc = (false, 0.0, 0.0))
    if verbose
        println("Starting setup of problem and assembler for case "*case*" ...")
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
Mantis_folder = dirname(dirname(pathof(Mantis)))
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

if verbose
    println("Creating 2D Geometry and spaces ...")
end

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



# Same problem as above, but on the 'crazy' mesh
if verbose
    println("Creating crazy mesh ...")
end
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



# Hierarchical refinement on the same mesh as the first 2D case.
if verbose
    println("Creating 2D Hierarchical geometry and spaces ...")
end
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



########################################################################
## Test cases for the 3D Poisson problem.                             ##
########################################################################

if verbose
    println("Creating 3D Geometry and spaces ...")
end

# Dimension
n_3d = 3
# Number of elements.
m_3d_x = 5
m_3d_y = 5
m_3d_z = 5
# polynomial degree and inter-element continuity.
p_3d = (3, 4, 1)
k_3d = (2, 2, 0)
# Domain. The length of the domain is chosen so that the normal 
# derivatives of the exact solution are zero at the boundary. This is 
# the only Neumann b.c. that we can specify at the moment.
const Lx1 = 0.25
const Lx2 = 0.75
const Ly1 = 0.25
const Ly2 = 0.75
const Lz1 = 0.25
const Lz2 = 0.75


function forcing_sine_3d(x::Float64, y::Float64, z::Float64)
    return 12.0 * pi^2 * sinpi(2.0 * x) * sinpi(2.0 * y) * sinpi(2.0 * z)
end

function exact_sol_sine_3d(x::Float64, y::Float64, z::Float64)
    return sinpi(2.0 * x) * sinpi(2.0 * y) * sinpi(2.0 * z)
end



# Tensor product b-spline case on a Cartesian geometry.
# Create Patch.
brk_3d_x = collect(LinRange(Lx1, Lx2, m_3d_x+1))
brk_3d_y = collect(LinRange(Ly1, Ly2, m_3d_y+1))
brk_3d_z = collect(LinRange(Lz1, Lz2, m_3d_z+1))
patch_3d_x = Mantis.Mesh.Patch1D(brk_3d_x)
patch_3d_y = Mantis.Mesh.Patch1D(brk_3d_y)
patch_3d_z = Mantis.Mesh.Patch1D(brk_3d_z)
# Continuity vector for OPEN knot vector.
kvec_3d_x = fill(k_3d[1], (m_3d_x+1,))
kvec_3d_x[1] = -1
kvec_3d_x[end] = -1
kvec_3d_y = fill(k_3d[2], (m_3d_y+1,))
kvec_3d_y[1] = -1
kvec_3d_y[end] = -1
kvec_3d_z = fill(k_3d[3], (m_3d_z+1,))
kvec_3d_z[1] = -1
kvec_3d_z[end] = -1
# Create function spaces (b-splines here).
trial_space_3d_x = Mantis.FunctionSpaces.BSplineSpace(patch_3d_x, p_3d[1], kvec_3d_x)
test_space_3d_x = Mantis.FunctionSpaces.BSplineSpace(patch_3d_x, p_3d[1], kvec_3d_x)
trial_space_3d_y = Mantis.FunctionSpaces.BSplineSpace(patch_3d_y, p_3d[2], kvec_3d_y)
test_space_3d_y = Mantis.FunctionSpaces.BSplineSpace(patch_3d_y, p_3d[2], kvec_3d_y)
trial_space_3d_z = Mantis.FunctionSpaces.BSplineSpace(patch_3d_z, p_3d[3], kvec_3d_z)
test_space_3d_z = Mantis.FunctionSpaces.BSplineSpace(patch_3d_z, p_3d[3], kvec_3d_z)

trial_space_3d_xy = Mantis.FunctionSpaces.TensorProductSpace(trial_space_3d_x, trial_space_3d_y)
test_space_3d_xy = Mantis.FunctionSpaces.TensorProductSpace(test_space_3d_x, test_space_3d_y)

trial_space_3d = Mantis.FunctionSpaces.TensorProductSpace(trial_space_3d_xy, trial_space_3d_z)
test_space_3d = Mantis.FunctionSpaces.TensorProductSpace(test_space_3d_xy, test_space_3d_z)

# Create the geometry.
geom_3d_cartesian = Mantis.Geometry.CartesianGeometry((brk_3d_x, brk_3d_y, brk_3d_z))

# Setup the quadrature rule.
q_nodes_3d_x, q_weights_3d_x = Mantis.Quadrature.gauss_legendre(p_3d[1]+1)
q_nodes_3d_y, q_weights_3d_y = Mantis.Quadrature.gauss_legendre(p_3d[2]+1)
q_nodes_3d_z, q_weights_3d_z = Mantis.Quadrature.gauss_legendre(p_3d[3]+1)
# This function computes the tensor product of the quadrature weights in 
# the reference domain. This ensures that this only need to be computed 
# once and makes the size compatible with our other outputs (it returns 
# a vector of the weights in the right order).
q_weights_3d_all = Mantis.Quadrature.tensor_product_weights((q_weights_3d_x, q_weights_3d_y, q_weights_3d_z))




# Running all testcases.
for case in ["sine1d", "const1d", "sine2d", "sine2d-crazy", "sine2dH", "sine3d"]

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
    elseif case == "sine2d-crazy"
        fe_run(forcing_sine_2d, trial_space_2d, test_space_2d, geom_crazy, 
                (q_nodes_x, q_nodes_y), q_weights_all, exact_sol_sine_2d, p_2d, 
                k_2d, case, n_2d, write_to_output_file, run_tests, verbose)
    elseif case == "sine2dH"
        fe_run(forcing_sine_2d, hspace, hspace, hierarchical_geo, 
               (q_nodes_x, q_nodes_y), q_weights_all, exact_sol_sine_2d, p_2d, 
               k_2d, case, n_2d, write_to_output_file, run_tests, verbose)
    elseif case == "sine3d"
        fe_run(forcing_sine_3d, trial_space_3d, test_space_3d, geom_3d_cartesian, 
                (q_nodes_3d_x, q_nodes_3d_y, q_nodes_3d_z), q_weights_3d_all, 
                exact_sol_sine_3d, p_3d, k_3d, case, n_3d, write_to_output_file, 
                run_tests, verbose)
    else
        println("Warning: case '"*case*"' unknown. Skipping.") 
    end
end


