

import Mantis

using Test
using LinearAlgebra

# Compute base directories for data input and output
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Poisson")

write_to_output_file = true









########################################################################
## Test cases for the 2D Poisson problem.                             ##
########################################################################

# This is how MANTIS is called to create a 2D problem.
function fe_run_2D(forcing_function_2d, trial_space, test_space, geom, q_nodes, 
                   q_weights, p_2d, case, output_to_file)
    # Function that gives the neumann condition on the boundary
    function neumann(x::Float64, which::String)
        if which == "left"
            return 2.0 * pi / geom.breakpoints[1][end] * cospi(2.0 * geom.breakpoints[1][1] / geom.breakpoints[1][end]) * sinpi(2.0 * x / geom.breakpoints[2][end])
        elseif which == "bottom"
            return 2.0 * pi / geom.breakpoints[2][end] * sinpi(2.0 * x / geom.breakpoints[1][end]) * cospi(2.0 * geom.breakpoints[2][1] / geom.breakpoints[2][end])
        elseif which == "right"
            return 2.0 * pi / geom.breakpoints[1][end] * cospi(2.0 * geom.breakpoints[1][end] / geom.breakpoints[1][end]) * sinpi(2.0 * x / geom.breakpoints[2][end])
        elseif which == "top"
            return 2.0 * pi / geom.breakpoints[2][end] * sinpi(2.0 * x / geom.breakpoints[1][end]) * cospi(2.0 * geom.breakpoints[2][end] / geom.breakpoints[2][end])
        else
            #return 0.0
            error("Boundary function not defined.")
        end
    end
    # Setup the element assembler.
    element_assembler = Mantis.Assemblers.PoissonBilinearForm(forcing_function_2d,
                                                              neumann,
                                                              trial_space,
                                                              test_space,
                                                              geom,
                                                              q_nodes,
                                                              q_weights)

    # Setup the global assembler.
    global_assembler = Mantis.Assemblers.Assembler()

    # Assemble.
    A, b = global_assembler(element_assembler)

    #@code_warntype element_assembler(1)

    A = vcat(A, ones((1,size(A)[2])))
    A = hcat(A, vcat(ones(size(A)[1]-1), 0.0))
    b = vcat(b, 0.0)

    # println("res")
    # display(A)
    # display(b)
    # println(LinearAlgebra.cond(Matrix(A)))

    # Solve & add bcs.
    sol = A \ Vector(b)

    # display(sol)

    # This is for the plotting. You can visualise the solution in 
    # Paraview, using the 'Plot over line'-filter.
    if output_to_file
        msave = Mantis.Geometry.get_num_elements(geom)
        output_filename = "Poisson-H-2D-p$p_2d-m$msave-case-"*case*".vtu"
        output_file = joinpath(output_data_folder, output_filename)
        field = Mantis.Fields.FEMField(trial_space, reshape(sol[1:end-1], :, 1))
        Mantis.Plot.plot(geom, field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = maximum([1, maximum(p_2d)]), ascii = false, compress = false)
    end

    # Compute error
    function exact_sol(x::Float64, y::Float64)
        return sinpi(2.0 * x) * sinpi(2.0 * y)
    end
    err_assembler = Mantis.Assemblers.AssemblerError(q_nodes, q_weights)
    err = err_assembler(trial_space, reshape(sol[1:end-1], :, 1), geom, exact_sol)
    println(err)

    err2 = err_assembler(trial_space, reshape(ones(length(sol)-1), :, 1), geom, (x,y) -> 0.0)
    println(err2)

    return sol
end




# # Returns a vector of the indices of the basis functions which have non-
# # zero support on the boundary, assuming open knot vector.
# function get_nz_boundary_indices(bspline::BSplineSpace)
#     return [1, get_dim(bspline)]
# end








# Create the space

ne1 = 20
ne2 = 20
breakpoints1 = collect(range(0.25,0.75,ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0.25,0.75,ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

deg1 = 2
deg2 = 2

CB1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])

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

    local eval = Mantis.FunctionSpaces.evaluate(hspace, el, xi_eval, 0)

    A[idx, eval[2]] = eval[1][0,0]
end

coeffs = A \ xs

hierarchical_geo = Mantis.Geometry.FEMGeometry(hspace, coeffs)














# Here we setup and specify the inputs to the FE problem.

# Number of elements.
# m_x = 10
# m_y = 10
# # polynomial degree and inter-element continuity.
p_2d = (deg1, deg2)
# k_2d = (2, 0)
# Domain. The length of the domain is chosen to be equal to the number 
# of elements to ensure elements of size one. This automatically removes 
# the implicit geometry assumed in the FunctionSpaces.
Lleft = 0.25
Lright = 0.75
Lbottom = 0.25
Ltop = 0.75


function forcing_sine_2d(x::Float64, y::Float64)
    return 8.0 * pi^2 * sinpi(2.0 * x) * sinpi(2.0 * y)
end

# # Create Patch.
# brk_x = collect(LinRange(Lleft, Lright, m_x+1))
# brk_y = collect(LinRange(Lbottom, Ltop, m_y+1))
# patch_x = Mantis.Mesh.Patch1D(brk_x)
# patch_y = Mantis.Mesh.Patch1D(brk_y)
# # Continuity vector for OPEN knot vector.
# kvec_x = fill(k_2d[1], (m_x+1,))
# kvec_x[1] = -1
# kvec_x[end] = -1
# kvec_y = fill(k_2d[2], (m_y+1,))
# kvec_y[1] = -1
# kvec_y[end] = -1
# Create function spaces (b-splines here).
# trial_space_x = Mantis.FunctionSpaces.BSplineSpace(patch_x, p_2d[1], kvec_x)
# test_space_x = Mantis.FunctionSpaces.BSplineSpace(patch_x, p_2d[1], kvec_x)
# trial_space_y = Mantis.FunctionSpaces.BSplineSpace(patch_y, p_2d[2], kvec_y)
# test_space_y = Mantis.FunctionSpaces.BSplineSpace(patch_y, p_2d[2], kvec_y)

# data = Dict{Int, Int}() # Why do we need this? (I made up the kv types)
trial_space_2d = hspace
test_space_2d = hspace

# # Create the geometry.
geom_2d = hierarchical_geo

# Setup the quadrature rule.
q_nodes_x, q_weights_x = Mantis.Quadrature.gauss_legendre(p_2d[1]+1)
q_nodes_y, q_weights_y = Mantis.Quadrature.gauss_legendre(p_2d[2]+1)
q_weights_all = Mantis.Quadrature.tensor_product_weights((q_weights_x, q_weights_y))

for case in ["sine"]

    if case == "sine"
        fe_run_2D(forcing_sine_2d, trial_space_2d, test_space_2d, geom_2d, 
                  (q_nodes_x, q_nodes_y), q_weights_all, p_2d, 
                   case, write_to_output_file)
    else
        error("Case: '",case,"' unknown.") 
    end
end


