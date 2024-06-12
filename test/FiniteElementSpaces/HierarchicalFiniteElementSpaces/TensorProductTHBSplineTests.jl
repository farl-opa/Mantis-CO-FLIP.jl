import Mantis

using Test
using Random

# Tests for a tensor product HierarchicalSplineSpace
ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0,1,ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0,1,ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
deg1 = 2
deg2 = 2
nsubs = (2, 2) 
nlevels = 3 # should be 2 or higher, otherwise the hierarchical space is not actually hierarchical
Random.seed!(9)

CB1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])
CTP = Mantis.FunctionSpaces.TensorProductSpace(CB1, CB2)


TTS, FTP = Mantis.FunctionSpaces.subdivide_bspline(CTP, nsubs)

spaces = [CTP, FTP]
operators = [TTS]

for level ∈ 3:nlevels
    new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(spaces[level-1], nsubs)
    push!(spaces, new_space)
    push!(operators, new_operator)
end

marked_elements_per_level = [Int[], Mantis.FunctionSpaces.get_finer_elements(operators[1], [7,8,9,12,13,14,17,18,19]), Mantis.FunctionSpaces.get_finer_elements(operators[2], [23, 24, 25, 33, 34, 35, 43, 44, 45])] 
hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_elements_per_level, true)

x1, _ = Mantis.Quadrature.gauss_legendre(deg1+1)
x2, _ = Mantis.Quadrature.gauss_legendre(deg2+1)
xi = (x1, x2)

# Tests for coefficients and evaluation
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(hspace, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = Mantis.FunctionSpaces.evaluate(hspace, el, xi, 0)
    # Positivity of the basis
    @test minimum(h_eval[0,0]) >= 0.0
    # Partition of unity
    @test all(isapprox.(sum(h_eval[0,0], dims=2), 1.0, atol=1e-14))
end

# Geometry visualization
#=
function get_geometry(hspace::Mantis.FunctionSpaces.HierarchicalFiniteElementSpace{n}) where {n}
    degrees = Mantis.FunctionSpaces.get_polynomial_degree_per_dim(hspace.spaces[1])
    nxi_per_dim = maximum(degrees) + 1
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

    return hierarchical_geo
end

hspace_geo = get_geometry(hspace)

# Generate the Plot
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Field")

output_filename = "THB-partition-unity-test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(hspace_geo; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)
=#