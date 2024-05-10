import Mantis

using Test

# Tests for a tensor product HierarchicalSplineSpace

ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0,1,ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0,1,ne2+1))
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
hspace = Mantis.FunctionSpaces.get_hierarchical_space(spaces, [CTS], refined_domains)

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
end


#=
# Generate the Plot
#hspace = CTP
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Geometry")

line_1_geo = Mantis.Geometry.CartesianGeometry((breakpoints1,))
line_2_geo = Mantis.Geometry.CartesianGeometry((breakpoints2,))

tensor_prod_geo = Mantis.Geometry.TensorProductGeometry(line_1_geo, line_2_geo)

field_coeffs = rand(Mantis.FunctionSpaces.get_dim(hspace), 1)
tensor_field = Mantis.Fields.FEMField(hspace, field_coeffs)

output_filename = "fem_geometry_tensor_hbsplines.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(tensor_prod_geo, tensor_field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)
=#