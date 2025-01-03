module HierarchicalGeometryTests

import Mantis

import ReadVTK
using Printf
using Test
using LinearAlgebra

# Create the space

ne1 = 5; ne2 = 5
deg1 = 2; deg2 = 2

nsub1 = 2; nsub2 = 2

CTP = Mantis.FunctionSpaces.create_bspline_space(
      (0.0, 0.0), (1.0, 1.0), (ne1, ne2),
      (deg1, deg2), (deg1-1, deg2-1)
)
CTS, FTP = Mantis.FunctionSpaces.build_two_scale_operator(CTP, (nsub1, nsub2))
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

coarse_elements_to_refine = [3,4,5,8,9,10,13,14,15]
refined_elements = vcat(Mantis.FunctionSpaces.get_element_children.((CTS,), coarse_elements_to_refine)...)

hier_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, [CTS], [Int[], refined_elements], true)
hier_geo = Mantis.Geometry.compute_parametric_geometry(hier_space)

field_coeffs = Matrix{Float64}(LinearAlgebra.I,Mantis.FunctionSpaces.get_num_basis(hier_space), Mantis.FunctionSpaces.get_num_basis(hier_space))
tensor_field = Mantis.Fields.FEMField(hier_space, field_coeffs)

# Compute base directories for data output
output_directory_tree = ["test", "data", "output", "Geometry"]

output_filename = "fem_geometry_tensor_hbsplines.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
@test_nowarn Mantis.Plot.plot(hier_geo, tensor_field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

end
