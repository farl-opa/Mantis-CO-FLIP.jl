module HierarchicalGeometryTests

import Mantis

import ReadVTK
using Printf
using Test
using LinearAlgebra

# Create the space

ne1 = 5
ne2 = 5
deg1 = 2
deg2 = 2

nsub1 = 2
nsub2 = 2

CTP = Mantis.FunctionSpaces.create_bspline_space(
    (0.0, 0.0), (1.0, 1.0), (ne1, ne2), (deg1, deg2), (deg1 - 1, deg2 - 1)
)
CTS, FTP = Mantis.FunctionSpaces.build_two_scale_operator(CTP, (nsub1, nsub2))
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

coarse_elements_to_refine = [3, 4, 5, 8, 9, 10, 13, 14, 15]
refined_elements = vcat(
    Mantis.FunctionSpaces.get_element_children.((CTS,), coarse_elements_to_refine)...
)

hier_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(
    spaces, [CTS], [Int[], refined_elements], true
)
hier_spaced = Mantis.FunctionSpaces.DirectSumSpace((hier_space,))
@test_nowarn hier_geo = Mantis.Geometry.compute_parametric_geometry(hier_space)

end
