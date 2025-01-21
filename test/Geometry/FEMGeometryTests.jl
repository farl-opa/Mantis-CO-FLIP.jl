module FEMGeometryTests

import Mantis

import ReadVTK
using Printf
import LinearAlgebra

using Test

# Compute base directories for data input and output
reference_directory_tree = ["test", "data", "reference", "Geometry"]
output_directory_tree = ["test", "data", "output", "Geometry"]

# Test FEMGeometry (Polar) --------------------------------------------------
deg = 2
Wt = pi/2
b_θ = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
b_r = Mantis.FunctionSpaces.Bernstein(deg)

(P_sol, E_sol), (P_geom, E_geom, geom_coeffs_polar), _ = Mantis.FunctionSpaces.create_polar_spline_space_and_geometry((4, 1), (b_θ, b_r), (1, -1), 1.0)
geom = Mantis.Geometry.FEMGeometry(P_geom.component_spaces[1], geom_coeffs_polar)
field_coeffs = Matrix{Float64}(LinearAlgebra.I, Mantis.FunctionSpaces.get_num_basis(P_sol), Mantis.FunctionSpaces.get_num_basis(P_sol))
# polar_surface_field = Mantis.Fields.FEMField(P_sol.component_spaces[1], field_coeffs)

# Generate the plot
output_filename = "fem_geometry_polar_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
# Mantis.Plot.plot(geom, polar_surface_field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Test FEMGeometry (Annulus) --------------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Mesh.Patch1D(breakpoints)
B = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])
GB = Mantis.FunctionSpaces.GTBSplineSpace((B,), [1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0
               geom_coeffs_0.*r1]
geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
# Generate the plot
output_filename = "fem_geometry_annulus_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Test geometry
# Read the cell data from the reference file
reference_file = Mantis.Plot.export_path(reference_directory_tree, output_filename)
vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# Read the cell data from the output file
vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# # Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry - LagrangexBernstein (Square w/ hole) ----------------------
deg = 1
b = Mantis.FunctionSpaces.LobattoLegendre(deg)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Mesh.Patch1D(breakpoints)
B = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 0, 0, 0, -1])
GB = Mantis.FunctionSpaces.GTBSplineSpace((B,), [0])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
1.0   1.0
-1.0   1.0
-1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0
               geom_coeffs_0.*r1]
geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)
# Generate the plot
output_filename = "fem_geometry_lagrange_square_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
Mantis.Plot.plot(geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 1, ascii = false, compress = false)

# Test geometry
# Read the cell data from the reference file
reference_file = Mantis.Plot.export_path(reference_directory_tree, output_filename)
vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# Read the cell data from the output file
vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# # Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry (Spiral) ---------------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Mesh.Patch1D(breakpoints)
GB = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])

# control points for geometry
geom_coeffs =   [0.0 -1.0 0.0
1.0  -1.0 0.25
1.0   1.0 0.5
-1.0   1.0 0.75
-1.0  -1.0 1.0
0.0  -1.0 1.25]
spiral_geom = Mantis.Geometry.FEMGeometry(GB, geom_coeffs)

# Generate the plot
output_filename = "fem_geometry_spiral_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
Mantis.Plot.plot(spiral_geom; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Test geometry
# Read the cell data from the reference file
reference_file = Mantis.Plot.export_path(reference_directory_tree, output_filename)
vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# Read the cell data from the output file
vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# # Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry (wavy surface) ---------------------------------------------
deg = 2
Wt = pi/2
b = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
breakpoints = [0.0, 1.0, 2.0, 3.0, 4.0]
patch = Mantis.Mesh.Patch1D(breakpoints)
B = Mantis.FunctionSpaces.BSplineSpace(patch, b, [-1, 1, 1, 1, -1])
GB = Mantis.FunctionSpaces.GTBSplineSpace((B,), [1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
    1.0   1.0
    -1.0   1.0
    -1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0 -[+1.0, -1.0, +1.0, -1.0]
               geom_coeffs_0.*r1 [+1.0, -1.0, +1.0, -1.0]]
wavy_surface_geom = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)

# field on the wavy surface
field_coeffs = rand(Float64, 8, 1)
# wavy_surface_field = Mantis.Fields.FEMField(TP, field_coeffs)

# Generate the plot
output_filename = "fem_geometry_wavy_surface_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
# Mantis.Plot.plot(wavy_surface_geom, wavy_surface_field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)
Mantis.Plot.plot(wavy_surface_geom, vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Test geometry
# Read the cell data from the reference file
reference_file = Mantis.Plot.export_path(reference_directory_tree, output_filename)
vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# Read the cell data from the output file
vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# # Check if cell data is identical
@test reference_points ≈ output_points atol = 1e-14
@test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry (NURBS quarter annulus) ---------------------------------------------
deg = 2
b = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), deg, [-1, -1])
B = Mantis.FunctionSpaces.RationalFiniteElementSpace(b, [1, 1/sqrt(2), 1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((B, b1))
# control points for geometry
geom_coeffs_0 =   [0.0 1.0
    1.0   1.0
    1.0   0.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0 zeros(3)
               geom_coeffs_0.*r1 zeros(3)]
quarter_annulus = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)

# Generate the plot
output_filename = "fem_geometry_nurbs_quarter_annulus_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
Mantis.Plot.plot(quarter_annulus; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# # Test geometry
# # Read the cell data from the reference file
# reference_file = Mantis.Plot.export_path(reference_directory_tree, output_filename)
# vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
# reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
# reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# # Read the cell data from the output file
# vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
# output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
# output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# # # Check if cell data is identical
# @test reference_points ≈ output_points atol = 1e-14
# @test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry (NURBS annulus) ---------------------------------------------
deg = 2
b = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), deg, [-1, -1])
br = Mantis.FunctionSpaces.RationalFiniteElementSpace(b, [1, 1/sqrt(2), 1])
B = (br, br, br, br)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
    1.0   1.0
    -1.0   1.0
    -1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0 zeros(4)
               geom_coeffs_0.*r1 zeros(4)]
nurbs_annulus = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)

# Generate the plot
output_filename = "fem_geometry_nurbs_annulus_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
Mantis.Plot.plot(nurbs_annulus; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# # Test geometry
# # Read the cell data from the reference file
# reference_file = Mantis.Plot.export_path(reference_directory_tree, output_filename)
# vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
# reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
# reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# # Read the cell data from the output file
# vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
# output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
# output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# # # Check if cell data is identical
# @test reference_points ≈ output_points atol = 1e-14
# @test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry (NURBS wavy surface) ---------------------------------------------
deg = 2
b = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), deg, [-1, -1])
br = Mantis.FunctionSpaces.RationalFiniteElementSpace(b, [1, 1/sqrt(2), 1])
B = (br, br, br, br)
GB = Mantis.FunctionSpaces.GTBSplineSpace(B, [1, 1, 1, 1])
b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), 1, [-1, -1])
TP = Mantis.FunctionSpaces.TensorProductSpace((GB, b1))
# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
    1.0   1.0
    -1.0   1.0
    -1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0 -[+1.0, -1.0, +1.0, -1.0]
               geom_coeffs_0.*r1 [+1.0, -1.0, +1.0, -1.0]]
nurbs_wavy_surface = Mantis.Geometry.FEMGeometry(TP, geom_coeffs)

# Generate the plot
output_filename = "fem_geometry_nurbs_wavy_surface_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
Mantis.Plot.plot(nurbs_wavy_surface; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# # Test geometry
# # Read the cell data from the reference file
# reference_file = Mantis.Plot.export_path(reference_directory_tree, output_filename)
# vtk_reference = ReadVTK.VTKFile(ReadVTK.get_example_file(reference_file))
# reference_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Points")["Points"])
# reference_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_reference, "Cells")["connectivity"])

# # Read the cell data from the output file
# vtk_output = ReadVTK.VTKFile(ReadVTK.get_example_file(output_file))
# output_points = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Points")["Points"])
# output_cells = ReadVTK.get_data(ReadVTK.get_data_section(vtk_output, "Cells")["connectivity"])

# # # Check if cell data is identical
# @test reference_points ≈ output_points atol = 1e-14
# @test reference_cells == output_cells
# -----------------------------------------------------------------------------

# Test FEMGeometry (NURBS vs GTB basis) ---------------------------------------------
deg = 2
b = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), deg, [-1, -1])
br = Mantis.FunctionSpaces.RationalFiniteElementSpace(b, [1, 1/sqrt(2), 1])
Bsp = Mantis.FunctionSpaces.GTBSplineSpace((b, b, b, b), [1, 1, 1, 1])
Nurbs = Mantis.FunctionSpaces.GTBSplineSpace((br, br, br, br), [1, 1, 1, 1])

Wt = pi/2
gt = Mantis.FunctionSpaces.GeneralizedTrigonometric(deg, Wt)
B = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0, 2.0, 3.0, 4.0]), gt, [-1, 1, 1, 1, -1])
GTB = Mantis.FunctionSpaces.GTBSplineSpace((B,), [1])

b1 = Mantis.FunctionSpaces.BSplineSpace(Mantis.Mesh.Patch1D([0.0, 1.0]), 1, [-1, -1])

TP_bsp = Mantis.FunctionSpaces.TensorProductSpace((Bsp, b1))
TP_nurbs = Mantis.FunctionSpaces.TensorProductSpace((Nurbs, b1))
TP_gtb = Mantis.FunctionSpaces.TensorProductSpace((GTB, b1))

# control points for geometry
geom_coeffs_0 =   [1.0  -1.0
    1.0   1.0
    -1.0   1.0
    -1.0  -1.0]
r0 = 1
r1 = 2
geom_coeffs = [geom_coeffs_0.*r0 zeros(4)
               geom_coeffs_0.*r1 zeros(4)]
nurbs_annulus = Mantis.Geometry.FEMGeometry(TP_nurbs, geom_coeffs)
gtb_annulus = Mantis.Geometry.FEMGeometry(TP_gtb, geom_coeffs)

# field on the annulus
field_coeffs = Matrix{Float64}(LinearAlgebra.I, 8, 8)
# bsp_field = Mantis.Fields.FEMField(TP_bsp, field_coeffs)
# gtb_field = Mantis.Fields.FEMField(TP_gtb, field_coeffs)

# Generate the plot - NURBS + BSP
output_filename = "fem_geometry_nurbs_bsp_basis_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
# Mantis.Plot.plot(nurbs_annulus, bsp_field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

# Generate the plot - GTB
output_filename = "fem_geometry_gtb_basis_test.vtu"
output_file = Mantis.Plot.export_path(output_directory_tree, output_filename)
# Mantis.Plot.plot(gtb_annulus, gtb_field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

end
