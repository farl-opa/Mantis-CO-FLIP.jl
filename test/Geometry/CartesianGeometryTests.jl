module CartesianGeometryTests

import Mantis

import ReadVTK
using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")

# Constructor, property, and getters and setters tests -------------------------------------
# Vector{Float64} input.
geometry = Mantis.Geometry.CartesianGeometry((
    [0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [-0.75, 0.0, 0.25, 0.75]
))
@test Mantis.Geometry.get_breakpoints(geometry) ==
    ([0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [-0.75, 0.0, 0.25, 0.75])
@test Mantis.Geometry._get_num_elements_per_dim(geometry) == (2, 2, 3)

@test Mantis.Geometry.get_num_elements(geometry) == 12
@test Mantis.Geometry.get_manifold_dim(geometry) == 3
@test Mantis.Geometry.get_image_dim(geometry) == 3

# LinRange input.
geometry = Mantis.Geometry.CartesianGeometry((
    LinRange(0.5, 2.5, 5), LinRange(-0.75, 0.75, 3)
))
@test Mantis.Geometry.get_breakpoints(geometry) ==
    ([0.5, 1.0, 1.5, 2.0, 2.5], [-0.75, 0.0, 0.75])
@test Mantis.Geometry._get_num_elements_per_dim(geometry) == (4, 2)

@test Mantis.Geometry.get_num_elements(geometry) == 8
@test Mantis.Geometry.get_manifold_dim(geometry) == 2
@test Mantis.Geometry.get_image_dim(geometry) == 2

# Test 2D CartesianGeometry ----------------------------------------------------------------
for nx in 1:3
    for ny in 1:3
        geometry = Mantis.Geometry.create_cartesian_box((0.0, 0.0), (1.0, 2.0), (nx, ny))

        # Set file name and path
        file_name = "cartesian_test_nx_$(nx)_ny_$(ny).vtu"
        output_file_path = Mantis.GeneralHelpers.export_path(output_directory_tree, file_name)
        # Generate the vtk file
        Mantis.Plot.plot(
            geometry;
            vtk_filename=output_file_path[1:(end - 4)],  # Remove the file extension.
            n_subcells=1,
            degree=1,
            ascii=false,
            compress=false,
        )

        # Read the cell data from the reference file.
        reference_points, reference_cells = get_point_cell_data(
            reference_directory_tree, file_name
        )
        # Read the cell data from the output file.
        output_points, output_cells = get_point_cell_data(output_file_path)

        # Check if cell data is point-wise identical.
        @test all(isapprox.(reference_points, output_points; rtol=rtol))
        @test all(isequal.(reference_cells, output_cells))
    end
end

end
