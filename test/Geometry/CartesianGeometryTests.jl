module CartesianGeometryTests

import Mantis

import ReadVTK
using Test

# Refer to the following file for method and variable definitions
include("GeometryTestsHelpers.jl")

# Test 2D CartesianGeometry ---------------------------------------------------
for nx = 1:3
    for ny = 1:3
        geometry = Mantis.Geometry.create_cartesian_box((0.0, 0.0), (1.0, 2.0), (nx, ny))
        
        # Set file name and path
        file_name = "cartesian_test_nx_$(nx)_ny_$(ny).vtu"
        output_file_path = Mantis.Plot.export_path(output_directory_tree, file_name)
        # Generate the vtk file 
        Mantis.Plot.plot(
            geometry;
            vtk_filename = output_file_path[1:end-4], #remove the file extension
            n_subcells = 1,
            degree = 1,
            ascii = false, 
            compress = false
        )

        # Read the cell data from the reference file
        reference_points, reference_cells = get_point_cell_data(
            reference_directory_tree, file_name
        )
        # Read the cell data from the output file
        output_points, output_cells = get_point_cell_data(output_file_path)
        # Check if cell data is identical
        @test all(isapprox.(reference_points, output_points; rtol=rtol))
        @test all(isequal.(reference_cells, output_cells))
    end
end
# -----------------------------------------------------------------------------

end
