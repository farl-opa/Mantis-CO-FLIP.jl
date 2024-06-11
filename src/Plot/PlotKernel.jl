


function _plot(geometry::Geometry.AbstractGeometry{1, range_dim}, field::Union{Nothing, F} = nothing; vtk_filename::String = "default", n_subcells::Int64 = 1, degree::Int64 = 1, ascii = false, compress = true) where {range_dim, F <: Fields.AbstractField{1, field_dim} where {field_dim}}
    # This function generates points per plotted 1D cell, so connectivity is lost, this is what requires less information
    # from the mesh. Each computational element is sampled at n_subsamples (minimum is 2 per direction). These subsamples 
    # create a structured grid, each cell of this refined grid is plotted.
    # These cells can be on a straight line, but they can also lie on a complex manifold of dimension 1 embedded in R^3
    
    domain_dim = 1  # this is hardcoded, because each type of cell plotted in VTK is different, so
                    # we need specialized functions

    # print("Geometry dimensions: ", domain_dim, " x ", range_dim, "\n")
    
    # Compute the total number of points
    n_vertices = prod(geometry.n_elements) * n_subcells * (degree+1)  # there are n_subcells cells per element, and (p+1) vertices per cell
    n_total_elements = prod(geometry.n_elements)
    n_total_cells = n_total_elements * n_subcells  # each computational element is subdivided into n_subcells
    
    vertices = Array{Float64, 2}(undef, range_dim, n_vertices)
    cells = Vector{WriteVTK.MeshCell}(undef, n_total_cells)
    if isnothing(field)
        field_dim = 0
    else
        field_dim = Fields.get_image_dim(field)
    end
    point_data = zeros(field_dim,n_vertices)
    
    vertex_idx = 1
    vertex_offset = 0
    dξ = 1.0/n_subcells
    dξ_sub = dξ/degree  # a polynomial degree approximatio of p created p subdivisions of a subcell

    # # this is where points evaluated inside an element will be stored
    # vertices_el = zeros(n_dim, n_eval_1, n_eval_2, n_eval_3, ...)
    # el_vertex_offset += prod(n_eval)
    # vertices_el[:,i,j,k] .= vector_output_of_evaluate

    for element_idx in 1:n_total_elements
        # vertices_el = geometry.evaluate(..., NTuple_of_directional_points)
        # then, loop to store things in vtk format

        for subcell_idx in 1:n_subcells
            # Left vertex of subcell
            ξ_0 = dξ * (subcell_idx - 1)

            # Boundary vertices
            for bnd_idx = 0:1
                ξ = ξ_0 + bnd_idx * dξ
                vertices[:, vertex_idx + bnd_idx] .= vec(Geometry.evaluate(geometry, element_idx, ([ξ],)))
                # Compute data to plot 
                if !isnothing(field)
                    point_data[:,vertex_idx + bnd_idx] = vec(Fields.evaluate(field, element_idx, ([ξ],)))
                end
            end
            
            # Add interior vertices 
            for interior_vertex_idx in 1:(degree-1)
                ξ = ξ_0 + dξ_sub * interior_vertex_idx
                vertices[:, vertex_idx + interior_vertex_idx + 1] .= vec(Geometry.evaluate(geometry, element_idx, ([ξ],)))
                # Compute data to plot 
                if !isnothing(field)
                    point_data[:,vertex_idx + interior_vertex_idx + 1] = vec(Fields.evaluate(field, element_idx, ([ξ],)))
                end
            end

            # Add cell
            cell_idx = (element_idx - 1) * n_subcells + subcell_idx
            cells[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_CURVE, collect(vertex_idx:(vertex_idx + degree)))  

            # Update vertex_idx to start again
            vertex_idx += degree + 1
        end
    end

    WriteVTK.vtk_grid(vtk_filename, vertices, cells; append = false, ascii = ascii, compress = compress, vtkversion = :latest) do vtk 
        vtk.version == "2.2"
        if !isnothing(field)
            vtk["point_data", WriteVTK.VTKPointData()] = point_data
        end
    end
end



function _plot(geometry::Geometry.AbstractGeometry{2, range_dim}, field::Union{Nothing, F} = nothing; vtk_filename::String = "default", n_subcells::Int64 = 1, degree::Int64 = 1, ascii = false, compress = true) where {range_dim, F <: Fields.AbstractField{2, field_dim} where {field_dim}}
    # This function generates points per plotted 2D cell, so connectivity is lost, this is what requires less information
    # from the mesh. Each computational element is sampled at n_subsamples (minimum is 2 per direction). These subsamples 
    # create a structured grid, each cell of this refined grid is plotted.
    # These cells can be on a plane, but then can also lie on a complex manifold of dimension 2 embedded in R^3
    
    domain_dim = 2  # this is hardcoded, because each type of cell plotted in VTK is different, so
                    # we need specialized functions

    # print("Geometry dimensions: ", domain_dim, " x ", range_dim, "\n")
    
    # Compute the total number of points
    n_vertices = prod(geometry.n_elements) * (n_subcells^2) * ((degree+1)^2)  # there are n_subcells^2 cells per element, and (p+1)^2 vertices per cell
    n_total_elements = prod(geometry.n_elements)
    n_total_cells = n_total_elements * (n_subcells^2)  # each computational element is subdivided into n_subcells^2
    
    vertices = Array{Float64, 2}(undef, range_dim, n_vertices)
    cells = Vector{WriteVTK.MeshCell}(undef, n_total_cells)
    if isnothing(field)
        field_dim = 0
    else
        field_dim = Fields.get_image_dim(field)
    end
    point_data = zeros(field_dim,n_vertices)
    
    vertex_idx = 1
    vertex_offset = 0
    dξ = 1.0/n_subcells
    dξ_sub = dξ/degree  # a polynomial degree approximatio of p created p subdivisions of a subcell

    subcell_cartesian_idx = CartesianIndices((n_subcells, n_subcells))

    # # this is where points evaluated inside an element will be stored
    # vertices_el = zeros(n_dim, n_eval_1, n_eval_2, n_eval_3, ...)
    # el_vertex_offset += prod(n_eval)
    # vertices_el[:,i,j,k] .= vector_output_of_evaluate

    for element_idx in 1:n_total_elements
        # vertices_el = geometry.evaluate(..., NTuple_of_directional_points)
        # then, loop to store things in vtk format

        for subcell_idx in 1:(n_subcells^2)
            # Corner vertices
            corner_idx = [0, 1, 3, 2]
            count = 0
            for v_idx = 0:1
                for h_idx = 0:1
                    ξ = (dξ .* (Tuple(subcell_cartesian_idx[subcell_idx]) .-  (1-h_idx, 1-v_idx)))
                    ξ = Tuple([ξi] for ξi in ξ)
                    vertices[:, vertex_idx + corner_idx[count+1]] .= vec(Geometry.evaluate(geometry, element_idx, ξ))
                    # Compute data to plot 
                    if !isnothing(field)
                        point_data[:,vertex_idx + corner_idx[count+1]] = vec(Fields.evaluate(field, element_idx, ξ))
                    end
                    count += 1
                end
            end
            
            # Edge vertices
            edge_endpts = [(1,1), (0,1), (1,0), (1,1)]
            edge_ornt = [0, 1, 0, 1]
            step = (0.0, 0.0)
            for e = 1:4
                # starting point on edge
                ξ_0 = dξ .* (Tuple(subcell_cartesian_idx[subcell_idx]) .- edge_endpts[e])
                if edge_ornt[e] == 0
                    step = (dξ_sub, 0.0)
                elseif edge_ornt[e] == 1
                    step = (0.0, dξ_sub)
                end
                for edge_vertex_idx in 1:(degree-1)
                    ξ = ξ_0 .+ edge_vertex_idx .* step
                    ξ = Tuple([ξi] for ξi in ξ)
                    vertices[:, vertex_idx + count] .= vec(Geometry.evaluate(geometry, element_idx, ξ))
                    # Compute data to plot 
                    if !isnothing(field)
                        point_data[:,vertex_idx + count] = vec(Fields.evaluate(field, element_idx, ξ))
                    end
                    count += 1
                end
            end
            vertex_offset += count - 1

            # Add interior vertices
            ξ_0 = (dξ .* (Tuple(subcell_cartesian_idx[subcell_idx]) .- (1,1)))
            for vertex_column_idx in 1:(degree-1)
                for vertex_row_idx in 1:(degree-1)
                    interior_vertex_idx = vertex_row_idx + (vertex_column_idx - 1) * (degree - 1)
                    ξ = ξ_0 .+ (dξ_sub * vertex_row_idx, dξ_sub * vertex_column_idx)
                    ξ = Tuple([ξi] for ξi in ξ)
                    vertices[:, vertex_idx + vertex_offset + interior_vertex_idx] .= vec(Geometry.evaluate(geometry, element_idx, ξ))
                    # Compute data to plot 
                    if !isnothing(field)
                        point_data[:,vertex_idx + vertex_offset + interior_vertex_idx] = vec(Fields.evaluate(field, element_idx, ξ))
                    end
                end
            end
            vertex_offset += (degree-1)*(degree-1)  # add the number of interior vertices added
            
            # Add cell
            cell_idx = (element_idx - 1) * (n_subcells^2) + subcell_idx
            cells[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, collect(vertex_idx:(vertex_idx+vertex_offset))) 

            # Update vertex_idx to start again
            vertex_idx += vertex_offset + 1
            vertex_offset = 0
        end
    end

    WriteVTK.vtk_grid(vtk_filename, vertices, cells; append = false, ascii = ascii, compress = compress, vtkversion = :latest) do vtk 
        vtk.version == "2.2"
        if !isnothing(field)
            vtk["point_data", WriteVTK.VTKPointData()] = point_data
        end
    end
end