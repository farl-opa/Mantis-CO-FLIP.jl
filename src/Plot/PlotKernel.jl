function _plot(geometry::Geometry.AbstractGeometry{manifold_dim}, field::Union{Nothing, F} = nothing, offset::Union{Nothing, Function} = nothing; vtk_filename::String = "default", n_subcells::Int = 1, degree::Int = 1, ascii = false, compress = true, subcell_wireframe = true) where {manifold_dim, F <: Fields.AbstractField{manifold_dim, field_dim} where {field_dim}}
    # This function generates points per plotted nD cell, so connectivity is lost, this is what requires less information
    # from the mesh. Each computational element is sampled at n_subsamples (minimum is 2 per direction). These subsamples 
    # create a structured grid, each cell of this refined grid is plotted.
    # Given Paraview's capabilities, the range_dim must be at most 3.
    
    range_dim = Geometry.get_image_dim(geometry)
    @assert range_dim <= 3 && manifold_dim <= 3 "Paraview can only plot 3D geometries."

    # Univariate evaluation coordinates for degree p
    ξ_ref_per_dir = collect(LinRange(0.0, 1.0, degree+1))
    # VTK ordering of cartesian grid of evaluation nodes within the reference subcell
    if manifold_dim == 1
        I_ref = node_ordering("line", degree)
    elseif manifold_dim == 2
        I_ref = node_ordering("quadrilateral", degree)
    elseif manifold_dim == 3
        I_ref = node_ordering("hexahedron", degree)
    end

    # Compute the total number of points
    n_total_elements = Geometry.get_num_elements(geometry)
    n_vertices_per_subcell = (degree+1)^manifold_dim
    n_total_cells = n_total_elements * (n_subcells^manifold_dim)  # each computational element is subdivided into n_subcells per direction
    n_vertices = n_total_cells * n_vertices_per_subcell  # there are n_subcells cells per element per direction, and (p+1) vertices per cell per direction

    ###################################
    ### EXPORT MAXIMAL CELLS
    ###################################

    vertices = Array{Float64, 2}(undef, range_dim, n_vertices)
    cells = Vector{WriteVTK.MeshCell}(undef, n_total_cells)
    if isnothing(field)
        field_dim = 0
    else
        field_dim = Fields.get_image_dim(field)
    end
    point_data = zeros(field_dim,n_vertices)

    vertex_offset = 0
    dξ = 1.0/n_subcells
    subcell_cartesian_idx = CartesianIndices(Tuple(n_subcells*ones(Int,manifold_dim)))

    # # this is where points evaluated inside an element will be stored
    # vertices_el = zeros(n_dim, n_eval_1, n_eval_2, n_eval_3, ...)
    # el_vertex_offset += prod(n_eval)
    # vertices_el[:,i,j,k] .= vector_output_of_evaluate

    for element_idx in 1:n_total_elements
        for subcell_idx in 1:n_subcells^manifold_dim
            ξ_shift_per_dir = dξ .* (Tuple(subcell_cartesian_idx[subcell_idx]) .-  Tuple(ones(Int,manifold_dim)))
            ξ = Tuple(ξ_shift_per_dir[i] .+ dξ .* ξ_ref_per_dir for i in 1:manifold_dim)
            # evaluate geometry and rearrange 
            vertices[:, vertex_offset .+ (1:n_vertices_per_subcell)] .= (Geometry.evaluate(geometry, element_idx, ξ)[I_ref, :])'

            # compute data to plot 
            if !isnothing(field)
                point_data[:, vertex_offset .+ (1:n_vertices_per_subcell)] = (Fields.evaluate(field, element_idx, ξ)[I_ref, :])'
            end
            # apply offset
            if !isnothing(offset)
                point_data[:, vertex_offset .+ (1:n_vertices_per_subcell)] .-= offset(vertices[:, vertex_offset .+ (1:n_vertices_per_subcell)]')'
            end
            
            # Add cell
            cell_idx = (element_idx - 1) * (n_subcells^manifold_dim) + subcell_idx
            if manifold_dim == 1
                cells[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_CURVE, collect(vertex_offset+1:(vertex_offset + n_vertices_per_subcell)))
            elseif manifold_dim == 2
                cells[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, collect(vertex_offset+1:(vertex_offset + n_vertices_per_subcell))) 
            else
                cells[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_HEXAHEDRON, collect(vertex_offset+1:(vertex_offset + n_vertices_per_subcell))) 
            end

            vertex_offset += (degree+1)^manifold_dim  # add the number of interior vertices added
        end
    end

    WriteVTK.vtk_grid(vtk_filename, vertices, cells; append = false, ascii = ascii, compress = compress, vtkversion = :latest) do vtk 
        vtk.version == "2.2"
        if !isnothing(field)
            vtk["point_data", WriteVTK.VTKPointData()] = point_data
        end
    end

    ###################################
    ### EXPORT EDGE WIREFRAME
    ###################################

    if manifold_dim > 1 && subcell_wireframe 

        if manifold_dim == 2
            edge_ordering = quadrilateral_edge_ordering()
            corner_vertices = reference_quadrilateral_corner_vertices()
        elseif manifold_dim == 3
            edge_ordering = hexahedron_edge_ordering()
            corner_vertices = reference_hexahedron_corner_vertices()
        end
        I_ref = node_ordering("line", degree)

        n_edges_per_element = length(edge_ordering)
        n_total_edges = n_total_elements * n_subcells * n_edges_per_element
        n_vertices_per_subedge = degree+1
        n_vertices_on_edges = n_total_edges * n_vertices_per_subedge

        vertices_on_edges = Array{Float64, 2}(undef, range_dim, n_vertices_on_edges)
        edges = Vector{WriteVTK.MeshCell}(undef, n_total_edges)

        vertex_offset = 0
        dξ = 1.0/n_subcells
        subcell_cartesian_idx = CartesianIndices(Tuple(n_subcells*ones(Int,1)))

        for element_idx in 1:n_total_elements
            for edge_idx in 1:n_edges_per_element
                # corner vertices that form this edge
                edge_verts = [corner_vertices[i] for i in edge_ordering[edge_idx]]
                # step size along the edge
                dξ_edge = dξ .* (edge_verts[2] .- edge_verts[1])
                # reference coordinates along the edge
                ξ_ref_scaled = Tuple((dξ_edge[i] > 0.0) ? (dξ .* ξ_ref_per_dir) : [edge_verts[1][i] * 1.0] for i in 1:manifold_dim)

                for subcell_idx in 1:n_subcells
                    ξ_shift_per_dir = dξ_edge .* (subcell_cartesian_idx[subcell_idx][1] - 1)
                    ξ = Tuple(ξ_shift_per_dir[i] .+ ξ_ref_scaled[i] for i in 1:manifold_dim)
                    # evaluate geometry and rearrange 
                    vertices_on_edges[:, vertex_offset .+ (1:n_vertices_per_subedge)] .= (Geometry.evaluate(geometry, element_idx, ξ)[I_ref, :])'

                    # Add cell
                    cell_idx = (element_idx - 1) * (n_subcells * n_edges_per_element) + (edge_idx - 1) * n_subcells + subcell_idx
                    edges[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_CURVE, collect(vertex_offset+1:(vertex_offset + n_vertices_per_subedge)))
                    
                    vertex_offset += (degree+1)  # add the number of interior vertices added
                end
            end
        end
        
        WriteVTK.vtk_grid(vtk_filename * "_wireframe", vertices_on_edges, edges; append = false, ascii = ascii, compress = compress, vtkversion = :latest) do vtk 
            vtk.version == "2.2"
        end
    end
end

function _plot(form::Forms.AbstractFormExpression{manifold_dim, form_rank, G}, offset::Union{Nothing, Function} = nothing; vtk_filename::String = "default", n_subcells::Int = 1, degree::Int = 1, ascii = false, compress = true, subcell_wireframe = true) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}}
    # This function generates points per plotted nD cell, so connectivity is lost, this is what requires less information
    # from the mesh. Each computational element is sampled at n_subsamples (minimum is 2 per direction). These subsamples 
    # create a structured grid, each cell of this refined grid is plotted.
    
    # Enforce that degree is at least 1, which is the minimum required by paraview 
    degree = max(degree, 1)

    # Given Paraview's capabilities, the range_dim must be at most 3.
    geometry = Forms.get_geometry(form)
    range_dim = Geometry.get_image_dim(geometry)
    @assert range_dim <= 3 && manifold_dim <= 3 "Paraview can only plot 3D geometries."

    # Univariate evaluation coordinates for degree p
    ξ_ref_per_dir = collect(LinRange(0.0, 1.0, degree+1))
    # VTK ordering of cartesian grid of evaluation nodes within the reference subcell
    if manifold_dim == 1
        I_ref = node_ordering("line", degree)
    elseif manifold_dim == 2
        I_ref = node_ordering("quadrilateral", degree)
    elseif manifold_dim == 3
        I_ref = node_ordering("hexahedron", degree)
    end

    # Compute the total number of points
    n_total_elements = Geometry.get_num_elements(geometry)
    n_vertices_per_subcell = (degree+1)^manifold_dim
    n_total_cells = n_total_elements * (n_subcells^manifold_dim)  # each computational element is subdivided into n_subcells per direction
    n_vertices = n_total_cells * n_vertices_per_subcell  # there are n_subcells cells per element per direction, and (p+1) vertices per cell per direction

    ###################################
    ### EXPORT MAXIMAL CELLS
    ###################################

    # Check if form expression is made up of fields of contains basis functions (form space)
    # We do this by evaluating all elements and checking if the indices are all the same (form field base expression)
    _, form_indices = Forms.evaluate(form, 1, Tuple([0.0] for _ = 1:manifold_dim))  # evaluate at the [0.0, 0.0] point of the first element
    form_index_ref = form_indices[1][1]  # use one index as reference 
    is_field = true  # then check if all indices are the same
    for element_idx in 1:n_total_elements
        _, form_indices = Forms.evaluate(form, element_idx, Tuple([0.0] for _ = 1:manifold_dim))
        for form_component_indices in form_indices
            is_field = is_field & all(y -> y == form_index_ref, form_component_indices)
        end
    end
    
    vertices = Array{Float64, 2}(undef, range_dim, n_vertices)
    cells = Vector{WriteVTK.MeshCell}(undef, n_total_cells)
    if !is_field
        field_dim = 0
    else
        field_dim = binomial(manifold_dim, form_rank)
    end
    if form_rank == 1 && form_rank < manifold_dim
        point_data = zeros(3, n_vertices) # 1-forms in 2D and 1- and 2-forms in 3D are plotted as vector fields
    else
        point_data = zeros(field_dim, n_vertices)
    end

    vertex_offset = 0
    dξ = 1.0/n_subcells
    subcell_cartesian_idx = CartesianIndices(Tuple(n_subcells*ones(Int,manifold_dim)))
    
    # # this is where points evaluated inside an element will be stored
    # vertices_el = zeros(n_dim, n_eval_1, n_eval_2, n_eval_3, ...)
    # el_vertex_offset += prod(n_eval)
    # vertices_el[:,i,j,k] .= vector_output_of_evaluate

    for element_idx in 1:n_total_elements
        for subcell_idx in 1:n_subcells^manifold_dim
            ξ_shift_per_dir = dξ .* (Tuple(subcell_cartesian_idx[subcell_idx]) .-  Tuple(ones(Int,manifold_dim)))
            ξ = Tuple(ξ_shift_per_dir[i] .+ dξ .* ξ_ref_per_dir for i in 1:manifold_dim)
            # evaluate geometry and rearrange 
            vertices[:, vertex_offset .+ (1:n_vertices_per_subcell)] .= (Geometry.evaluate(geometry, element_idx, ξ)[I_ref, :])'
            
            # Compute data to plot 
            if is_field
                if form_rank == 0
                    point_data[:, vertex_offset .+ (1:n_vertices_per_subcell)] = hcat(Forms.evaluate(form, element_idx, ξ)[1]...)[I_ref, :]'
                    
                elseif form_rank == manifold_dim
                    point_data[:, vertex_offset .+ (1:n_vertices_per_subcell)] .= hcat(Forms.evaluate(Forms.hodge(form), element_idx, ξ)[1]...)[I_ref, :]'

                elseif form_rank == 1
                    if range_dim == 2
                        # convert 2D vector field to 3D vector field
                        point_data[:, vertex_offset .+ (1:n_vertices_per_subcell)] .= vcat(hcat(reduce.(+, Forms.evaluate_sharp_pushforward(form, element_idx, ξ)[1],dims=2)...)[I_ref,:]', zeros(n_vertices_per_subcell))

                    elseif range_dim == 3
                        point_data[:, vertex_offset .+ (1:n_vertices_per_subcell)] .= hcat(reduce.(+, Forms.evaluate_sharp_pushforward(form, element_idx, ξ)[1],dims=2)...)[I_ref,:]'

                    end
                end
            end

            if !isnothing(offset)
                point_data[:, vertex_offset .+ (1:n_vertices_per_subcell)] .-= offset(vertices[:, vertex_offset .+ (1:n_vertices_per_subcell)]')'
            end

            # Add cell
            cell_idx = (element_idx - 1) * (n_subcells^manifold_dim) + subcell_idx
            if manifold_dim == 1
                cells[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_CURVE, collect(vertex_offset+1:(vertex_offset + n_vertices_per_subcell)))
            elseif manifold_dim == 2
                cells[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, collect(vertex_offset+1:(vertex_offset + n_vertices_per_subcell))) 
            else
                cells[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_HEXAHEDRON, collect(vertex_offset+1:(vertex_offset + n_vertices_per_subcell))) 
            end

            vertex_offset += (degree+1)^manifold_dim  # add the number of interior vertices added
        end
    end

    WriteVTK.vtk_grid(vtk_filename, vertices, cells; append = false, ascii = ascii, compress = compress, vtkversion = :latest) do vtk 
        vtk.version == "2.2"
        if is_field
            vtk["point_data", WriteVTK.VTKPointData()] = point_data
        end
    end

    ###################################
    ### EXPORT EDGE WIREFRAME
    ###################################

    if manifold_dim > 1 && subcell_wireframe 

        if manifold_dim == 2
            edge_ordering = quadrilateral_edge_ordering()
            corner_vertices = reference_quadrilateral_corner_vertices()
        elseif manifold_dim == 3
            edge_ordering = hexahedron_edge_ordering()
            corner_vertices = reference_hexahedron_corner_vertices()
        end
        I_ref = node_ordering("line", degree)

        n_edges_per_element = length(edge_ordering)
        n_total_edges = n_total_elements * n_subcells * n_edges_per_element
        n_vertices_per_subedge = degree+1
        n_vertices_on_edges = n_total_edges * n_vertices_per_subedge

        vertices_on_edges = Array{Float64, 2}(undef, range_dim, n_vertices_on_edges)
        edges = Vector{WriteVTK.MeshCell}(undef, n_total_edges)

        vertex_offset = 0
        dξ = 1.0/n_subcells
        subcell_cartesian_idx = CartesianIndices(Tuple(n_subcells*ones(Int,1)))

        for element_idx in 1:n_total_elements
            for edge_idx in 1:n_edges_per_element
                # corner vertices that form this edge
                edge_verts = [corner_vertices[i] for i in edge_ordering[edge_idx]]
                # step size along the edge
                dξ_edge = dξ .* (edge_verts[2] .- edge_verts[1])
                # reference coordinates along the edge
                ξ_ref_scaled = Tuple((dξ_edge[i] > 0.0) ? (dξ .* ξ_ref_per_dir) : [edge_verts[1][i] * 1.0] for i in 1:manifold_dim)

                for subcell_idx in 1:n_subcells
                    ξ_shift_per_dir = dξ_edge .* (subcell_cartesian_idx[subcell_idx][1] - 1)
                    ξ = Tuple(ξ_shift_per_dir[i] .+ ξ_ref_scaled[i] for i in 1:manifold_dim)
                    # evaluate geometry and rearrange 
                    vertices_on_edges[:, vertex_offset .+ (1:n_vertices_per_subedge)] .= (Geometry.evaluate(geometry, element_idx, ξ)[I_ref, :])'

                    # Add cell
                    cell_idx = (element_idx - 1) * (n_subcells * n_edges_per_element) + (edge_idx - 1) * n_subcells + subcell_idx
                    edges[cell_idx] = WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LAGRANGE_CURVE, collect(vertex_offset+1:(vertex_offset + n_vertices_per_subedge)))
                    
                    vertex_offset += (degree+1)  # add the number of interior vertices added
                end
            end
        end
        
        WriteVTK.vtk_grid(vtk_filename * "_wireframe", vertices_on_edges, edges; append = false, ascii = ascii, compress = compress, vtkversion = :latest) do vtk 
            vtk.version == "2.2"
        end
    end
    
end

function n_verts_between(n, frm, to)
    """Places `n` vertices on the edge between `frm` and `to`"""
    if n <= 0
        return Vector{NTuple{3,Int}}(undef, 0) # empty
    end
    edge_verts = hcat(collect.([LinRange(frm[i]*(n+1), to[i]*(n+1), n+2) for i in 1:3])...)
    return [Tuple(Int.(edge_verts[i,:])) for i in 2:n+1] # exclude start and end point
end

function reference_line_corner_vertices()
    return [(0, 0, 0), (1, 0, 0)]
end

function number_line(corner_verts::Vector{NTuple{3,Int}}, order::Int; skip::Bool = false)
    """Outputs the list of coordinates of a line of arbitrary order in the right ordering"""
    # initialize empty coordinates
    coords = Vector{NTuple{3,Int}}(undef, 0)
    # second: edges
    num_verts_on_edge = order - 1
    edges = [(1,2)]
    for (frm, to) in edges
        if !skip
            coords = vcat(coords, n_verts_between(num_verts_on_edge, corner_verts[frm], corner_verts[to]))
        end
    end
    # first: vertices
    coords = !skip ? vcat([corner_verts[i] .* order for i in eachindex(corner_verts)], coords) : coords # add corners if not skipped
    return coords
end

function quadrilateral_edge_ordering()
    return [(1,2), (2,3), (4,3), (1,4)]
end

function reference_quadrilateral_corner_vertices()
    return [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
end

function number_quadrilateral(corner_verts::Vector{NTuple{3,Int}}, order::Int; skip::Bool = false)
    """Outputs the list of coordinates of a right-angled quadrilateral of arbitrary order in the right ordering"""
    # initialize empty coordinates
    coords = Vector{NTuple{3,Int}}(undef, 0)
    # second: edges
    num_verts_on_edge = order - 1
    edges = quadrilateral_edge_ordering()
    for (frm, to) in edges
        if !skip
            coords = vcat(coords, n_verts_between(num_verts_on_edge, corner_verts[frm], corner_verts[to]))
        end
    end
    # third: face
    e_x = corner_verts[2] .- corner_verts[1]
    e_y = corner_verts[4] .- corner_verts[1]
    for j in 1:num_verts_on_edge
        pos_y = corner_verts[1] .* order .+ (j .* e_y)
        for i in 1:num_verts_on_edge
            pos_yx = pos_y .+ (i .* e_x)
            coords = vcat(coords, pos_yx)
        end
    end
    # first: vertices
    coords = !skip ? vcat([corner_verts[i] .* order for i in eachindex(corner_verts)], coords) : coords # add corners if not skipped
    return coords
end

function hexahedron_edge_ordering()
    return [(1,2), (2,3), (4,3), (1,4), (5,6), (6,7), (8,7), (5,8), (1,5), (2,6), (3,7), (4,8)]
end

function hexahedron_face_ordering()
    return [(1,4,8,5), (2,3,7,6), (1,2,6,5), (4,3,7,8), (1,2,3,4), (5,6,7,8)]
end

function reference_hexahedron_corner_vertices()
    return [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
end

function number_hexahedron(corner_verts::Vector{NTuple{3,Int}}, order::Int)
    """Outputs the list of coordinates of a right-angled hexahedron of arbitrary order in the right ordering"""
    # initialize empty coords
    coords = Vector{NTuple{3,Int}}(undef, 0)
    # second: edges
    edges = hexahedron_edge_ordering()
    num_verts_on_edge = order - 1
    for (frm, to) in edges
        coords = vcat(coords, n_verts_between(num_verts_on_edge, corner_verts[frm], corner_verts[to]))
    end
    # third: faces
    faces = hexahedron_face_ordering()
    for indices in faces
        sub_corner_verts = [corner_verts[q] for q in indices]
        face_coords = number_quadrilateral(sub_corner_verts, order, skip=true) # use number_quadrilateral to number face, but skip corners and edges
        coords = vcat(coords, face_coords)
    end
    # fourth: interior
    e_x = corner_verts[2] .- corner_verts[1]
    e_y = corner_verts[4] .- corner_verts[1]
    e_z = corner_verts[5] .- corner_verts[1]
    for k in 1:num_verts_on_edge
        pos_z = corner_verts[1] .* order .+ (k .* e_z)
        for j in 1:num_verts_on_edge
            pos_zy = pos_z .+ (j .* e_y)
            for i in 1:num_verts_on_edge
                pos_zyx = pos_zy .+ (i .* e_x)
                coords = vcat(coords, pos_zyx)
            end
        end
    end
    # first: corner vertices
    return vcat([corner_verts[i] .* order for i in eachindex(corner_verts)], coords)
end

function reduced_node_ordering(ordering::Vector{NTuple{3,Int}}, manifold_dim::Int, order::Int)
    # max dimensions in each direction
    max_dim = [(order+1) .* ones(Int, manifold_dim); zeros(Int, 3-manifold_dim)]
    # linearize and convert to 1-based indexing
    return [ordering[i][1] + ordering[i][2]*max_dim[1] + ordering[i][3]*prod(max_dim[1:2]) for i in eachindex(ordering)] .+ 1
end

function node_ordering(element_type, order)
    order = Int(order)
    if order < 1 || order > 10
        throw(ArgumentError("order must in interval [1, 10]"))
    end
    if element_type == "line"
        return reduced_node_ordering(number_line(reference_line_corner_vertices(), order), 1, order)
    elseif element_type == "quadrilateral"
        return reduced_node_ordering(number_quadrilateral(reference_quadrilateral_corner_vertices(), order), 2, order)
    elseif element_type == "hexahedron"
        return reduced_node_ordering(number_hexahedron(reference_hexahedron_corner_vertices(), order), 3, order)
    else
        throw(ArgumentError("Unknown element type '$element_type'"))
    end
end