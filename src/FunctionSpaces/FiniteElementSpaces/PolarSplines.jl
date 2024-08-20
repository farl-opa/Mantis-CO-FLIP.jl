"""
    PolarSplineSpace(space_p::AbstractFiniteElementSpace{1}, space_r::AbstractFiniteElementSpace{1})

Create a polar spline space from the given poloidal and radial finite element spaces.

# Arguments
- `space_p::AbstractFiniteElementSpace{1}`: The poloidal finite element space.
- `space_r::AbstractFiniteElementSpace{1}`: The radial finite element space.

# Returns
- `UnstructuredSpace`: The resulting polar spline space.

# Throws
- `ArgumentError`: If the poloidal space has boundary degrees of freedom (DOFs), indicating it is likely not periodic.
"""
function PolarSplineSpace(space_p::AbstractFiniteElementSpace{1}, space_r::AbstractFiniteElementSpace{1})

    # number of dofs for the poloidal and radial spaces
    n_p = get_num_basis(space_p)
    n_r = get_num_basis(space_r)

    # Get dof partitions of the poloidal space ...
    dof_partition_p = get_dof_partition(space_p)
    # ... and check that the poloidal space has no bounday dofs ...
    bdof_p = vcat(dof_partition_p[1][1], dof_partition_p[1][end])
    # ... and throw an error if it does.
    if length(bdof_p) > 0
        throw(ArgumentError("Poloidal space has boundary dofs and is likely not periodic."))
    end

    # tensor-product space which will form the foundation of the polar spline space
    tp_space = TensorProductSpace(space_p, space_r)

    # polar control points
    _, _, theta = build_base_polar_control_points(n_p, n_r, 1.0)

    # the extraction sub-matrix in the neighborhood of the polynomial_degree
    ex_coeff_pole = Matrix{Float64}(undef,2*n_p,3)
    for j ∈ 1:n_p
        ex_coeff_pole[j,:] .= 1/3
    end
    t_mat = Matrix{Float64}(undef,3,3)
    t_mat[:,1] .= [1/3, -1/6, -1/6]
    t_mat[:,2] .= [0, sqrt(3)/6, -sqrt(3)/6]
    t_mat[:,3] .= [1/3, 1/3, 1/3]
    for j = n_p+1:2*n_p
        ex_coeff_pole[j,:] = t_mat * [cos(theta[j-n_p]), sin(theta[j-n_p]), 1]
    end
    # polar offset
    pole_offset = 2*n_p - 3

    # number of elements
    num_elements = get_num_elements(tp_space)

    # number of polar dofs
    space_dim = get_num_basis(tp_space) - pole_offset

    # allocate space for extraction coefficients and basis indices
    extraction_coefficients = Vector{Array{Float64}}(undef,num_elements)
    basis_indices = Vector{Vector{Int}}(undef,num_elements)
    for e ∈ 1:num_elements
        # get tp_extraction basis indices
        _, tp_basis_inds = get_extraction(tp_space,e)
        n_tp_basis = length(tp_basis_inds)
        # if none of the basis functions are relevant for the polar construction, then extraction is identity
        if minimum(tp_basis_inds) > 2*n_p
            extraction_coefficients[e] = Matrix(LinearAlgebra.I, n_tp_basis, n_tp_basis)
            basis_indices[e] = tp_basis_inds .- pole_offset
        else
            # find which functions are unmodified
            i_unmodified = findall(tp_basis_inds .> 2*n_p)
            i_modified = findall(tp_basis_inds .<= 2*n_p)
            # indices of all active functions
            basis_indices[e] = cat([1,2,3],tp_basis_inds[i_unmodified] .- pole_offset,dims=1)
            # extraction coefficients will be stored here
            el_extraction = Array{Float64}(undef,n_tp_basis,length(i_unmodified)+3)
            # extraction coefficients for modified basis functions
            for j ∈ 1:3
                el_extraction[i_modified,j] .= ex_coeff_pole[tp_basis_inds[i_modified],j]
            end
            # extraction coefficients for unmodified basis functions
            count = 1
            for i ∈ i_unmodified
                el_extraction[i,count+3] = 1.0
                count += 1
            end
            extraction_coefficients[e] = el_extraction
        end
    end
    
    # Polar spline extraction operator
    extraction_op = ExtractionOperator(extraction_coefficients, basis_indices, num_elements, space_dim)

    # unstructured spline config
    patch_neighbours = [-1
                        -1]
    # number of elements per patch
    patch_nels = [0; get_num_elements(tp_space)]

    # assemble patch config in a dictionary
    us_config = Dict("patch_neighbours" => patch_neighbours, "patch_nels" => patch_nels)

    # dof partitioning for the tensor product space
    dof_partition_tp = get_dof_partition(tp_space)
    n_partn = length(dof_partition_tp)
    dof_partition = Vector{Vector{Vector{Int}}}(undef,1)
    dof_partition[1] = Vector{Vector{Int}}(undef, n_partn)
    for i ∈ 1:n_partn
        if length(dof_partition_tp[i]) == 0
            dof_partition[1][i] = []
        else
            i_unmodified = findall(dof_partition_tp[i] .> 2*n_p)
            i_modified = findall(dof_partition_tp[i] .<= 2*n_p)
            if length(i_modified) == 0
                dof_partition[1][i] = dof_partition_tp[i] .- pole_offset
            else
                dof_partition[1][i] = cat([1,2,3],dof_partition_tp[i][i_unmodified] .- pole_offset,dims=1)
            end
        end
    end
    
    # return Polar spline space
    return UnstructuredSpace((tp_space,), extraction_op, dof_partition, us_config, Dict("regularity" => 1))
end

"""
    build_base_polar_control_points(n_p::Int, n_r::Int, R::Float64)

Generate the base polar control points for a given number of poloidal and radial divisions.

# Arguments
- n_p::Int: Number of poloidal divisions.
- n_r::Int: Number of radial divisions.
- R::Float64: Radius of the polar space.

# Returns
- polar_control_points::Matrix{Float64}: The generated polar control points.
- radii::Vector{Float64}: The radii values.
- theta::Vector{Float64}: The theta values.
"""
function build_base_polar_control_points(n_p::Int, n_r::Int, R::Float64)
    radii = LinRange(0.0, R, n_r)
    theta = 2*pi .+ (1 .- 2 .* (1:n_p)) .* pi / n_p
    polar_control_points = zeros(Float64, n_p*(n_r-2)+3, 2)
    # the control points that are identical to the tensor-product control points
    for idx ∈ Iterators.product(1:n_p, 3:n_r)
        pidx = ordered_to_linear_index(idx, (n_p, n_r))
        pidx = pidx - (2*n_p - 3)
        polar_control_points[pidx,1] = radii[idx[2]]*cos(theta[idx[1]])
        polar_control_points[pidx,2] = radii[idx[2]]*sin(theta[idx[1]])
    end
    # the control points for the control triangle
    rho_2 = radii[2]
    polar_control_points[1,:] .= [2*rho_2, 0]
    polar_control_points[2,:] .= [-rho_2, sqrt(3)*rho_2]
    polar_control_points[3,:] .= [-rho_2, -sqrt(3)*rho_2]

    return polar_control_points, radii, theta
end