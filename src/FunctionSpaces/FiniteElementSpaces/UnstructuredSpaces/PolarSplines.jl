struct ScalarPolarSplineSpace{T} <: AbstractFESpace{2, 1, 1}
    patch_spaces::NTuple{1,T}
    extraction_op::ExtractionOperator
    dof_partition::Vector{Vector{Vector{Int}}}
    num_elements_per_patch::Vector{Int}
    regularity::Int

    global_extraction_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    control_triangle::Matrix{Float64}
    two_poles::Bool
    zero_at_poles::Bool

    """
        ScalarPolarSplineSpace(
            space_p::AbstractFESpace{1, 1},
            space_r::AbstractFESpace{1, 1},
            degenerate_control_points::NTuple{2, Matrix{Float64}},
            two_poles::Bool=false,
            zero_at_poles::Bool=false,
            dspace_p::Union{Nothing, AbstractFESpace{1, 1}}=nothing,
            dspace_r::Union{Nothing, AbstractFESpace{1, 1}}=nothing
        )

    Build a polar spline space from the given poloidal and radial spaces.

    # Arguments
    - `space_p::AbstractFESpace{1, 1}`: The poloidal space.
    - `space_r::AbstractFESpace{1, 1}`: The radial space.
    - `degenerate_control_points::NTuple{2,Matrix{Float64}}`: The degenerate control points.
    - `two_poles::Bool=false`: Whether the polar spline space has two poles.
    - `zero_at_poles::Bool=false`: Whether functions are constrained to zero at poles.
    - `dspace_p::Union{Nothing, AbstractFESpace{1, 1}}=nothing`: The derivative space for the poloidal space.
    - `dspace_r::Union{Nothing, AbstractFESpace{1, 1}}=nothing`: The derivative space for the radial space.

    # Returns
    - `polar_splines::AbstractFESpace`: The polar spline space.
    - `E::ExtractionOperator`: The extraction operator.
    """
    function ScalarPolarSplineSpace(
        space_p::AbstractFESpace{1, 1},
        space_r::AbstractFESpace{1, 1},
        degenerate_control_points::NTuple{2, Matrix{Float64}},
        dspace_p::Union{Nothing, AbstractFESpace{1, 1}}=nothing,
        dspace_r::Union{Nothing, AbstractFESpace{1, 1}}=nothing,
        two_poles::Bool=false,
        zero_at_poles::Bool=false
    )

        if zero_at_poles
            if isnothing(dspace_p) || isnothing(dspace_r)
                throw(ArgumentError("Derivative spaces are required for zero_at_poles=true."))
            end
        end

        # number of dofs for the poloidal and radial spaces
        n_p = get_num_basis(space_p)
        n_r = get_num_basis(space_r)

        # first, build extraction operator and control triangle
        E, control_triangle = extract_scalar_polar_splines_to_tensorproduct(
            degenerate_control_points,
            n_p,
            n_r,
            two_poles,
            zero_at_poles,
        )

        # build underlying tensor-product space
        if zero_at_poles
            tp_space = TensorProductSpace((dspace_p, dspace_r))
            regularity = -1
        else
            tp_space = TensorProductSpace((space_p, space_r))
            regularity = 1
        end

        # build polar spline space and return along with the extraction matrix
        extraction_op, dof_partition = _build_polar_spline_space(tp_space, E, control_triangle, two_poles)

        new{typeof(tp_space)}(
            (tp_space,),
            extraction_op,
            dof_partition,
            [get_num_elements(tp_space)],
            regularity,
            E,
            control_triangle,
            two_poles,
            zero_at_poles
        )
    end
end

struct VectorPolarSplineSpace{T} <: AbstractFESpace{2, 2, 1}
    patch_spaces::NTuple{1,T}
    extraction_op::ExtractionOperator
    dof_partition::Vector{Vector{Vector{Int}}}
    num_elements_per_patch::Vector{Int}
    regularity::Int

    global_extraction_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    control_triangle::Matrix{Float64}
    two_poles::Bool
    zero_at_poles::Bool
    """
        PolarSplineVectors(
            space_p::AbstractFESpace{1, 1},
            space_r::AbstractFESpace{1, 1},
            degenerate_control_points::NTuple{2, Matrix{Float64}},
            dspace_p::AbstractFESpace{1, 1},
            dspace_r::AbstractFESpace{1, 1},
            two_poles::Bool=false
        )

    Build a polar spline space from the given poloidal and radial spaces.

    # Arguments
    - `space_p::AbstractFESpace{1, 1}`: The poloidal space.
    - `space_r::AbstractFESpace{1, 1}`: The radial space.
    - `degenerate_control_points::NTuple{2,Matrix{Float64}}`: The degenerate control points.
    - `dspace_p::AbstractFESpace{1, 1}`: The derivative space for the poloidal space.
    - `dspace_r::AbstractFESpace{1, 1}`: The derivative space for the radial space.
    - `two_poles::Bool=false`: Whether the polar spline space has two poles.

    # Returns
    - `polar_splines::AbstractFESpace`: The polar spline space.
    - `E::ExtractionOperator`: The extraction operator.
    """
    function PolarSplineVectors(
        space_p::AbstractFESpace{1, 1},
        space_r::AbstractFESpace{1, 1},
        degenerate_control_points::NTuple{2, Matrix{Float64}},
        dspace_p::AbstractFESpace{1, 1},
        dspace_r::AbstractFESpace{1, 1},
        two_poles::Bool=false
    )

        # number of dofs for the poloidal and radial spaces
        n_p = get_num_basis(space_p)
        n_r = get_num_basis(space_r)

        # first, build extraction operator and control triangle
        E, control_triangle = extract_vector_polar_splines_to_tensorproduct(
            degenerate_control_points,
            n_p,
            n_r,
            two_poles,
        )

        # build underlying tensor-product space
        tp_space = (
            TensorProductSpace((dspace_p, space_r)), TensorProductSpace((space_p, dspace_r))
        )

        # build polar spline space and return along with the extraction matrix
        polar_splines = Vector{AbstractFESpace}(undef, length(E))
        for component_idx in eachindex(polar_splines)
            polar_splines[component_idx] =
                _build_polar_spline_space(
                    tp_space[component_idx], E[component_idx], control_triangle, two_poles
                )
        end

        return SumSpace(tuple(polar_splines...), maximum(map(get_num_basis, polar_splines))), E
    end
end

"""
    build_polar_spline_space(
        tp_space::TensorProductSpace,
        E::SparseMatrixCSC{Float64,Int},
        control_triangle::Matrix{Float64},
        two_poles::Bool
    )

Build the polar spline space from the tensor product space.

# Arguments
- `tp_space::TensorProductSpace`: The tensor product space.
- `E::SparseMatrixCSC{Float64,Int}`: The extraction operator.
- `control_triangle::Matrix{Float64}`: The control triangle.
- `two_poles::Bool`: Whether the polar spline space has two poles.

# Returns
- `polar_splines::AbstractFESpace`: The polar spline space.
"""
function _build_polar_extraction_and_dof_partition(tp_space, E, two_poles)
    # number of elements
    num_elements = get_num_elements(tp_space)

    # allocate space for extraction coefficients and basis indices
    extraction_coefficients = Vector{Matrix{Float64}}(undef, num_elements)
    basis_indices = Vector{Vector{Int}}(undef, num_elements)

    # Convert global extraction matrix to element local extractions
    # (the matrix is transposed so that [tp_spline_space] * [extraction] = [Polar-splines])
    count = 0
    for i in 1:num_elements
        _, cols_ij = get_extraction(tp_space, i)
        eij = SparseArrays.findnz(E[:, cols_ij])
        # Unique indices for non-zero rows and columns
        basis_indices[count + 1] = unique(eij[1])
        # Matrix of coefficients
        extraction_coefficients[count + 1] =
            Array(E[basis_indices[count + 1], cols_ij])'
        count += 1
    end

    # number of polar dofs
    space_dim = size(E, 1)

    # Polar spline extraction operator
    extraction_op = ExtractionOperator(
        extraction_coefficients, basis_indices, num_elements, space_dim
    )

    # dof partitioning for the tensor product space
    dof_partition_tp = get_dof_partition(tp_space)
    n_partn = length(dof_partition_tp[1])
    dof_partition = Vector{Vector{Vector{Int}}}(undef, 1)
    dof_partition[1] = Vector{Vector{Int}}(undef, n_partn)
    if two_poles
        for i in 1:n_partn
            dof_partition[1][i] = []
        end
    else
        for i in 1:n_partn
            if length(dof_partition_tp[1][i]) == 0
                dof_partition[1][i] = []
            else
                # TODO!!
                dof_partition[1][i] = []
            end
        end
    end

    # return data needed for building polar splines
    return extraction_op, dof_partition
end

"""
    extract_vector_polar_splines_to_tensorproduct(
        degenerate_control_points::NTuple{2, Matrix{Float64}},
        num_basis_p::Int,
        num_basis_r::Int;
        two_poles::Bool=false
    )

Build the extraction operator to extract the vector polar spline basis functions w.r.t.
the tensor product basis functions.

# Arguments
- `degenerate_control_points::NTuple{2,Matrix{Float64}}`: The degenerate control points.
- `num_basis_p::Int`: Number of poloidal basis functions.
- `num_basis_r::Int`: Number of radial basis functions.
- `two_poles::Bool=false`: Whether the polar spline space has two poles.

# Returns
- `E::Tuple{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Float64,Int}}`: The extraction operators.
The first element corresponds to the horizontal edges and the second element to the vertical edges.
- `tri::Matrix{Float64}`: The control triangle.
"""
function extract_vector_polar_splines_to_tensorproduct(
    degenerate_control_points::NTuple{2, Matrix{Float64}},
    num_basis_p::Int,
    num_basis_r::Int,
    two_poles::Bool=false
)
    if ~all(size.(degenerate_control_points, 2) .== 2)
        throw(
            ArgumentError(
                "The control points need to be two-dimensional, got $(size.(degenerate_control_points, 2))."
            )
        )
    end
    if ~all(size.(degenerate_control_points, 1) .== num_basis_p)
        throw(
            ArgumentError(
                "The input points need to be two sets of $num_basis_p points, got $(size.(degenerate_control_points, 1))."
            )
        )
    end
    if ~all(
        isapprox.(
            degenerate_control_points[1], degenerate_control_points[1][1:1,:], atol=1e-12
        )
    )
        throw(ArgumentError("All points in the first set should be identical."))
    else
        origin = degenerate_control_points[1][1, :]
    end

    # build triangle circumscribing the input degenerate control points
    tri = _get_circumscribing_triangle(vcat(degenerate_control_points...), origin)

    # number of vector basis functions
    num_basis =
        num_basis_p * num_basis_r + num_basis_p * (num_basis_r - 1) -
        (two_poles + 1) * (3 * num_basis_p - 2)

    # allocate row, col, val vectors for extraction matrix for vertical and horizontal edges
    E1_v_r = zeros(Int, num_basis * 4)
    E1_v_c = zeros(Int, num_basis * 4)
    E1_v_v = zeros(Float64, num_basis * 4)
    E1_h_r = zeros(Int, num_basis * 4)
    E1_h_c = zeros(Int, num_basis * 4)
    E1_h_v = zeros(Float64, num_basis * 4)
    ind_v = 0
    ind_h = 0

    # compute barycentric coordinates and extraction sub-matrices for the input points
    E0_1, E0_2 = _get_scalar_polar_extraction_submatrix(
        degenerate_control_points, tri, origin, two_poles
    )

    # assemble the extraction near the bottom pole
    baryc = hcat(
        reshape(E0_1[:, 1:num_basis_p]', num_basis_p, 1, 3),
        reshape(E0_1[:, num_basis_p .+ (1:num_basis_p)]', num_basis_p, 1, 3),
    )
    for k in 0:(num_basis_p - 1)
        l = 0
        kp = mod(k + 1, num_basis_p)
        r = [0, 1]

        c = k
        v = baryc[k + 1, l + 2, 2:3] .- baryc[k + 1, l + 1, 2:3]
        E1_v_r[ind_v .+ (1:length(r))] = r .+ 1
        E1_v_c[ind_v .+ (1:length(r))] .= c + 1
        E1_v_v[ind_v .+ (1:length(r))] = v
        ind_v += length(r)

        c = k + num_basis_p
        v = baryc[kp + 1, l + 2, 2:3] .- baryc[k + 1, l + 2, 2:3]
        E1_h_r[ind_h .+ (1:length(r))] = r .+ 1
        E1_h_c[ind_h .+ (1:length(r))] .= c + 1
        E1_h_v[ind_h .+ (1:length(r))] = v
        ind_h += length(r)
    end

    # assemble the extraction for intermediate vertical edges
    k = 0:(num_basis_p - 1)
    l = 2:(num_basis_r - two_poles - 1)
    for idx in Iterators.product(k, l)
        r = idx[1] + (2 * idx[2] - 4) * num_basis_p + 2
        c = idx[1] + (idx[2] - 1) * num_basis_p
        E1_v_r[ind_v + 1] = r + 1
        E1_v_c[ind_v + 1] = c + 1
        E1_v_v[ind_v + 1] = 1.0
        ind_v += 1
    end

    # assemble the extraction for intermediate horizontal edges
    k = 0:(num_basis_p - 1)
    l = 2:(num_basis_r + 1 - 2 * (two_poles + 1))
    for idx in Iterators.product(k, l)
        r = idx[1] + (2 * idx[2] - 3) * num_basis_p + 2
        c = idx[1] + idx[2] * num_basis_p
        E1_h_r[ind_h + 1] = r + 1
        E1_h_c[ind_h + 1] = c + 1
        E1_h_v[ind_h + 1] = 1.0
        ind_h += 1
    end

    # assemble the extraction near the top pole
    if two_poles
        baryc = hcat(
            reshape(E0_2[:, 1:num_basis_p]', num_basis_p, 1, 3),
            reshape(E0_2[:, num_basis_p .+ (1:num_basis_p)]', num_basis_p, 1, 3),
        )
        for k in 0:(num_basis_p - 1)
            kp = mod(k + 1, num_basis_p)
            l = num_basis_r - 2
            r = [num_1_forms - 2, num_1_forms - 1]
            c = k + l * num_basis_p

            v =
                baryc[k + 1, l - num_basis_r + 4, 2:3] .-
                baryc[k + 1, l - num_basis_r + 3, 2:3]
            E1_v_r[ind_v .+ (1:length(r))] = r .+ 1
            E1_v_c[ind_v .+ (1:length(r))] .= c + 1
            E1_v_v[ind_v .+ (1:length(r))] = v
            ind_v += length(r)

            v =
                baryc[kp + 1, l - num_basis_r + 3, 2:3] .-
                baryc[k + 1, l - num_basis_r + 3, 2:3]
            E1_h_r[ind_h .+ (1:length(r))] = r .+ 1
            E1_h_c[ind_h .+ (1:length(r))] .= c + 1
            E1_h_v[ind_h .+ (1:length(r))] = v
            ind_h += length(r)
        end
    end

    # assemble the full extraction matrix w.r.t. vertical tensor-product edges
    E1_v_r = E1_v_r[1:ind_v]
    E1_v_c = E1_v_c[1:ind_v]
    E1_v_v = E1_v_v[1:ind_v]
    E_v = SparseArrays.sparse(
        E1_v_r, E1_v_c, E1_v_v, num_basis, num_basis_p * (num_basis_r - 1)
    )
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E_v)

    # assemble the full extraction matrix w.r.t. horizontal tensor-product edges
    E1_h_r = E1_h_r[1:ind_h]
    E1_h_c = E1_h_c[1:ind_h]
    E1_h_v = E1_h_v[1:ind_h]
    E_h = SparseArrays.sparse(
        E1_h_r, E1_h_c, E1_h_v, num_basis, num_basis_p * num_basis_r
    )
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E_h)

    return (E_h, E_v), tri
end

"""
    extract_scalar_polar_splines_to_tensorproduct(
        degenerate_control_points::NTuple{2, Matrix{Float64}},
        num_basis_p::Int,
        num_basis_r::Int,
        two_poles::Bool=false,
        zero_at_poles::Bool=nothing
    )

Build the extraction operator to extract the polar spline basis functions from the tensor
product space.

# Arguments
- `degenerate_control_points::NTuple{2,Matrix{Float64}}`: The degenerate control points.
- `num_basis_p::Int`: Number of poloidal basis functions.
- `num_basis_r::Int`: Number of radial basis functions.
- `two_poles::Bool=false`: Whether the polar spline space has two poles.
- `zero_at_poles::Bool=nothing`: Whether functions are constrained to zero at poles.

# Returns
- `E::SparseMatrixCSC{Float64,Int}`: The extraction operator.
- `tri::Matrix{Float64}`: The control triangle.
"""
function extract_scalar_polar_splines_to_tensorproduct(
    degenerate_control_points::NTuple{2, Matrix{Float64}},
    num_basis_p::Int,
    num_basis_r::Int,
    two_poles::Bool=false,
    zero_at_poles::Bool=nothing
)
    if ~all(size.(degenerate_control_points, 2) .== 2)
        throw(
            ArgumentError(
                "The control points need to be two-dimensional, got $(size.(degenerate_control_points, 2))."
            )
        )
    end
    if ~all(size.(degenerate_control_points, 1) .== num_basis_p)
        throw(
            ArgumentError(
                "The input points need to be two sets of $num_basis_p points, got $(size.(degenerate_control_points, 1))."
            )
        )
    end
    if ~all(
        isapprox.(
            degenerate_control_points[1], degenerate_control_points[1][1:1,:], atol=1e-12
        )
    )
        throw(ArgumentError("All points in the first set should be identical."))
    else
        origin = degenerate_control_points[1][1, :]
    end

    # build triangle circumscribing the input degenerate control points
    tri = _get_circumscribing_triangle(vcat(degenerate_control_points...), origin)

    if zero_at_poles
        # eliminate all functions that are non-zero at the poles
        num_basis = num_basis_p * (num_basis_r - 2 - two_poles)

        # allocate row, col, val vectors for extraction matrix
        E2_r = zeros(Int, num_basis)
        E2_c = zeros(Int, num_basis)
        E2_v = zeros(Float64, num_basis)
        ind = 0

        # generate data for extraction
        k = 0:(num_basis_p - 1)
        l = 1:(num_basis_r - 2 - two_poles)
        for idx in Iterators.product(k, l)
            r = idx[1] + (idx[2] - 1) * num_basis_p
            c = idx[1] + idx[2] * num_basis_p
            E2_r[ind + 1] = r + 1
            E2_c[ind + 1] = c + 1
            E2_v[ind + 1] = 1.0
            ind += 1
        end

        # assembly
        E = SparseArrays.sparse(
            E2_r, E2_c, E2_v, num_basis, num_basis_p * (num_basis_r - 1)
        )

    else
        # two rows of functions at each pole will be linearly combined into three
        num_basis = num_basis_p * num_basis_r - (two_poles + 1) * (2 * num_basis_p - 3)

        # compute barycentric coordinates and extraction sub-matrices for the input points
        E0_1, E0_2 = _get_scalar_polar_extraction_submatrix(
            degenerate_control_points, tri, origin, two_poles
        )

        # assemble full polar extraction matrix
        E = SparseArrays.blockdiag(
            E0_1, SparseArrays.spdiagm(ones(max(0, num_basis - 6))), E0_2
        )
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E)

    end

    return E, tri
end

"""
    _get_scalar_polar_extraction_submatrix(
        degenerate_control_points::NTuple{2, Matrix{Float64}},
        tri::Matrix{Float64},
        origin::Vector{Float64}=[0.0, 0.0],
        two_poles::Bool=false
    )

Compute the extraction sub-matrices for the scalar polar spline space. These are used to
build the scalar polar spline extraction operator, as well as the vector polar spline
extraction operator.

# Arguments
- `degenerate_control_points::NTuple{2,Matrix{Float64}}`: The degenerate control points.
- `tri::Matrix{Float64}`: The control triangle.
- `origin::Vector{Float64}=[0.0, 0.0]`: The origin.
- `two_poles::Bool=false`: Whether the polar spline space has two poles.

# Returns
- `E0_1::SparseMatrixCSC{Float64,Int}`: The extraction sub-matrix for the first pole.
- `E0_2::SparseMatrixCSC{Float64,Int}`: The extraction sub-matrix for the second pole.
"""
function _get_scalar_polar_extraction_submatrix(
    degenerate_control_points::NTuple{2, Matrix{Float64}},
    tri::Matrix{Float64},
    origin::Vector{Float64}=[0.0, 0.0],
    two_poles::Bool=false
)
    # compute barycentric coordinates w.r.t. the control triangle
    baryc_1 = permutedims(
        _get_barycentric_coordinates(degenerate_control_points[1], tri, origin)
    )
    baryc_2 = permutedims(
        _get_barycentric_coordinates(degenerate_control_points[2], tri, origin)
    )

    # create extraction matrix for first pole
    E0_1 = SparseArrays.SparseArrays.sparse([baryc_1 baryc_2])
    if two_poles
        # create symmetric extraction matrix for second pole
        E0_2 = reverse(E0_1; dims=2)
    else
        E0_2 = SparseArrays.spdiagm(ones(3))
    end
    return E0_1, E0_2
end

"""
    _get_circumscribing_triangle(points, origin)

Build a standard triangle that contains the input points.

# Arguments
- `points::Matrix{Float64}`: The input points, where each row corresponds to a point.
- `origin::Vector{Float64}`: The point which is translated to the origin.

# Returns
- `::Matrix{Float64}`: The circumscribing triangle.
"""
function _get_circumscribing_triangle(
    points::Matrix{Float64}, origin::Vector{Float64}=[0.0, 0.0]
)
    rad = sqrt.(sum((points .- permutedims(origin)) .^ 2; dims=2))
    tau = 2 * maximum(rad)
    trix = [tau, -tau / 2, -tau / 2]
    triy = [0, -sqrt(3) * tau / 2, sqrt(3) * tau / 2]
    return [trix triy]
end

"""
    _get_barycentric_coordinates(points, tri, origin)

Compute the barycentric coordinates of the input points w.r.t. the control triangle. The
points are translated by the offset before computing the barycentric coordinates.

# Arguments
- `points::Matrix{Float64}`: The input points, where each row corresponds to a point.
- `tri::Matrix{Float64}`: Matrix with the coordinates of the `i`-th vertex of the control
    triangle in the `i`-th row.
- `origin::Vector{Float64}`: The point which is translated to the origin.

# Returns
- `::Matrix{Float64}`: The barycentric coordinates.
"""
function _get_barycentric_coordinates(
    points::Matrix{Float64}, tri::Matrix{Float64}, origin::Vector{Float64}=[0.0, 0.0]
)
    L = [tri ones(3)]
    return hcat(points .- permutedims(origin), ones(size(points, 1))) / L
end
