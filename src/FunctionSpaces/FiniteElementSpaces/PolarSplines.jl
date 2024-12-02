"""
    PolarSplineSpace(space_p::AbstractFiniteElementSpace{1}, space_r::AbstractFiniteElementSpace{1})

Create a polar spline space from the given poloidal and radial finite element spaces.

# Arguments
- `space_p::AbstractFiniteElementSpace{1}`: The poloidal finite element space.
- `space_r::AbstractFiniteElementSpace{1}`: The radial finite element space.
- `degenerate_control_points::NTuple{2,Matrix{Float64}}`: The degenerate control points.
- `singularity_type::Int=1`: The type of singularity in the polar spline space.
- `form_rank::Int=0`: The rank of the form for which the polar spline space is to be built.
- `dspace_p::AbstractFiniteElementSpace{1}=nothing`: The derivative space for the poloidal space.
- `dspace_r::AbstractFiniteElementSpace{1}=nothing`: The derivative space for the radial space.

# Returns
- `polar_splines::UnstructuredSpace...`: The polar spline space(s).

"""
function PolarSplineSpace(space_p::AbstractFiniteElementSpace{1}, space_r::AbstractFiniteElementSpace{1}, degenerate_control_points::NTuple{2,Matrix{Float64}}; singularity_type::Int=1, form_rank::Int=0, dspace_p::Union{Nothing,AbstractFiniteElementSpace{1}}=nothing, dspace_r::Union{Nothing,AbstractFiniteElementSpace{1}}=nothing)

    # number of dofs for the poloidal and radial spaces
    n_p = get_num_basis(space_p)
    n_r = get_num_basis(space_r)

    if form_rank > 0
        if isnothing(dspace_p) || isnothing(dspace_r)
            throw(ArgumentError("Derivative spaces are required for form_rank > 0 but haven't been provided."))
        end
    end

    # first, build extraction operator
    E, control_triangle = build_polar_extraction_operators(singularity_type, degenerate_control_points, n_p, n_r, form_rank=form_rank)
    E = E[form_rank+1]
    # tensor-product space which will form the foundation of the polar spline space
    if form_rank == 0
        tp_space = TensorProductSpace((space_p, space_r))
    elseif form_rank == 2
        tp_space = TensorProductSpace((dspace_p, dspace_r))
    elseif form_rank == 1
        tp_space = (TensorProductSpace((dspace_p, space_r)), TensorProductSpace((space_p, dspace_r)))
    end

    # convert to tuple if form_rank is not 1 for uniformity
    if form_rank != 1
        tp_space = (tp_space,)
    end

    # allocate space for polar spline space(s)
    polar_splines = Vector{UnstructuredSpace}(undef, length(E))
    for component_idx ∈ eachindex(E)
        # number of elements
        num_elements = get_num_elements(tp_space[component_idx])

        # allocate space for extraction coefficients and basis indices
        extraction_coefficients = Vector{Matrix{Float64}}(undef, num_elements)
        basis_indices = Vector{Vector{Int}}(undef, num_elements)

        # Convert global extraction matrix to element local extractions
        # (here, the matrix is transposed so that [tp_spline_space] * [extraction] = [Polar-splines])
        count = 0
        for i = 1:num_elements
            _, cols_ij = get_extraction(tp_space[component_idx], i)
            eij = SparseArrays.findnz(E[component_idx][:, cols_ij])
            # Unique indices for non-zero rows and columns
            basis_indices[count+1] = unique(eij[1])
            # Matrix of coefficients
            extraction_coefficients[count+1] = Array(E[component_idx][basis_indices[count+1], cols_ij])'
            count += 1
        end

        # number of polar dofs
        space_dim = size(E[component_idx], 1)
        
        # Polar spline extraction operator
        extraction_op = ExtractionOperator(extraction_coefficients, basis_indices, num_elements, space_dim)

        # unstructured spline config
        patch_neighbours = [-1
        -1]
        # number of elements per patch
        patch_nels = [0; get_num_elements(tp_space[component_idx])]
        # assemble patch config in a dictionary
        us_config = Dict("patch_neighbours" => patch_neighbours, "patch_nels" => patch_nels)

        # dof partitioning for the tensor product space
        dof_partition_tp = get_dof_partition(tp_space[component_idx])
        n_partn = length(dof_partition_tp[1])
        dof_partition = Vector{Vector{Vector{Int}}}(undef,1)
        dof_partition[1] = Vector{Vector{Int}}(undef, n_partn)
        if singularity_type == 2
            for i ∈ 1:n_partn
                dof_partition[1][i] = []
            end
        else
            for i ∈ 1:n_partn
                if length(dof_partition_tp[1][i]) == 0
                    dof_partition[1][i] = []
                else
                    # TODO!!
                    dof_partition[1][i] = []
                end
            end
        end

        # build and store polar spline space
        polar_splines[component_idx] = UnstructuredSpace((tp_space[component_idx],), extraction_op, dof_partition, us_config, Dict("regularity" => 1, "control_triangle" => control_triangle))
    end

    return Tuple(polar_splines), E
end

"""
    build_standard_degenerate_control_points(n_p::Int, n_r::Int, R::Float64)

Generate degenerate control points for a given number of poloidal and radial divisions.

# Arguments
- n_p::Int: Number of poloidal divisions.
- n_r::Int: Number of radial divisions.
- R::Float64: Radius of the polar domain.

# Returns
- degenerate_control_points::Matrix{Float64}: The degenerate control points.
- radii::Vector{Float64}: The radii values.
- theta::Vector{Float64}: The theta values.
"""
function build_standard_degenerate_control_points(n_p::Int, n_r::Int, R::Float64)
    radii = LinRange(0.0, R, n_r)
    theta = 2*pi .+ (1 .- 2 .* (1:n_p)) .* pi / n_p
    degenerate_control_points = zeros(Float64, n_p, n_r, 2)
    # the control points that are identical to the tensor-product control points
    for idx ∈ Iterators.product(1:n_p, 1:n_r)
        degenerate_control_points[idx[1],idx[2],1] = radii[idx[2]]*cos(theta[idx[1]])
        degenerate_control_points[idx[1],idx[2],2] = radii[idx[2]]*sin(theta[idx[1]])
    end
    
    return degenerate_control_points, radii, theta
end

import LinearAlgebra, SparseArrays

"""
    build_polar_extraction_operators(polar_config::Int, degenerate_control_points::NTuple{2,Matrix{Float64}}, num_basis_p::Int, num_basis_r::Int)

build the extraction operators for the polar spline space. The extraction operators are used to extract the polar spline basis functions from the tensor product space.

# Arguments
- `polar_config::Int`: The type of polar singularity.
- `degenerate_control_points::NTuple{2,Matrix{Float64}}`: The degenerate control points.
- `num_basis_p::Int`: Number of poloidal basis functions.

# Returns
- `E0::SparseMatrixCSC{Float64,Int}`: The zero form extraction operator.
- `E1::Tuple{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Float64,Int}}`: The one form extraction operators.
- `E2::SparseMatrixCSC{Float64,Int}`: The two form extraction operator.
"""
function build_polar_extraction_operators(polar_config::Int, degenerate_control_points::NTuple{2,Matrix{Float64}}, num_basis_p::Int, num_basis_r::Int; form_rank::Union{Nothing,Int}=nothing)

    @assert (polar_config == 1) || (polar_config == 2) # type of polar singularity
    @assert all(size.(degenerate_control_points, 1) .== num_basis_p) # two rows of control points
    @assert all(size.(degenerate_control_points, 2) .== 2) # two dimensional control points

    ## Basic quantities
    # build control triangle that contains all degenerate control points
    trix, triy = build_control_triangle_(degenerate_control_points)
    # compute barycentric coordinates of the degenerate control points w.r.t. the control triangle
    baryc = barycentric_coordinates_(degenerate_control_points, trix, triy)
    
    # number of degrees of freedom for all polar spline forms
    num_0_forms = num_basis_p * num_basis_r - polar_config * (2 * num_basis_p - 3)
    num_1_forms = num_basis_p * num_basis_r + num_basis_p * (num_basis_r - 1) - polar_config * (3 * num_basis_p - 2)
    num_2_forms = num_basis_p * (num_basis_r - 1) - polar_config * num_basis_p
    
    build_forms = [false, false, false]
    if isnothing(form_rank)
        build_forms .= true
    else
        build_forms[form_rank+1] = true
    end

    ## Zero forms
    E0_1 = SparseArrays.sparse(hcat(baryc[1]', baryc[2]'))
    if polar_config == 2
        E0_2 = reverse(E0_1, dims=2)
    else
        E0_2 = SparseArrays.spdiagm(ones(3))
    end
    if build_forms[1]
        E0 = SparseArrays.blockdiag(E0_1, SparseArrays.spdiagm(ones(max(0,num_0_forms - 6))), E0_2)
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E0)
    else
        E0 = nothing
    end
    
    ## One forms
    
    if build_forms[2]
        # space for one form extraction data
        E1_v_r = zeros(Int, num_1_forms * 4)
        E1_v_c = zeros(Int, num_1_forms * 4)
        E1_v_v = zeros(Float64, num_1_forms * 4)
        E1_h_r = zeros(Int, num_1_forms * 4)
        E1_h_c = zeros(Int, num_1_forms * 4)
        E1_h_v = zeros(Float64, num_1_forms * 4)
        ind_v = 0
        ind_h = 0
        
        # near the bottom pole
        baryc = hcat(reshape(E0_1[:, 1:num_basis_p]', num_basis_p, 1, 3), reshape(E0_1[:, num_basis_p .+ (1:num_basis_p)]', num_basis_p, 1, 3))
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
        
        # intermediate vertical edges
        k = 0:(num_basis_p - 1)
        l = 2:(num_basis_r - polar_config)
        for idx ∈ Iterators.product(k, l)
            r = idx[1] + (2 * idx[2] - 4) * num_basis_p + 2
            c = idx[1] + (idx[2]-1) * num_basis_p
            E1_v_r[ind_v + 1] = r + 1
            E1_v_c[ind_v + 1] = c + 1
            E1_v_v[ind_v + 1] = 1.0
            ind_v += 1
        end

        # intermediate horizontal edges
        k = 0:(num_basis_p - 1)
        l = 2:(num_basis_r + 1 - 2 * polar_config)
        for idx ∈ Iterators.product(k, l)
            r = idx[1] + (2 * idx[2] - 3) * num_basis_p + 2
            c = idx[1] + idx[2] * num_basis_p
            E1_h_r[ind_h + 1] = r + 1
            E1_h_c[ind_h + 1] = c + 1
            E1_h_v[ind_h + 1] = 1.0
            ind_h += 1
        end
        
        # near top pole
        if polar_config == 2
            baryc = hcat(reshape(E0_2[:, 1:num_basis_p]', num_basis_p, 1, 3), reshape(E0_2[:, num_basis_p .+ (1:num_basis_p)]', num_basis_p, 1, 3))
            for k in 0:(num_basis_p - 1)
                kp = mod(k + 1, num_basis_p)
                l = num_basis_r - 2
                r = [num_1_forms - 2, num_1_forms - 1]
                c = k + l * num_basis_p

                v = baryc[k + 1, l - num_basis_r + 4, 2:3] .- baryc[k + 1, l - num_basis_r + 3, 2:3]
                E1_v_r[ind_v .+ (1:length(r))] = r .+ 1
                E1_v_c[ind_v .+ (1:length(r))] .= c + 1
                E1_v_v[ind_v .+ (1:length(r))] = v
                ind_v += length(r)

                v = baryc[kp + 1, l - num_basis_r + 3, 2:3] .- baryc[k + 1, l - num_basis_r + 3, 2:3]
                E1_h_r[ind_h .+ (1:length(r))] = r .+ 1
                E1_h_c[ind_h .+ (1:length(r))] .= c + 1
                E1_h_v[ind_h .+ (1:length(r))] = v        
                ind_h += length(r)
            end
        end
        
        # assembly
        E1_v_r = E1_v_r[1:ind_v]
        E1_v_c = E1_v_c[1:ind_v]
        E1_v_v = E1_v_v[1:ind_v]
        E1_v = sparse(E1_v_r, E1_v_c, E1_v_v, num_1_forms, num_basis_p * (num_basis_r - 1))
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E1_v)
        
        E1_h_r = E1_h_r[1:ind_h]
        E1_h_c = E1_h_c[1:ind_h]
        E1_h_v = E1_h_v[1:ind_h]
        E1_h = sparse(E1_h_r, E1_h_c, E1_h_v, num_1_forms, num_basis_p * num_basis_r)
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E1_h)
    else
        E1_h = nothing
        E1_v = nothing
    end
    
    ## Two forms

    if build_forms[3]
        # space for two form extraction data
        E2_r = zeros(Int, num_2_forms)
        E2_c = zeros(Int, num_2_forms)
        E2_v = zeros(Float64, num_2_forms)
        ind = 0

        # data for extraction
        k = 0:(num_basis_p - 1)
        l = 1:(num_basis_r - 1 - polar_config)
        for idx ∈ Iterators.product(k, l)
            r = idx[1] + (idx[2] - 1) * num_basis_p
            c = idx[1] + idx[2] * num_basis_p
            E2_r[ind + 1] = r + 1
            E2_c[ind + 1] = c + 1
            E2_v[ind + 1] = 1.0
            ind += 1
        end
        # assembly
        E2 = sparse(E2_r, E2_c, E2_v, num_2_forms, num_basis_p * (num_basis_r - 1))
        # Remove small values obtained as a result of round-off errors
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E2)

    else
        E2 = nothing
    end
    
    return ((E0,), (E1_h, E1_v), (E2,)), (trix, triy)
end

function build_control_triangle_(points::NTuple{2,Matrix{Float64}})
    rad = sqrt.(sum((points[2] .- points[1]).^2, dims=2))
    tau = 2 * maximum(rad)
    trix = [tau, -tau / 2, -tau / 2]
    triy = [0, -sqrt(3) * tau / 2, sqrt(3) * tau / 2]
    return trix, triy
end

function barycentric_coordinates_(points, trix, triy)
    L = hcat(trix, triy, ones(3))
    return (hcat(zeros(size(points[1])), ones(size(points[1], 1))) / L, hcat(points[2] .- points[1], ones(size(points[1], 1))) / L)
end