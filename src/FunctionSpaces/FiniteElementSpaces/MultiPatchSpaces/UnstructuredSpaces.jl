
"""
    AbstractMultiPatchFESpace{manifold_dim, num_patches} <: AbstractFESpace{manifold_dim, 1, num_patches}

An `manifold_dim`-variate multi-patch space with `num_patches` patches, representing an
unstructured finite element space.

# Fields
- `function_spaces::NTuple{num_patches, AbstractFESpace{manifold_dim}}`: Collection of `num_patches` (uni or
    multivariate) function spaces.
- `extraction_op::ExtractionOperator`: Extraction operator that specifies how to combine
    functions from the `num_patches` spaces into functions for the unstructured space.
- `dof_partition::Vector{Vector{Int}}`: Partition of degrees of freedom.
- `us_config::Dict`: Dictionary that stores helper functionality (e.g., connectivity) for
    the unstructured space.
- `data::Dict`: Any auxiliary data that the user wants to store for this unstructured space.
"""
struct AbstractMultiPatchFESpace{manifold_dim, num_patches} <:
    AbstractFESpace{manifold_dim, 1, num_patches}

    function_spaces::NTuple{num_patches, AbstractFESpace{manifold_dim}}
    extraction_op::ExtractionOperator
    dof_partition::Vector{Vector{Vector{Int}}}
    us_config::Dict
    data::Dict

    function AbstractMultiPatchFESpace(
        function_spaces::NTuple{num_patches, AbstractFESpace{manifold_dim, 1}},
        extraction_op::ExtractionOperator,
        dof_partition::Vector{Vector{Vector{Int}}},
        us_config::Dict,
        data::Dict,
    ) where {manifold_dim, num_patches}
        # Initialize with empty dof partitioning
        return new{manifold_dim, num_patches}(
            function_spaces, extraction_op, dof_partition, us_config, data
        )
    end

    function AbstractMultiPatchFESpace(
        function_spaces::NTuple{num_patches, AbstractFESpace{1}},
        extraction_op::ExtractionOperator,
        data::Dict,
    ) where {num_patches}
        return AbstractMultiPatchFESpace(function_spaces, extraction_op, 1, 1, data)
    end

    function AbstractMultiPatchFESpace(
        function_spaces::NTuple{num_patches, AbstractFESpace{1, 1}},
        extraction_op::ExtractionOperator,
        n_dofs_left::Int,
        n_dofs_right::Int,
        data::Dict,
    ) where {num_patches}
        # Build 1D topology
        patch_neighbours = [
            -1 (1:(num_patches - 1))...
            (2:num_patches)... -1
        ]
        # Number of elements per patch
        patch_nels = [0; cumsum([get_num_elements(function_spaces[i]) for i in 1:num_patches])]

        # Assemble patch config in a dictionary
        us_config = Dict("patch_neighbours" => patch_neighbours, "patch_nels" => patch_nels)

        # Allocate memory for degree of freedom partitioning: note, even if the 1D space is
        # implemented as a multipatch object, the dof partition treats it as a single patch!
        dof_partition = Vector{Vector{Vector{Int}}}(undef, 1)
        dof_partition[1] = Vector{Vector{Int}}(undef, 3)
        # First, store the left dofs ...
        dof_partition[1][1] = collect(1:n_dofs_left)
        # ... then the interior dofs ...
        dof_partition[1][2] = collect(
            (n_dofs_left + 1):(get_num_basis(extraction_op) - n_dofs_right)
        )
        # ... and then finally the right dofs.
        dof_partition[1][3] = collect(
            (get_num_basis(extraction_op) - n_dofs_right + 1):get_num_basis(extraction_op)
        )

        return new{1, num_patches}(function_spaces, extraction_op, dof_partition, us_config, data)
    end
end

function get_element_lengths(us_space::AbstractMultiPatchFESpace, element_id::Int)
    # Find the space ID and local element ID
    space_id, space_element_id = get_patch_and_element_id(us_space, element_id)

    return get_element_lengths(us_space.function_spaces[space_id], space_element_id)
end

"""
    get_polynomial_degree(us_space::AbstractMultiPatchFESpace, element_id::Int)

Get the polynomial degree of the specified element in the unstructured space.

# Arguments
- `us_space::AbstractMultiPatchFESpace`: The unstructured space.
- `element_id::Int`: The global element ID.

# Returns
- `::Int`: The polynomial degree of the specified element.
"""
function get_polynomial_degree(us_space::AbstractMultiPatchFESpace, element_id::Int)
    # Find the space ID and local element ID
    space_id, space_element_id = get_patch_and_element_id(us_space, element_id)

    # Get the polynomial degree from the corresponding function space
    return get_polynomial_degree(us_space.function_spaces[space_id], space_element_id)
end

"""
    get_patch_id(us_space::AbstractMultiPatchFESpace, element_id::Int)

Get the ID of the constituent space to which the specified element belongs.

# Arguments
- `us_space::AbstractMultiPatchFESpace`: The unstructured space.
- `element_id::Int`: The global element ID.

# Returns
- `::Int`: ID of the constituent space to which the element belongs.
"""
function get_patch_id(us_space::AbstractMultiPatchFESpace, element_id::Int)
    return findlast(us_space.us_config["patch_nels"] .< element_id)
end

function get_max_local_dim(space::AbstractMultiPatchFESpace)
    return maximum(get_max_local_dim.(space.function_spaces))
end

function get_local_basis(
    space::AbstractMultiPatchFESpace{manifold_dim, num_patches},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int,
) where {manifold_dim, num_patches}
    # We need space ID and local element ID to know which function space to evaluate.
    space_id, space_element_id = get_patch_and_element_id(space, element_id)

    return evaluate(space.function_spaces[space_id], space_element_id, xi, nderivatives)[1]
end

"""
    get_patch_and_element_id(us_space::AbstractMultiPatchFESpace, element_id::Int)

Get the constituent space ID and local element ID for the specified global element ID.

# Arguments
- `us_space::AbstractMultiPatchFESpace`: The unstructured space.
- `element_id::Int`: The global element ID.

# Returns
- `::Tuple{Int,Int}`: Tuple of constituent space ID and local element ID.
"""
function get_patch_and_element_id(us_space::AbstractMultiPatchFESpace, element_id::Int)
    space_id = get_patch_id(us_space, element_id)
    space_element_id = element_id - us_space.us_config["patch_nels"][space_id]
    return space_id, space_element_id
end

"""
    get_global_element_id(us_space::AbstractMultiPatchFESpace, space_id::Int, space_element_id::Int)

Get the global element ID for the specified constituent space ID and local element ID.

# Arguments
- `us_space::AbstractMultiPatchFESpace`: The unstructured space.
- `space_id::Int`: The constituent space ID.
- `space_element_id::Int`: The local element ID.

# Returns
- `::Int`: The global element ID.
"""
function get_global_element_id(
    us_space::AbstractMultiPatchFESpace, space_id::Int, space_element_id::Int
)
    return us_space.us_config["patch_nels"][space_id] + space_element_id
end

"""
    assemble_global_extraction_matrix(us_space::AbstractMultiPatchFESpace)

Loops over all elements and assembles the global extraction matrix for the unstructured
space. The extraction matrix is a sparse matrix that maps the local basis functions to the
global basis functions.

# Arguments
- `us_space::AbstractMultiPatchFESpace`: The unstructured space.

# Returns
- `::Array{Float64,2}`: Global extraction matrix.
"""
function assemble_global_extraction_matrix(us_space::AbstractMultiPatchFESpace)
    # Initialize the global extraction matrix
    num_global_basis = get_num_basis(us_space)
    num_local_basis = get_num_basis.(us_space.function_spaces)
    local_basis_offset = cumsum([0; num_local_basis...])
    global_extraction_matrix = zeros(Float64, local_basis_offset[end], num_global_basis)

    # Loop over all elements
    for element_id in 1:get_num_elements(us_space)
        # Get the extraction coefficients and global basis indices
        extraction_coefficients, global_basis_indices = get_extraction(us_space, element_id)

        # Get the local space ID and local element ID
        space_id, space_element_id = get_patch_and_element_id(us_space, element_id)

        # Get the (offsetted) local basis indices
        _, local_basis_indices = get_extraction(
            us_space.function_spaces[space_id], space_element_id
        )

        # Assemble the global extraction matrix
        global_extraction_matrix[
            local_basis_indices .+ local_basis_offset[space_id], global_basis_indices
        ] = extraction_coefficients
    end

    return SparseArrays.sparse(global_extraction_matrix)
end

include("PolarSplines.jl")
