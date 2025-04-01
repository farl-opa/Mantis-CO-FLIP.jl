##################################################################
# GENERAL METHODS
#####################################################################

"""
    get_num_patches(mp_space::AbstractMultiPatchFESpace)

Get the number of patches in the multi-patch space.

# Arguments
- `mp_space::AbstractMultiPatchFESpace`: The multi-patch space.

# Returns
- `::Int`: Number of patches in the multi-patch space.
"""
function get_num_patches(
    _::AbstractMultiPatchFESpace{manifold_dim, num_components, num_patches}
) where {manifold_dim, num_components, num_patches}
    return num_patches
end

"""
    get_patch_id(mp_space::AbstractMultiPatchFESpace, element_id::Int)

Get the ID of the patch to which the specified element belongs.

# Arguments
- `mp_space::AbstractMultiPatchFESpace`: The multi-patch space.
- `element_id::Int`: The global element ID.

# Returns
- `::Int`: ID of the patch to which the element belongs.
"""
function get_patch_id(mp_space::AbstractMultiPatchFESpace, element_id::Int)
    return findlast(cumsum(_get_num_elements_per_patch(mp_space)) .<= element_id)
end

function _get_num_elements_per_patch(mp_space::AbstractMultiPatchFESpace)
    return mp_space.num_elements_per_patch
end

"""
    get_patch_and_element_id(mp_space::AbstractMultiPatchFESpace, element_id::Int)

Get the constituent patch index and local element index for the specified global element index.

# Arguments
- `mp_space::AbstractMultiPatchFESpace`: The multi-patch space.
- `element_id::Int`: The global element ID.

# Returns
- `::Tuple{Int,Int}`: Tuple of constituent patch ID and local element ID.
"""
function get_patch_and_element_id(mp_space::AbstractMultiPatchFESpace, element_id::Int)
    patch_id = get_patch_id(mp_space, element_id)
    patch_element_id = element_id - sum(_get_num_elements_per_patch(mp_space)[1:patch_id-1])
    return patch_id, patch_element_id
end

"""
    get_global_element_id(mp_space::AbstractMultiPatchFESpace, patch_id::Int, patch_element_id::Int

Get the global element ID for the specified constituent patch ID and local element ID.

# Arguments
- `mp_space::AbstractMultiPatchFESpace`: The multi-patch space.
- `patch_id::Int`: The constituent patch ID.
- `patch_element_id::Int`: The local element ID.

# Returns
- `::Int`: The global element ID.
"""
function get_global_element_id(
    mp_space::AbstractMultiPatchFESpace, patch_id::Int, patch_element_id::Int
)
    return sum(_get_num_elements_per_patch(mp_space)[1:patch_id-1]) + patch_element_id
end

"""
    get_polynomial_degree(mp_space::AbstractMultiPatchFESpace, element_id::Int)

Get the polynomial degree of the specified element in the multi-patch space.

# Arguments
- `mp_space::AbstractMultiPatchFESpace`: The multi-patch space.
- `element_id::Int`: The global element ID.

# Returns
- `::Int`: The polynomial degree of the specified element.
"""
function get_polynomial_degree(mp_space::AbstractMultiPatchFESpace, element_id::Int)
    # Find the space ID and local element ID
    patch_id, patch_element_id = get_patch_and_element_id(mp_space, element_id)

    # Get the polynomial degree from the corresponding function space
    return get_polynomial_degree(mp_space.patch_spaces[patch_id], patch_element_id)
end

function get_element_dimensions(mp_space::AbstractMultiPatchFESpace, element_id::Int)
    # Find the space ID and local element ID
    patch_id, patch_element_id = get_patch_and_element_id(mp_space, element_id)
    return get_element_dimensions(mp_space.patch_spaces[patch_id], patch_element_id)
end

function get_element_vertices(mp_space::AbstractMultiPatchFESpace, element_id::Int)
    patch_id, patch_element_id = get_patch_and_element_id(mp_space, element_id)
    return get_element_vertices(mp_space.patch_spaces[patch_id], patch_element_id)
end

function get_max_local_dim(mp_space::AbstractMultiPatchFESpace)
    return maximum(get_max_local_dim.(mp_space.patch_spaces))
end

function get_local_basis(
    mp_space::AbstractMultiPatchFESpace{manifold_dim, num_spaces},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int,
) where {manifold_dim, num_spaces}
    # We need space ID and local element ID to know which function space to evaluate.
    patch_id, patch_element_id = get_patch_and_element_id(mp_space, element_id)

    return evaluate(mp_space.patch_spaces[patch_id], patch_element_id, xi, nderivatives)[1]
end

"""
    assemble_global_extraction_matrix(mp_space::AbstractMultiPatchFESpace)

Loops over all elements and assembles the global extraction matrix for the multi-patch
space. The extraction matrix is a sparse matrix that maps the local basis functions to the
global basis functions.

# Arguments
- `mp_space::AbstractMultiPatchFESpace`: The multi-patch space.

# Returns
- `::Array{Float64,2}`: Global extraction matrix.
"""
function assemble_global_extraction_matrix(mp_space::AbstractMultiPatchFESpace)
    # Initialize the global extraction matrix
    num_global_basis = get_num_basis(mp_space)
    num_local_basis = get_num_basis.(mp_space.patch_spaces)
    local_basis_offset = cumsum([0; num_local_basis...])
    global_extraction_matrix = zeros(Float64, local_basis_offset[end], num_global_basis)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # Loop over all elements
    for element_id in 1:get_num_elements(mp_space)
        # Get the extraction coefficients and global basis indices
        extraction_coefficients, global_basis_indices = get_extraction(mp_space, element_id)

        # Get the local space ID and local element ID
        patch_id, patch_element_id = get_patch_and_element_id(mp_space, element_id)

        # Get the (offsetted) local basis indices
        _, local_basis_indices = get_extraction(
            mp_space.patch_spaces[patch_id], patch_element_id
        )

        # store the entries
        for (r, c) in Iterators.product(local_basis_indices, global_basis_indices)
            push!(rows, r + local_basis_offset[patch_id])
            push!(cols, c)
            push!(vals, extraction_coefficients[r, c])
        end
    end

    return SparseArrays.sparse(rows, cols, vals, local_basis_offset[end], num_global_basis)
end

##################################################################
# CONCRETE IMPLEMENTATIONS
#####################################################################

include("GTBSplines.jl")
include("PolarSplines.jl")
