"""
    struct TwoScaleOperator{manifold_dim, S} <: AbstractTwoScaleOperator{manifold_dim}

A structure representing a two-scale operator that dechilds the relationships between parent
and child finite element spaces.

# Fields
- `parent_space::S`: The parent finite element space.
- `child_space::S`: The child finite element space.
- `global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}`: The global subdivision
    matrix. The size of this matrix is `(num_child_basis, num_parent_basis)` where
    `num_child_basis` is the dimension of `child_space` and `num_parent_basis` the dimension
    of `parent_space`.
- `parent_to_child_elements::Vector{Vector{Int}}`: A vector of vectors containing the child
    element IDs for each parent element.
- `child_to_parent_elements::Vector{Int}`: A vector containing the parent element ID for
    each child element.
- `parent_to_child_basis::Vector{Vector{Int}}`: A vector of vectors containing the child
    basis function IDs for each parent basis function.
- `child_to_parent_basis::Vector{Vector{Int}}`: A vector of vectors containing the
    parent basis function IDs for each child basis function.
"""
struct TwoScaleOperator{manifold_dim, num_components, num_patches, S} <:
       AbstractTwoScaleOperator{manifold_dim, num_components, num_patches}
    parent_space::S
    child_space::S
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    parent_to_child_elements::Vector{Vector{Int}}
    child_to_parent_elements::Vector{Int}
    parent_to_child_basis::Vector{Vector{Int}}
    child_to_parent_basis::Vector{Vector{Int}}

    function TwoScaleOperator(
        parent_space::S,
        child_space::S,
        global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int},
        parent_to_child_elements::Vector{Vector{Int}},
        child_to_parent_elements::Vector{Int},
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        S <: AbstractFESpace{manifold_dim, num_components, num_patches},
    }
        num_child_basis, num_parent_basis = size(global_subdiv_matrix)
        parent_to_child_basis = Vector{Vector{Int}}(undef, num_parent_basis)
        child_to_parent_basis = Vector{Vector{Int}}(undef, num_child_basis)
        gm_data = SparseArrays.findnz(global_subdiv_matrix)
        transpose_matrix = SparseArrays.sparse(gm_data[2], gm_data[1], gm_data[3])
        for i in 1:num_parent_basis
            parent_to_child_basis[i] = global_subdiv_matrix.rowval[SparseArrays.nzrange(
                global_subdiv_matrix, i
            )]
        end

        for i in 1:num_child_basis
            child_to_parent_basis[i] = transpose_matrix.rowval[SparseArrays.nzrange(
                transpose_matrix, i
            )]
        end

        return new{manifold_dim, num_components, num_patches, S}(
            parent_space,
            child_space,
            global_subdiv_matrix,
            parent_to_child_elements,
            child_to_parent_elements,
            parent_to_child_basis,
            child_to_parent_basis,
        )
    end
end

function get_parent_to_child_elements(operator::TwoScaleOperator)
    return operator.parent_to_child_elements
end

function get_child_to_parent_elements(operator::TwoScaleOperator)
    return operator.child_to_parent_elements
end

function get_parent_to_child_basis(operator::TwoScaleOperator)
    return operator.parent_to_child_basis
end

function get_child_to_parent_basis(operator::TwoScaleOperator)
    return operator.child_to_parent_basis
end

"""
    get_element_children(operator::TwoScaleOperator, element_id::Int)

Retrieve and return the child element IDs for a given element ID within a two-scale
operator.

# Arguments
- `operator::TwoScaleOperator`: The two-scale operator that dechilds the
    parent-child relationships between elements.
- `element_id::Int`: The identifier of the element whose children are to be retrieved.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child elements.
"""
function get_element_children(operator::TwoScaleOperator, element_id::Int)
    return get_parent_to_child_elements(operator)[element_id]
end

"""
    get_basis_children(operator::TwoScaleOperator, basis_id::Int)

Retrieve and return the child basis function IDs for a given basis function ID within a
two-scale operator.

# Arguments
- `operator::TwoScaleOperator`: The two-scale operator that dechilds the
    parent-child relationships between basis basis.
- `basis_id::Int`: The identifier of the basis function whose children are to be retrieved.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child basis basis.
"""
function get_basis_children(operator::TwoScaleOperator, basis_id::Int)
    return get_parent_to_child_basis(operator)[basis_id]
end

"""
    get_element_parent(operator::TwoScaleOperator, element_id::Int)

Retrieve and return the parent element ID for a given element ID within a two-scale
operator.

# Arguments
- `operator::TwoScaleOperator`: The two-scale operator that dechilds the
    parent-child relationships between elements.
- `element_id::Int`: The identifier of the element whose parent is to be retrieved.

# Returns
- `::Int`: The identifier of the parent element.
"""
function get_element_parent(operator::TwoScaleOperator, element_id::Int)
    return get_child_to_parent_elements(operator)[element_id]
end

"""
    get_basis_parents(operator::TwoScaleOperator, basis_id::Int)

Retrieve and return the parent basis basis for a given basis function ID within a
two-scale operator.

# Arguments
- `operator::TwoScaleOperator`: The two-scale operator that dechilds the
    parent-child relationships between basis basis.
- `basis_id::Int`: The identifier of the basis function whose parent is to be retrieved.

# Returns
- `::Vector{Int}`: The identifier of the parent basis basis.
"""
function get_basis_parents(operator::TwoScaleOperator, basis_id::Int)
    return get_child_to_parent_basis(operator)[basis_id]
end

"""
    get_global_subdiv_matrix(operator::TwoScaleOperator)

Retrieve and return the global subdivision matrix from a two-scale operator.

# Arguments
- `operator::TwoScaleOperator`: The two-scale operator containing the global subdivision
    matrix.

# Returns
- `::SparseArrays.SparseMatrixCSC{Float64, Int}`: The global subdivision
"""
function get_global_subdiv_matrix(operator::TwoScaleOperator)
    return operator.global_subdiv_matrix
end
