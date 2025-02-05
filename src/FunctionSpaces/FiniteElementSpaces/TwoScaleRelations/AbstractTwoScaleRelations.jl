"""
Functions and algorithms used for two scale relations.

"""

"""
    AbstractTwoScaleOperator

Supertype for all two scale relations.
"""
abstract type AbstractTwoScaleOperator{manifold_dim} end

# Basic getters for AbstractTwoScaleOperator

"""
    get_global_subdiv_matrix(twoscale_operator::AbstractTwoScaleOperator)

Retrieve and return the global subdivision matrix associated with a two-scale operator.

# Arguments
- `twoscale_operator::AbstractTwoScaleOperator`: The two-scale operator from which to retrieve the global subdivision matrix.

# Returns
- `::SparseArrays.SparseMatrixCSC{Float64, Int}`: The global subdivision matrix associated with the two-scale operator.
"""
get_global_subdiv_matrix(twoscale_operator::AbstractTwoScaleOperator) = twoscale_operator.global_subdiv_matrix

"""
    get_coarse_space(twoscale_operator::TwoScaleOperator)

Retrieve and return the coarse space associated with a two-scale operator.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator from which to retrieve the coarse space.

# Returns
- `::AbstractFiniteElementSpace`: The coarse space associated with the two-scale operator.
"""
get_coarse_space(twoscale_operator::AbstractTwoScaleOperator) =  twoscale_operator.coarse_space

"""
    get_fine_space(twoscale_operator::TwoScaleOperator)
Retrieve and return the fine space associated with a two-scale operator.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator from which to retrieve the fine space.

# Returns
- `::AbstractFiniteElementSpace`: The fine space associated with the two-scale operator.
"""
get_fine_space(twoscale_operator::AbstractTwoScaleOperator) =  twoscale_operator.fine_space

"""
    get_element_ancestor(two_scale_operators, child_element_id::Int, child_level::Int, num_ancestor_levels::Int)

Compute and return the ancestor element ID for a given child element ID within a specified number of ancestor levels.

# Arguments
- `two_scale_operators`: A collection of operators that define the parent-child relationships between elements at different levels.
- `child_element_id::Int`: The identifier of the child element.
- `child_level::Int`: The level of the child element.
- `num_ancestor_levels::Int`: The number of ancestor levels to traverse.

# Returns
- `ancestor_id::Int`: The identifier of the ancestor element.
"""
function get_element_ancestor(two_scale_operators::Vector{TS}, child_element_id::Int, child_level::Int, num_ancestor_levels::Int) where {TS <: AbstractTwoScaleOperator}
    ancestor_id = child_element_id
    for level ∈ child_level:-1:child_level-num_ancestor_levels+1
        ancestor_id = get_element_parent(two_scale_operators[level-1], ancestor_id)
    end

    return ancestor_id
end

"""
    get_finer_basis_coeffs(coarse_basis_coeffs::Vector{Float64}, twoscale_operator::AbstractTwoScaleOperator)

Perform a change of basis using a two-scale operator from a coarse basis and return the resulting fine basis coefficients.

# Arguments
- `coarse_basis_coeffs::Vector{Float64}`: A vector containing the coefficients of the coarse basis.
- `twoscale_operator::T`: The two-scale operator that defines the subdivision process, where `T` is a subtype of `AbstractTwoScaleOperator`.

# Returns
- `::Vector{Float64}`: A vector containing the coefficients of the fine basis after subdivision.
"""
function get_finer_basis_coeffs(coarse_basis_coeffs::Vector{Float64}, twoscale_operator::AbstractTwoScaleOperator)
    return twoscale_operator.global_subdiv_matrix * coarse_basis_coeffs
end

"""
    get_local_subdiv_matrix(twoscale_operator::AbstractTwoScaleOperator, coarse_element_id::Int, fine_element_id::Int)

Retrieve and return the local subdivision matrix for a given pair of coarse and fine elements within a two-scale operator.

# Arguments
- `twoscale_operator::AbstractTwoScaleOperator`: The two-scale operator that defines the subdivision process.
- `coarse_element_id::Int`: The identifier of the coarse element.
- `fine_element_id::Int`: The identifier of the fine element.

# Returns
- `::Matrix{Float64}`: The local subdivision matrix corresponding to the specified coarse and fine elements.
"""
function get_local_subdiv_matrix(twoscale_operator::AbstractTwoScaleOperator, coarse_element_id::Int, fine_element_id::Int)
    fine_basis_indices = get_basis_indices(twoscale_operator.fine_space, fine_element_id)
    coarse_basis_indices = get_basis_indices(twoscale_operator.coarse_space, coarse_element_id)

    return twoscale_operator.global_subdiv_matrix[fine_basis_indices, coarse_basis_indices]
end

# Includes for concrete two scale relations

include("TwoScaleRelations.jl")
include("UnivariateTwoScaleRelations.jl")
include("UnivariateBSplineTwoScaleRelations.jl")
include("TensorProductTwoScaleRelations.jl")
include("UnstructuredTwoScaleRelations.jl")
