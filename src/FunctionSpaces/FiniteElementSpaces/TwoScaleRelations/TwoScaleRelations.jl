"""
    struct TwoScaleOperator{manifold_dim, S} <: AbstractTwoScaleOperator{manifold_dim}

A structure representing a two-scale operator that defines the relationships between coarse
and fine finite element spaces.

# Fields
- `coarse_space::S`: The coarse finite element space.
- `fine_space::S`: The fine finite element space.
- `global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}`: The global subdivision
    matrix. The size of this matrix is `(num_fine_basis, num_coarse_basis)` where
    `num_fine_basis` is the dimension of `fine_space` and `num_coarse_basis` the dimension
    of `coarse_space`.
- `coarse_to_fine_elements::Vector{Vector{Int}}`: A vector of vectors containing the child
    element IDs for each coarse element.
- `fine_to_coarse_elements::Vector{Int}`: A vector containing the parent element ID for
    each fine element.
- `coarse_to_fine_functions::Vector{Vector{Int}}`: A vector of vectors containing the child
    basis function IDs for each coarse basis function.
- `fine_to_coarse_functions::Vector{Vector{Int}}`: A vector of vectors containing the
    parent basis function IDs for each fine basis function.
"""
struct TwoScaleOperator{manifold_dim, S} <: AbstractTwoScaleOperator{manifold_dim}
    coarse_space::S
    fine_space::S
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    coarse_to_fine_elements::Vector{Vector{Int}}
    fine_to_coarse_elements::Vector{Int}
    coarse_to_fine_functions::Vector{Vector{Int}}
    fine_to_coarse_functions::Vector{Vector{Int}}

    function TwoScaleOperator(
        coarse_space::AbstractFESpace{manifold_dim, 1},
        fine_space::AbstractFESpace{manifold_dim, 1},
        global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int},
        coarse_to_fine_elements::Vector{Vector{Int}},
        fine_to_coarse_elements::Vector{Int},
    ) where {manifold_dim}
        if typeof(coarse_space) != typeof(fine_space)
            throw(ArgumentError("The coarse and fine spaces must be of the same type."))
        end

        dims = size(global_subdiv_matrix)
        coarse_to_fine_functions = Vector{Vector{Int}}(undef, dims[2])
        fine_to_coarse_functions = Vector{Vector{Int}}(undef, dims[1])
        gm_data = SparseArrays.findnz(global_subdiv_matrix)
        transpose_matrix = SparseArrays.sparse(gm_data[2], gm_data[1], gm_data[3])
        for i in 1:dims[2]
            coarse_to_fine_functions[i] = global_subdiv_matrix.rowval[SparseArrays.nzrange(
                global_subdiv_matrix, i
            )]
        end

        for i in 1:dims[1]
            fine_to_coarse_functions[i] = transpose_matrix.rowval[SparseArrays.nzrange(
                transpose_matrix, i
            )]
        end

        return new{manifold_dim, typeof(coarse_space)}(
            coarse_space,
            fine_space,
            global_subdiv_matrix,
            coarse_to_fine_elements,
            fine_to_coarse_elements,
            coarse_to_fine_functions,
            fine_to_coarse_functions,
        )
    end
end

"""
    get_element_children(twoscale_operator::TwoScaleOperator, el_id::Int)

Retrieve and return the child element IDs for a given element ID within a two-scale
operator.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator that defines the
    parent-child relationships between elements.
- `el_id::Int`: The identifier of the element whose children are to be retrieved.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child elements.
"""
function get_element_children(twoscale_operator::TwoScaleOperator, el_id::Int)
    return twoscale_operator.coarse_to_fine_elements[el_id]
end

"""
    get_basis_children(twoscale_operator::TwoScaleOperator, basis_id::Int)

Retrieve and return the child basis function IDs for a given basis function ID within a
two-scale operator.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator that defines the
    parent-child relationships between basis functions.
- `basis_id::Int`: The identifier of the basis function whose children are to be retrieved.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child basis functions.
"""
function get_basis_children(twoscale_operator::TwoScaleOperator, basis_id::Int)
    return twoscale_operator.coarse_to_fine_functions[basis_id]
end

"""
    get_element_parent(twoscale_operator::TwoScaleOperator, el_id::Int)

Retrieve and return the parent element ID for a given element ID within a two-scale
operator.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator that defines the
    parent-child relationships between elements.
- `el_id::Int`: The identifier of the element whose parent is to be retrieved.

# Returns
- `::Int`: The identifier of the parent element.
"""
function get_element_parent(twoscale_operator::TwoScaleOperator, el_id::Int)
    return twoscale_operator.fine_to_coarse_elements[el_id]
end

"""
    get_basis_parents(twoscale_operator::TwoScaleOperator, basis_id::Int)

Retrieve and return the parent basis functions for a given basis function ID within a
two-scale operator.

# Arguments
- `twoscale_operator::TwoScaleOperator`: The two-scale operator that defines the
    parent-child relationships between basis functions.
- `basis_id::Int`: The identifier of the basis function whose parent is to be retrieved.

# Returns
- `::Vector{Int}`: The identifier of the parent basis functions.
"""
function get_basis_parents(twoscale_operator::TwoScaleOperator, basis_id::Int)
    return twoscale_operator.fine_to_coarse_functions[basis_id]
end
