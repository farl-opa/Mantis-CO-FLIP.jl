
"""
    ExtractionOperator

Stores extraction coefficients and basis indices for a function space.

# Fields
- `extraction_coefficients::Vector{Matrix{Float64}}`: A vector of extraction coefficient
    matrices, where each matrix corresponds to an element.
- `basis_indices::Vector{Vector{Int}}`: A vector of basis index vectors, where each vector
    corresponds to the basis indices for an element.
- `num_elements::Int`: The number of elements.
- `space_dim::Int`: The dimension of the function space.
"""
struct ExtractionOperator
    extraction_coefficients::Vector{Matrix{Float64}}
    basis_indices::Vector{Vector{Int}}
    num_elements::Int
    space_dim::Int
end

"""
    get_extraction_coefficients(extraction_op::ExtractionOperator, element_id::Int)

Returns the extraction coefficients for the given element.

# Arguments
- `extraction_op::ExtractionOperator`: The `ExtractionOperator` object.
- `element_id::Int`: The identifier of the element.

# Returns
- `::Matrix{Float64}`: The extraction coefficients on the requested element..
"""
function get_extraction_coefficients(extraction_op::ExtractionOperator, element_id::Int)
    return extraction_op.extraction_coefficients[element_id]
end

function get_basis_indices(extraction_op::ExtractionOperator, element_id::Int)
    return extraction_op.basis_indices[element_id]
end

"""
    get_num_basis(extraction_op::ExtractionOperator)

Returns the dimension of the function space associated with the `extraction_op`.

# Arguments
- `extraction_op::ExtractionOperator`: The `ExtractionOperator` object.

# Returns
- `::Int`: The dimension of the function space.
"""
function get_num_basis(extraction_op::ExtractionOperator)
    return extraction_op.space_dim
end

"""
    get_num_basis(extraction_op::ExtractionOperator, element_id::Int)

Returns the number of basis functions supported on a specific element.

# Arguments
- `extraction_op::ExtractionOperator`: The `ExtractionOperator` object.
- `element_id::Int`: The index of the element.

# Returns
- `::Vector{Int}`: The number of basis functions for the specified element.
"""
function get_num_basis(extraction_op::ExtractionOperator, element_id::Int)
    return length(get_basis_indices(extraction_op, element_id))
end

function get_extraction(extraction_op::ExtractionOperator, element_id::Int)
    return (
        get_extraction_coefficients(extraction_op, element_id),
        get_basis_indices(extraction_op, element_id),
    )
end

get_num_elements(extraction_op::ExtractionOperator) = extraction_op.num_elements
