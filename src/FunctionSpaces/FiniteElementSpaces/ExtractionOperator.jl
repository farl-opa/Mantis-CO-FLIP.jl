
"""
    struct ExtractionOperator

Structure to store extraction operators and coefficients.
"""
struct ExtractionOperator
    extraction_coefficients::Vector{Array{Float64}}
    basis_indices::Vector{Vector{Int}}
    num_elements::Int
    space_dim::Int
end

# Getters for extraction operators
function get_dim(extraction_op::ExtractionOperator)
    return extraction_op.space_dim
end

function get_extraction(extraction_op::ExtractionOperator, element_id::Int)
    return @views extraction_op.extraction_coefficients[element_id], extraction_op.basis_indices[element_id]
end

function get_num_elements(extraction_op::ExtractionOperator)
    return extraction_op.num_elements
end