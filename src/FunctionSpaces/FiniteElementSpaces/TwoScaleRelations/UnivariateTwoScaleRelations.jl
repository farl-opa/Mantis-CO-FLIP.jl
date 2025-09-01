"""
    get_element_children(coarse_element_id::Int, nsubdivisions::Int) -> Vector{Int}

Compute and return the child element IDs for a given coarse element ID based on the number
    of subdivisions.

# Arguments
- `coarse_element_id::Int`: The identifier of the coarse element.
- `nsubdivisions::Int`: The number of subdivisions for the coarse element.

# Returns
- `::Vector{Int}`: A vector containing the identifiers of the child elements.
"""
function get_element_children(parent_element_id::Int, num_subdivisions::Int)
    return [i + ((parent_element_id - 1) * num_subdivisions) for i in 1:num_subdivisions]
end

function get_element_parent(child_element_id::Int, num_subdivisions::Int)
    return floor(Int, (child_element_id - 1) / num_subdivisions + 1)
end
