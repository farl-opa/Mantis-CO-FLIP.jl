@doc raw"""
    get_dorfler_marking(element_errors::Vector{Float64}, dorfler_parameter::Float64) 

Computes the indices of elements with at least `dorfler_parameter*100`% of the highest error in `element_errors`.

# Arguments
- `element_errors::Vector{Float64}`: element-wise errors.
- `dorfler_parameter::Float64`: dorfler parameter determing how many elements are selected.

# Returns
- `::Vector{Int}`: indices of elements with at least `dorfler_parameter*100`% of the highest error.
"""
function get_dorfler_marking(element_errors::Vector{Float64}, dorfler_parameter::Float64) 
    0.0 <= dorfler_parameter < 1.0 || throw(ArgumentError("Dorfler parameter should be between 0 and 1. The given value was $dorfler_parameter."))
    
    max_error = maximum(element_errors)
    
    return findall(error -> error > (1.0-dorfler_parameter)*max_error, element_errors)
end

@doc raw"""
    get_marked_element_padding(hier_space::HierarchicalFiniteElementSpace{n, S, T}, marked_elements_per_level::Vector{Vector{Int}}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns all the elements in the support of basis functions supported on `marked_elements_per_level`.

# Arguments

- `hier_space::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical finite element space.
- `marked_elements_per_level::Vector{Vector{Int}}`: marked elements, separated by level.

# Returns

- `element_padding::Vector{Vector{Int}}`: padding of marked elements.
"""
function get_marked_element_padding(hier_space::HierarchicalFiniteElementSpace{n, S, T}, marked_elements_per_level::Vector{Vector{Int}}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    num_levels = get_num_levels(hier_space)
    get_basis_indices_from_extraction(space, element) = get_extraction(space, element)[2]

    element_padding = [Int[] for _ ∈ 1:num_levels]
    level_padding = Int[]

    for level ∈ 1:num_levels
        if marked_elements_per_level[level] == Int[]
            continue
        end
        basis_in_marked_elements = reduce(union, get_basis_indices_from_extraction.(Ref(hier_space.spaces[level]), marked_elements_per_level[level]))
        
        level_padding = union(get_support.(Ref(hier_space.spaces[level]), basis_in_marked_elements)...)
        append!(element_padding[level], level_padding)
    end

    return element_padding
end