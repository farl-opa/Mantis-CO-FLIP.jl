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
    get_marked_element_padding(hier_space::HierarchicalFiniteElementSpace, marked_elements_per_level::Vector{Vector{Int}})

Returns all the elements in the support of basis functions supported on `marked_elements_per_level`.

# Arguments

- `hier_space::HierarchicalFiniteElementSpace`: hierarchical finite element space.
- `marked_elements_per_level::Vector{Vector{Int}}`: marked elements, separated by level.

# Returns

- `element_padding::Vector{Vector{Int}}`: padding of marked elements.
"""
function compute_marked_elements_padding(hier_space::HierarchicalFiniteElementSpace, marked_elements_per_level::Vector{Vector{Int}})
    num_levels = get_num_levels(hier_space)
    get_basis_indices_from_extraction(space, element) = get_extraction(space, element)[2]

    element_padding = Vector{Vector{Int}}(undef, num_levels)

    for level ∈ 1:num_levels
        if marked_elements_per_level[level] == Int[]
            element_padding[level] = Int[]
            continue
        end

        basis_in_marked_elements = reduce(union, get_basis_indices_from_extraction.(Ref(hier_space.spaces[level]), marked_elements_per_level[level]))
        element_padding[level] = union(get_support.(Ref(hier_space.spaces[level]), basis_in_marked_elements)...)
    end

    return element_padding
end

function get_marked_elements_children(hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}, marked_elements_per_level::Vector{Vector{Int}}, new_operator::T) where {manifold_dim, S<:AbstractFESpace{manifold_dim, 1}, T<:AbstractTwoScaleOperator}

    num_levels = get_num_levels(hier_space)

    marked_children = Vector{Vector{Int}}(undef, num_levels)
    marked_children[1] = Int[]

    for level ∈ 1:num_levels
        if level < num_levels
            if marked_elements_per_level[level] == Int[]
                marked_children[level+1] = Int[]
            else
                marked_children[level+1] = get_element_children(get_twoscale_operator(hier_space, level),marked_elements_per_level[level])
            end
        elseif marked_elements_per_level[level] != Int[]
            push!(marked_children, get_element_children(new_operator,marked_elements_per_level[level]))
        end
    end

    return marked_children
end

function get_refinement_domains(hier_space::HierarchicalFiniteElementSpace, marked_elements::Vector{Int}, new_operator)
    element_ids_per_level = convert_element_vector_to_elements_per_level(hier_space, marked_elements)

    element_ids_per_level = compute_marked_elements_padding(hier_space, element_ids_per_level)

    refinement_domains = get_marked_elements_children(hier_space, element_ids_per_level, new_operator)

    return refinement_domains
end
