
function get_marked_elements_per_level(hspace::HierarchicalFiniteElementSpace{n}, marked_elements::Vector{Int}) where {n}
    L = get_num_levels(hspace)
    marked_elements_per_level = [Int[] for _ ∈ 1:L]
    
    # Separate the marked elements per level
    for el ∈ marked_elements
        el_level = get_active_level(hspace.active_elements, el)
        append!(marked_elements_per_level[el_level], hspace.active_elements.ids[el])
    end

    return marked_elements_per_level
end

function get_marked_basis_support(hspace::HierarchicalFiniteElementSpace{n}, level::Int, marked_basis) where {n}
    return Vector{Int}(unique!(vcat(get_support.((get_space(hspace, level),), marked_basis)...)))
end
