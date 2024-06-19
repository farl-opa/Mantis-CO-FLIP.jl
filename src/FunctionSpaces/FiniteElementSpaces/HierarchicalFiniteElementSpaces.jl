"""
This (sub-)module provides functionality related with hierarchical refinment.

The exported names are:
"""

"""
    struct HierarchicalActiveInfo

A structure that contains hierarchical information of active objects. `ids` is a collection of linear ids of 
active objects, whether elements or functions, and `levels` contains information about the level
of each active object by enconding the indexes of the last active objects, of the `ids` vector, from each level. I.e., if `levels = [0, n1, n2, ...]`, then `ids[1:n1]` will contain all active objects
from level 1, `ids[n1+1:n2]` all active objects from level 2, and so forth.

# Fiels
- `ids::Vector{Int}`: the index of the object in its corresponding structure.
- `levels::Vector{Int}`: level information of active objects.
"""
struct HierarchicalActiveInfo
    ids::Vector{Int}
    levels::Vector{Int} 
end

"""
    struct HierarchicalFiniteElementSpace{n} <: AbstractFiniteElementSpace{n}

A hierarchical space that is built from a nested hierarchy of `n`–variate function spaces. 

# Fields
- `spaces::Vector{AbstractFiniteElementSpace{n}} `: collection of `n`–variate 
function spaces.
- `two_scale_operators::Vector{TwoScaleOperator}`: collection of two–scale operators
relating each consequtive pair of function spaces.
- `active_elements::HierarchicalActiveInfo`: information about the active elements in each level.
- `active_functions::HierarchicalActiveInfo`: information about the active functions in each level.
"""
struct HierarchicalFiniteElementSpace{n} <: AbstractFiniteElementSpace{n}
    spaces::Vector{AbstractFiniteElementSpace{n}} 
    two_scale_operators::Vector{O} where {O<:AbstractTwoScaleOperator}
    active_elements::HierarchicalActiveInfo
    active_functions::HierarchicalActiveInfo
    multilevel_elements::SparseArrays.SparseVector{Int, Int}
    multilevel_extraction_coeffs::Vector{Matrix{Float64}}
    multilevel_basis_indices::Vector{Vector{Int}}

    function HierarchicalFiniteElementSpace(spaces::Vector{T}, two_scale_operators::Vector{O}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo, multilevel_elements::SparseArrays.SparseVector{Int, Int}, multilevel_extraction_coeffs::Vector{Array{Float64, 2}}, multilevel_basis_indices::Vector{Vector{Int}}) where {n, T <: AbstractFiniteElementSpace{n}, O<:AbstractTwoScaleOperator}
        L = length(spaces)

        # Checks for incompatible arguments
        if length(two_scale_operators) != L-1
            msg1 = "Number of two-scale operators should be 1 less than the number of hierarchical spaces. "
            msg2 = " $L refinement levels and $(length(two_scale_operators)) two-scale operators were received."
            throw(ArgumentError(msg1*msg2))
        elseif length(active_elements.levels) != L + 1
            msg1 = "Number of element levels should be one more than the number of refinement levels. "
            msg2 = " $L refinement levels and $(length(active_elements.levels)) element levels were received."
            throw(ArgumentError(msg1*msg2))
        elseif  length(active_functions.levels) != L + 1
            msg1 = "Number of function levels should be one more than the number of refinement levels. "
            msg2 = " $L refinement levels and $(length(active_functions.levels)) function levels were received."
            throw(ArgumentError(msg1*msg2))
        end
        
        new{n}(spaces, two_scale_operators, active_elements, active_functions, multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices)
    end

    function HierarchicalFiniteElementSpace(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, refined_domains::HierarchicalActiveInfo, truncated::Bool=false) where {T<:AbstractFiniteElementSpace{n} where n, O<:AbstractTwoScaleOperator}
        active_elements, active_functions = get_active_objects(fe_spaces, two_scale_operators, refined_domains)
        multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices = get_multilevel_extraction(fe_spaces, two_scale_operators, active_elements, active_functions, truncated)

        return HierarchicalFiniteElementSpace(fe_spaces, two_scale_operators, active_elements, active_functions, multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices)
    end

    function HierarchicalFiniteElementSpace(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, refinement_domains_per_level::Vector{Vector{Int}}, truncated::Bool=false) where {T<:AbstractFiniteElementSpace{n} where n, O<:AbstractTwoScaleOperator}
        refinement_domains = convert_element_vector_to_active_info(refinement_domains_per_level)
        
        return HierarchicalFiniteElementSpace(fe_spaces, two_scale_operators, refinement_domains, truncated)
    end
end

# Getters for HierarchicalActiveInfo

function convert_element_vector_to_active_info(elements_to_refine_per_level::Vector{Vector{Int}})
    ids = vcat(elements_to_refine_per_level...)
    levels = [0; cumsum(map(x -> length(x), elements_to_refine_per_level))]

    return HierarchicalActiveInfo(ids, levels)
end

function convert_element_vector_to_elements_per_level(hspace:: HierarchicalFiniteElementSpace{n}, element_vector::Vector{Int}) where {n}
    L = get_num_levels(hspace)
    elements_per_level = [Int[] for _ ∈ 1:L]
    for element ∈ element_vector
        element_level = get_active_level(hspace.active_elements, element)
        append!(elements_per_level[element_level], get_active_id(hspace.active_elements, element))
    end

    return elements_per_level
end

# Getters for HierarchicalFiniteElementSpace

"""
    get_num_active(active_info::HierarchicalActiveInfo)

Returns the number of active objects in `active_info`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: information about active objects.
# Returns
- `::Int`: number of active objects.
"""
function get_num_active(active_info::HierarchicalActiveInfo)
    return length(active_info.ids)
end

"""
    get_num_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of active elements in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
# Returns
- `::Int`: number of active elements.
"""
function get_num_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return get_num_active(hierarchical_space.active_elements)
end

"""
    get_dim(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of active functions in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
# Returns
- `::Int`: number of active functions.
"""
function get_dim(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return get_num_active(hierarchical_space.active_functions)
end

function get_max_local_dim(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return 27
end

"""
    get_num_levels(active_info::HierarchicalActiveInfo)

Returns the number of levels in `active_info`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: information about active objects.
# Returns
- `::Int`: number of levels.
"""
function get_num_levels(active_info::HierarchicalActiveInfo)
    return length(active_info.levels)-1
end

"""
    get_num_levels(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of levels in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
# Returns
- `::Int`: number of levels.
"""
function get_num_levels(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return get_num_levels(hierarchical_space.active_elements)
end

"""
    get_active_level(active_info::HierarchicalActiveInfo, index::Int)

Returns the level of the object given by `index`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: information about active objects.
- `index::Int`: index of the active object.
# Returns
- `::Int`: level of the object.
"""
function get_active_level(active_info::HierarchicalActiveInfo, index::Int)
    return findlast( x -> x < index, active_info.levels)
end

"""
    get_element_level(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}

Returns the level of the element given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `index::Int`: index of the active element.
# Returns
- `::Int`: level of the element.
"""
function get_element_level(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return get_active_level(hierarchical_space.active_elements, index)
end

"""
    get_function_level(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}

Returns the level of the function given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `index::Int`: index of the active function.
# Returns
- `::Int`: level of the function.
"""
function get_function_level(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return get_active_level(hierarchical_space.active_functions, index)
end

"""
    get_active_id(active_info::HierarchicalActiveInfo, index::Int)

Returns the corresponding id of the object given by `index` in the objects' structure.

# Arguments 
- `active_info::HierarchicalActiveInfo`: information about active objects.
- `index::Int`: index of the active object.
# Returns
- `::Int`: id of the active object.
"""
function get_active_id(active_info::HierarchicalActiveInfo, index::Int)
    return active_info.ids[index]
end


"""
    get_element_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}

Returns the corresponding id of the element given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `index::Int`: index of the active element.
# Returns
- `::Int`: id of the active element.
"""
function get_element_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return get_active_id(hierarchical_space.active_elements, index)
end

"""
    get_function_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}

Returns the corresponding id of the function given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `index::Int`: index of the active function.
# Returns
- `::Int`: id of the active function.
"""
function get_function_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return get_active_id(hierarchical_space.active_functions, index)
end

"""
    get_function_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}

Returns the function space at `level`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `level::Int`: refinement level.
# Returns
- `::AbstractFiniteElementSpace{n}`: function space at `level`.
"""
function get_space(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    return hierarchical_space.spaces[level]
end

"""
    get_level_active(active_info::HierarchicalActiveInfo, level::Int)

Returns the indices and active objects of `active_info` at `level`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: information about active objects.
- `level::Int`: refinement level.
# Returns
- `::Tuple{UnitRange{Int}, @view Vector{Int}}`: indices and active objects at `level`.
"""
function get_level_active(active_info::HierarchicalActiveInfo, level::Int)
    return active_info.levels[level]+1:active_info.levels[level+1], @view active_info.ids[active_info.levels[level]+1:active_info.levels[level+1]]
end

"""
    get_level_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}

Returns the active elements of `hierarchical_space` at `level`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `level::Int`: refinement level.
# Returns
- `::@view Vector{Int}`: active elements at `level`.
"""
function get_level_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    return get_level_active(hierarchical_space.active_elements, level)
end

"""
    get_level_functions(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}

Returns the active functions of `hierarchical_space` at `level`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `level::Int`: refinement level.
# Returns
- `::@view Vector{Int}`: active functions at `level`.
"""
function get_level_functions(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    return get_level_active(hierarchical_space.active_functions, level)
end

function get_active_index(active_info::HierarchicalActiveInfo, id::Int, level::Int)
    ids_range = active_info.levels[level]+1:active_info.levels[level+1]
    findfirst(x -> x==id, active_info.ids[ids_range])

    return ids_range[findfirst(x -> x==id, active_info.ids[ids_range])]
end

function get_active_indices(active_info::HierarchicalActiveInfo, ids, level::Int)
    ids_range = active_info.levels[level]+1:active_info.levels[level+1]

    indices = Vector{Int}(undef, length(ids))
    for (count, id) ∈ enumerate(ids)
        range_id = findfirst(x -> x== id, active_info.ids[ids_range])
        indices[count] = ids_range[range_id]
    end

    return indices
end

function truncate_refinement_matrix!(refinement_matrix, active_indices::Vector{Int})
    active_length = length(active_indices)
    refinement_matrix[active_indices, active_length+1:end] .= 0.0

    return refinement_matrix
end

# Getters for hierarchical space constructor

# 1
function get_active_objects(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, refined_domains::HierarchicalActiveInfo) where {O<:AbstractTwoScaleOperator, T<:AbstractFiniteElementSpace{n} where n}
    L = get_num_levels(refined_domains)

    # Initialize active functions and elements
    active_elements_per_level = [collect(1:get_num_elements(fe_spaces[1]))]
    active_functions_per_level = [collect(1:get_dim(fe_spaces[1]))]

    for level in 1:L-1 # Loop over levels
        next_level_domain = [get_level_active(refined_domains, level+1)[2]]

        elements_to_remove = Int[]
        elements_to_add = Int[]
        functions_to_remove = Int[]
        functions_to_add = Int[]

        for Ni ∈ active_functions_per_level[level] # Loop over active basis on current level
            # Gets the support of basis on current level and the next one
            support = get_support(fe_spaces[level], Ni)
            finer_support = get_finer_elements(two_scale_operators[level], support)
            check_in_next_domain = finer_support .∈ next_level_domain

            # Checks if the basis is active (sum_check==0), passive or inactive
            if all(check_in_next_domain) # Inactive basis
                union!(elements_to_remove, support)
                union!(functions_to_remove, Ni)
                union!(elements_to_add, finer_support)
                union!(functions_to_add, get_finer_basis_id(two_scale_operators[level], Ni))
            end
        end
        
        # Update of active elements and basis
        active_elements_per_level[level] = setdiff(active_elements_per_level[level], elements_to_remove)
        active_functions_per_level[level] = setdiff(active_functions_per_level[level], functions_to_remove)
        push!(active_elements_per_level, unique(elements_to_add))
        push!(active_functions_per_level, unique(functions_to_add))

    end

    map(x -> sort!(x), active_elements_per_level)
    map(x -> sort!(x), active_functions_per_level)

    active_elements = convert_element_vector_to_active_info(active_elements_per_level)
    active_functions = convert_element_vector_to_active_info(active_functions_per_level)
    
    return active_elements, active_functions
end

function get_inactive_active_children(active_elements, element_id, level, two_scale_operators)
    active_children = NTuple{2, Int}[]

    current_element_ids = [element_id]
    current_level = level
    all_active_check = false
    while !all_active_check
        all_active_check = true
        inactive_children = Int[]
        for curr_element_id ∈ current_element_ids 
            children = get_finer_elements(two_scale_operators[current_level], curr_element_id)
            children_check = children .∈ [get_level_active(active_elements, current_level+1)[2]]
            for child_id ∈ children[children_check]
                push!(active_children, (current_level+1, child_id))
            end
            append!(inactive_children, children[map(!, children_check)])
            all_active_check = all_active_check && all(children_check)
        end
        current_element_ids = inactive_children
        current_level += 1
    end

    return active_children
end

function get_multilevel_information(fe_spaces, two_scale_operators, active_elements, active_functions)
    L = get_num_levels(active_elements)
    multilevel_information = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    # Above, first tuple is level and id of ml element, second tuple
    # is level and id of ml basis in that element
    for level ∈ 1:L
        level_active_elements = [get_level_active(active_elements, level)[2]]
        _, active_basis = get_level_active(active_functions, level)

        for basis ∈ active_basis
            support = get_support(fe_spaces[level], basis)
            if level == 1 && basis==26
                nothing
            end

            active_support_checks = support .∈ level_active_elements
            for inactive_element ∈ support[map(!, active_support_checks)]
                active_children = get_inactive_active_children(active_elements, inactive_element, level, two_scale_operators)
                for (child_level, child_id) ∈ active_children
                    if haskey(multilevel_information, (child_level, child_id))
                        push!(multilevel_information[(child_level, child_id)], (level, basis))
                    else
                        multilevel_information[(child_level, child_id)] = [(level, basis)]
                    end
                end
            end
        end
    end

    return multilevel_information
end

function get_active_basis_matrix(fe_spaces, element, level, active_functions)
    full_coeffs, full_indices = get_extraction(fe_spaces[level], element)
    local_active_indices = findall(x -> x in get_level_active(active_functions, level)[2], full_indices)

    return Matrix{Float64}(LinearAlgebra.I, size(full_coeffs))[:, local_active_indices], local_active_indices
end

function get_multilevel_basis_evaluation(fe_spaces, two_scale_operators, active_functions, basis_level, basis_id, element_level, element_id, truncated::Bool)
    local_subdiv_matrix = LinearAlgebra.I
    current_child_element = element_id

    for level ∈ element_level:-1:basis_level+1
        current_parent_element = get_coarser_element(two_scale_operators[level-1], current_child_element)

        current_subdiv_matrix = get_local_subdiv_matrix(two_scale_operators[level-1], current_parent_element, current_child_element)

        if truncated
            _, full_level_indices = get_extraction(fe_spaces[level], current_child_element)
            active_indices = findall(x -> x in get_level_active(active_functions, level)[2], full_level_indices)
            current_subdiv_matrix[active_indices, :] .= 0.0
        end
        
        if level==element_level
            local_subdiv_matrix = local_subdiv_matrix * current_subdiv_matrix
        else
            local_subdiv_matrix .= local_subdiv_matrix * current_subdiv_matrix
        end
        current_child_element = current_parent_element
    end

    level_diff = element_level-basis_level
    basis_element_id = get_ancestor_element(two_scale_operators, element_id, element_level, level_diff)
    _, lowest_level_basis_indices = get_extraction(fe_spaces[basis_level], basis_element_id)
    basis_local_id = findfirst(x -> x == basis_id, lowest_level_basis_indices)

    return @view local_subdiv_matrix[:, basis_local_id]
end

function get_refinement_data(active_basis_matrix, local_active_indices, fe_spaces, two_scale_operators, active_functions, element_id, element_level, multilevel_information, truncated)
    active_basis_size = size(active_basis_matrix)
    multilevel_basis_length = length(multilevel_information[(element_level, element_id)])

    refinement_matrix = zeros(active_basis_size .+ (0, multilevel_basis_length))

    refinement_matrix[1:active_basis_size[1],1:active_basis_size[2]] .= active_basis_matrix

    multilevel_basis_hspace_indices = Vector{Int}(undef, multilevel_basis_length)
    
    ml_basis_count = 1
    for (basis_level, basis_id) ∈ multilevel_information[(element_level, element_id)]
        refinement_matrix[:,active_basis_size[2]+ml_basis_count] .= get_multilevel_basis_evaluation(fe_spaces, two_scale_operators, active_functions, basis_level, basis_id, element_level, element_id, truncated)

        multilevel_basis_hspace_indices[ml_basis_count] = get_active_index(active_functions, basis_id, basis_level)

        ml_basis_count += 1
    end
    if truncated
        refinement_matrix = truncate_refinement_matrix!(refinement_matrix, local_active_indices)
    end

    return refinement_matrix, multilevel_basis_hspace_indices
end

# 2
function get_multilevel_extraction(fe_spaces, two_scale_operators, active_elements, active_functions, truncated)
    multilevel_information = get_multilevel_information(fe_spaces, two_scale_operators, active_elements, active_functions)
    
    num_multilevel_elements = length(keys(multilevel_information))

    if num_multilevel_elements == 0 # Skip trivial case (first adaptive iteration step)
        return SparseArrays.spzeros(Int, get_num_active(active_elements)), Matrix{Float64}[], [Int[]]
    end

    multilevel_element_indices = Vector{Int}(undef, num_multilevel_elements)
    multilevel_extraction_coeffs = Vector{Matrix{Float64}}(undef, num_multilevel_elements)
    multilevel_basis_indices = Vector{Vector{Int}}(undef, num_multilevel_elements)

    ml_id_count = 1
    for (level, element) ∈ keys(multilevel_information)
        # Create multilevel element specific extraction coefficients
        if (level, element) == (3,15)
            nothing
        end
        active_basis_matrix, local_active_indices = get_active_basis_matrix(fe_spaces, element, level, active_functions)
        refinement_matrix, multilevel_basis_hspace_indices = get_refinement_data(active_basis_matrix, local_active_indices, fe_spaces, two_scale_operators, active_functions, element, level, multilevel_information, truncated)

        element_coeffs, element_basis_indices = get_extraction(fe_spaces[level], element)
        element_hspace_basis_indices = map(x -> get_active_index(active_functions, x, level), element_basis_indices[local_active_indices])

        # Add multilevel extraction data
        multilevel_extraction_coeffs[ml_id_count] = element_coeffs * refinement_matrix
        multilevel_basis_indices[ml_id_count] = append!(element_hspace_basis_indices, multilevel_basis_hspace_indices)

        # Add multilevel element specific index
        multilevel_element_indices[ml_id_count] = get_active_index(active_elements, element, level)
        ml_id_count += 1
    end

    multilevel_elements = SparseArrays.sparsevec(multilevel_element_indices, 1:num_multilevel_elements, get_num_active(active_elements))

    return multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices
end

# Extraction method for hierarchical space

function get_local_basis(hierarchical_space::HierarchicalFiniteElementSpace{n}, element::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n}
    element_level = get_active_level(hierarchical_space.active_elements, element)
    element_id = get_active_id(hierarchical_space.active_elements, element)

    return get_local_basis(hierarchical_space.spaces[element_level], element_id, xi, nderivatives)
end

function get_extraction(hierarchical_space::HierarchicalFiniteElementSpace{n}, element::Int) where {n}
    if hierarchical_space.multilevel_elements[element] == 0
        element_level = get_active_level(hierarchical_space.active_elements, element)
        element_id = get_active_id(hierarchical_space.active_elements, element)

        coeffs, level_basis_indices = get_extraction(hierarchical_space.spaces[element_level], element_id)

        # Convert level space basis indices to hierarchical space basis indices
        basis_indices = get_active_indices(hierarchical_space.active_functions, level_basis_indices, element_level)
    else
        multilevel_idx = hierarchical_space.multilevel_elements[element]
        coeffs = hierarchical_space.multilevel_extraction_coeffs[multilevel_idx]
        basis_indices = hierarchical_space.multilevel_basis_indices[multilevel_idx]
    end

    return coeffs, basis_indices
end

# Useful for L-chain
function get_level_inactive_domain(hspace::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    inactive_basis = setdiff(1:get_num_elements(hspace.spaces[level]), get_level_active(hspace.active_elements, level)[2])
    if level > 1
        inactive_basis = setdiff(inactive_basis, get_level_active(hspace.active_elements, level-1)[2])
    end

    return inactive_basis
end

function get_basis_contained_in_next_level(hspace::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    basis_contained_in_next_level = Int[]
    inactive_domain = get_level_inactive_domain(hspace, level)
    for basis ∈ setdiff(1:get_dim(hspace.spaces[level]), get_level_active(hspace.active_functions, level)[2])
        basis_support = get_support(hspace.spaces[level], basis)
        support_in_omega, _ = Mesh.check_contained(basis_support, inactive_domain)
        if support_in_omega
            append!(basis_contained_in_next_level, basis)
        end
    end

    return basis_contained_in_next_level
end

function get_level_domain(hspace::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    if level == get_num_levels(hspace)
        return get_level_active(hspace.active_elements, level)[2]
    else
        level_active = collect(get_level_active(hspace.active_elements, level)[2])
        for l ∈ level+1:get_num_levels(hspace)
            next_domain = get_level_active(hspace.active_elements, l)[2]
            next_domain_in_level = get_ancestor_element(hspace.two_scale_operators, next_domain, l, l-level)
            append!(level_active, next_domain_in_level)
        end

        return level_active
    end
end

