@doc raw"""
    struct HierarchicalActiveInfo

# Description
`ids` is a collection of linear ids of active objects, whether elements or functions.
`levels` contains information about the level of each active object by encoding the 
indexes of the last active objects, of the `ids` vector, from each level. 
For example, if `levels = [0, n1, n2, ...]`, then:
- `ids[1:n1]` will contain all active objects from level 1
- `ids[n1+1:n2]` will contain all active objects from level 2, and so forth.
"""
struct HierarchicalActiveInfo
    ids::Vector{Int}
    levels::Vector{Int} 
end

@doc raw"""
    struct HierarchicalFiniteElementSpace{n, S, T} <: AbstractFiniteElementSpace{n}

A hierarchical space that is built from nested hierarchies of `n`-variate function spaces and domains. 

# Fields
- `spaces::Vector{AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}} `: collection of `n`-variate 
    function spaces.
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: collection of two-scale operators. 
    relating each consequtive pair of function spaces.
- `active_elements::HierarchicalActiveInfo`: information about the active elements in each level.
- `active_basis::HierarchicalActiveInfo`: information about the active basis in each level.
- `multilevel_elements::SparseArrays.SparseVector{Int, Int}`: elements where basis from multiple levels
    have non-empty support.
- `multilevel_extraction_coeffs::Vector{Matrix{Float64}}`: extraction coefficients of active basis in `multilevel_elements`.
- `multilevel_basis_indices::Vector{Vector{Int}}`: indices of active basis in `multilevel_elements`.
"""
struct HierarchicalFiniteElementSpace{n, S, T} <: AbstractFiniteElementSpace{n}
    spaces::Vector{S}   
    two_scale_operators::Vector{T}
    active_elements::HierarchicalActiveInfo
    active_basis::HierarchicalActiveInfo
    multilevel_elements::SparseArrays.SparseVector{Int, Int}
    multilevel_extraction_coeffs::Vector{Matrix{Float64}}
    multilevel_basis_indices::Vector{Vector{Int}}

    # General constructor which checks for argument logic
    function HierarchicalFiniteElementSpace(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo, multilevel_elements::SparseArrays.SparseVector{Int, Int}, multilevel_extraction_coeffs::Vector{Array{Float64, 2}}, multilevel_basis_indices::Vector{Vector{Int}}) where {n,  S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
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
        elseif  length(active_basis.levels) != L + 1
            msg1 = "Number of function levels should be one more than the number of refinement levels. "
            msg2 = " $L refinement levels and $(length(active_basis.levels)) function levels were received."
            throw(ArgumentError(msg1*msg2))
        end
        
        new{n, S, T}(spaces, two_scale_operators, active_elements, active_basis, multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices)
    end

    # Constructor that builds the space
    function HierarchicalFiniteElementSpace(spaces::Vector{S}, two_scale_operators::Vector{T}, marked_domains::HierarchicalActiveInfo, truncated::Bool=false) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
        active_elements, active_basis = get_active_objects(spaces, two_scale_operators, marked_domains)
        multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices = get_multilevel_extraction(spaces, two_scale_operators, active_elements, active_basis, truncated)

        return HierarchicalFiniteElementSpace(spaces, two_scale_operators, active_elements, active_basis, multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices)
    end

    # Helper constructor for domains given on a per-level way.
    function HierarchicalFiniteElementSpace(spaces::Vector{S}, two_scale_operators::Vector{T}, marked_domains_per_level::Vector{Vector{Int}}, truncated::Bool=false) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
        marked_domains = convert_elements_per_level_to_active_info(marked_domains_per_level)
        
        return HierarchicalFiniteElementSpace(spaces, two_scale_operators, marked_domains, truncated)
    end
end

# Getters for HierarchicalActiveInfo
@doc raw"""
    convert_element_vector_to_elements_per_level(hspace::HierarchicalFiniteElementSpace{n, S, T}, marked_elements::Vector{Int}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns a set of marked elements, separated by refinement level, as `::Vector{Vector{Int}}` from the `marked_elements`
given in the `hspace` indexing. 

# Arguments
- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical space of the marked elements.
- `marked_elements::Vector{Int}`: set of marked elements in the hierarchical space indexing.

# Returns
- `::Vector{Vector{Int}}`: marked elements separated by level.
"""
function convert_element_vector_to_elements_per_level(hspace::HierarchicalFiniteElementSpace{n, S, T}, marked_elements::Vector{Int}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    L = get_num_levels(hspace)
    marked_elements_per_level = [Int[] for _ ∈ 1:L]
    
    # Separate the marked elements per level
    for el ∈ marked_elements
        el_level = get_active_level(hspace.active_elements, el)
        append!(marked_elements_per_level[el_level], hspace.active_elements.ids[el])
    end

    return marked_elements_per_level
end

@doc raw"""
    convert_elements_per_level_to_active_info(marked_elements_per_level::Vector{Vector{Int}})

Converts a set of `marked_elements_per_level` into an `HierarchicalActiveInfo` structure.

# Arguments

`marked_elements_per_level::Vector{Vector{Int}}`: set of marked elements separated by level.

# Returns
`::HierarchicalActiveInfo`: marked elements as `HierarchicalActiveInfo`
"""
function convert_elements_per_level_to_active_info(marked_elements_per_level::Vector{Vector{Int}})
    ids = vcat(marked_elements_per_level...)
    levels = [0; cumsum(map(x -> length(x), marked_elements_per_level))]

    return HierarchicalActiveInfo(ids, levels)
end

# Getters for HierarchicalFiniteElementSpace

"""
    get_num_active(active_info::HierarchicalActiveInfo)

Returns the number of active objects in `active_info`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: Information about active objects.

# Returns
- `::Int`: Number of active objects.
"""
function get_num_active(active_info::HierarchicalActiveInfo)
    return length(active_info.ids)
end

"""
    get_num_elements(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the number of active elements in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
# Returns
- `::Int`: Number of active elements.
"""
function get_num_elements(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_num_active(hierarchical_space.active_elements)
end

"""
    get_num_basis(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the number of active functions in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
# Returns
- `::Int`: Number of active functions.
"""
function get_num_basis(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_num_active(hierarchical_space.active_basis)
end

# get_extraction(hierarchical_space, element)[2]

function get_max_local_dim(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_max_local_dim(hierarchical_space.spaces[1])*2 # This needs to be checked
end


"""
    get_num_levels(active_info::HierarchicalActiveInfo)

Returns the number of levels in `active_info`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: Information about active objects.

# Returns
- `::Int`: Number of levels.
"""
function get_num_levels(active_info::HierarchicalActiveInfo)
    return length(active_info.levels) - 1
end

"""
    get_num_levels(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the number of levels in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
# Returns
- `::Int`: Number of levels.
"""
function get_num_levels(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_num_levels(hierarchical_space.active_elements)
end

"""
    get_active_level(active_info::HierarchicalActiveInfo, index::Int)

Returns the level of the object given by `index`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: Information about active objects.
- `index::Int`: Index of the active object.

# Returns
- `::Int`: Level of the object.
"""
function get_active_level(active_info::HierarchicalActiveInfo, index::Int)
    return findlast(x -> x < index, active_info.levels)
end

"""
    get_element_level(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, index::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the level of the element given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
- `index::Int`: index of the active element.
# Returns
- `::Int`: Level of the element.
"""
function get_element_level(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, index::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_active_level(hierarchical_space.active_elements, index)
end

"""
    get_function_level(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, index::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the level of the function given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
- `index::Int`: index of the active function.
# Returns
- `::Int`: Level of the function.
"""
function get_function_level(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, index::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_active_level(hierarchical_space.active_basis, index)
end

"""
    get_active_id(active_info::HierarchicalActiveInfo, index::Int)

Returns the corresponding id of the object given by `index` in the objects' structure.

# Arguments 
- `active_info::HierarchicalActiveInfo`: Information about active objects.
- `index::Int`: Index of the active object.

# Returns
- `::Int`: ID of the active object.
"""
function get_active_id(active_info::HierarchicalActiveInfo, index::Int)
    return active_info.ids[index]
end

"""
    get_element_id(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, index::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the corresponding id of the element given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
- `index::Int`: index of the active element.
# Returns
- `::Int`: ID of the active element.
"""
function get_element_id(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, index::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_active_id(hierarchical_space.active_elements, index)
end

"""
    get_function_id(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, index::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the corresponding id of the function given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
- `index::Int`: index of the active function.
# Returns
- `::Int`: ID of the active function.
"""
function get_function_id(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, index::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_active_id(hierarchical_space.active_basis, index)
end

"""
    get_function_id(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the function space at the specified `level` from the hierarchical space.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
- `level::Int`: refinement level.
# Returns
- `::AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}`: function space at `level`.
"""
function get_space(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return hierarchical_space.spaces[level]
end

"""
    get_level_active(active_info::HierarchicalActiveInfo, level::Int)

Returns the indices and active objects of `active_info` at the specified `level`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: Information about active objects.
- `level::Int`: Refinement level.

# Returns
- `::Tuple{UnitRange{Int}, SubArray{Int,1}}`: A tuple containing:
  1. A `UnitRange` of indices for the specified level.
  2. A view of the active object IDs for the specified level.

# Notes
- The function uses the `levels` field of `active_info` to determine the range of indices for the given level.
- The active object IDs are accessed using a view to avoid unnecessary copying.
"""
function get_level_active(active_info::HierarchicalActiveInfo, level::Int)
    # Calculate the range of indices for the specified level
    index_range = active_info.levels[level]+1:active_info.levels[level+1]
    
    # Return the range and a view of the active IDs for the level
    return index_range, @view active_info.ids[index_range]
end

"""
    get_level_elements(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the active elements of `hierarchical_space` at the specified `level`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
- `level::Int`: refinement level.
# Returns
- `::Tuple{UnitRange{Int}, SubArray{Int,1}}`: A tuple containing:
  1. A `UnitRange` of indices for the active elements at the specified level.
  2. A view of the active element IDs for the specified level.

# Notes
- This function delegates to `get_level_active` using the `active_elements` field of the hierarchical space.
"""
function get_level_elements(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level_active(hierarchical_space.active_elements, level)
end

"""
    get_level_functions(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the active functions of `hierarchical_space` at the specified `level`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}`: Hierarchical function space.
- `level::Int`: refinement level.
# Returns
- `::Tuple{UnitRange{Int}, SubArray{Int,1}}`: A tuple containing:
  1. A `UnitRange` of indices for the active functions at the specified level.
  2. A view of the active function IDs for the specified level.

# Notes
- This function delegates to `get_level_active` using the `active_functions` field of the hierarchical space.
"""
function get_level_functions(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level_active(hierarchical_space.active_basis, level)
end

"""
    get_active_index(active_info::HierarchicalActiveInfo, id::Int, level::Int)

Returns the index in hierarchical indexing of the `id` in `level` indexing.

# Arguments

- `active_info::HierarchicalActiveInfo`: information about active hierarchical objects.
- `id::Int`: index in level indexing.
- `level::Int`: hierarchical level of indexing.

# Returns
`::Int`: index in hierarchical indexing.
"""
function get_active_index(active_info::HierarchicalActiveInfo, id::Int, level::Int)
    ids_range = active_info.levels[level]+1:active_info.levels[level+1]

    return ids_range[findfirst(x -> x==id, @view active_info.ids[ids_range])]
end

@doc raw"""
    get_active_indices(active_info::HierarchicalActiveInfo, id::Int, level::Int)

Equivalent to `get_active_index` for multiple `ids` in `level`.
"""
function get_active_indices(active_info::HierarchicalActiveInfo, ids, level::Int)
    # Determine the range of indices for the given level
    ids_range = active_info.levels[level]+1:active_info.levels[level+1]

    return ids_range[findall(x -> x ∈ ids, @view active_info.ids[ids_range])]
end

# Truncation of refinement matrix

@doc raw"""
    truncate_refinement_matrix!(refinement_matrix, active_indices::Vector{Int})

Updates `refinement_matrix` by the rows of `active_indices` to zeros in lower level basis functions.

# Arguments

- `refinement_matrix`: the refinement matrix to be updated.
- `active_indices::Vector{Int}`: element local indices of active basis functions from the highest refinement level.

# Returns

- `refinement_matrix`: truncated refinement matrix.
"""
function truncate_refinement_matrix!(refinement_matrix, active_indices::Vector{Int})
    active_length = length(active_indices)
    refinement_matrix[active_indices, active_length+1:end] .= 0.0

    return refinement_matrix
end

# Getters for hierarchical space constructor

@doc raw"""
    get_active_objects(spaces::Vector{S}, two_scale_operators::Vector{T}, marked_domains::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Computes the active elements and basis on each level based on `spaces`, `two_scale_operators` and the set of nested `marked_domains`.

The construction loops over the `marked_domains` on each level and selects the active basis in the next level as the children of deactivated basis, based on their supports,
in the current level. The active elments in the next level are then given as the union of support of said basis in the next level. 
This differs slightly from the usual algorithm for generating the hierarchical space, where basis in the next level are only determined by whether their support is fully contained
in the next level domain, regardless of whether their parent basis are active or not.


# Arguments
- `spaces::Vector{AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}}`: finite element spaces at each level. 
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the finite element spaces at each level.
- `marked_domains::HierarchicalActiveInfo`: nested domains where the support of active basis is determined.

# Returns
- `active_elements::HierarchicalActiveInfo`: active elements on each level.
- `active_basis::HierarchicalActiveInfo`: active basis on each level.
"""
function get_active_objects(spaces::Vector{S}, two_scale_operators::Vector{T}, marked_domains::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    L = get_num_levels(marked_domains)

    # Initialize active basis and elements on first level
    active_elements_per_level = [collect(1:get_num_elements(spaces[1]))]
    active_basis_per_level = [collect(1:get_num_basis(spaces[1]))]

    for level in 1:L-1 # Loop over levels
        next_level_domain = [get_level_active(marked_domains, level+1)[2]]

        elements_to_remove = Int[]
        elements_to_add = Int[]
        basis_to_remove = Int[]
        basis_to_add = Int[]

        for Ni ∈ active_basis_per_level[level] # Loop over active basis on current level
            # Gets the support of Ni on current level and the next one
            support = get_support(spaces[level], Ni)
            finer_support = get_element_children(two_scale_operators[level], support)
            check_in_next_domain = finer_support .∈ next_level_domain # checks if the support is contained in the next level domain

            # Updates elements and basis to add and remove based on check_in_next_domain
            if all(check_in_next_domain)
                union!(elements_to_remove, support)
                append!(basis_to_remove, Ni)
                union!(elements_to_add, finer_support)
                union!(basis_to_add, get_finer_basis_id(two_scale_operators[level], Ni))
            end
        end
        
        # Remove inactive elements and basis on current level
        active_elements_per_level[level] = setdiff(active_elements_per_level[level], elements_to_remove)
        active_basis_per_level[level] = setdiff(active_basis_per_level[level], basis_to_remove)
        # Add active elements and basis on next level
        push!(active_elements_per_level, unique(elements_to_add))
        push!(active_basis_per_level, unique(basis_to_add))
    end

    map(x -> sort!(x), active_elements_per_level)
    map(x -> sort!(x), active_basis_per_level)

    active_elements = convert_elements_per_level_to_active_info(active_elements_per_level)
    active_basis = convert_elements_per_level_to_active_info(active_basis_per_level)
    
    return active_elements, active_basis
end

@doc raw"""
    get_inactive_active_children(active_elements::HierarchicalActiveInfo, element_id::Int, level::Int, two_scale_operators::Vector{T}) where {T<:AbstractTwoScaleOperator}

Computes all the active elements that are the children of a deactivated element.

# Arguments
- `active_elements::HierarchicalActiveInfo`: active elements in all levels.
- `element_id::Int`
- `level::Int`: level of the element give by `element_id`.
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators between levels.

# Returns
- `active_children::Vector{NTuple{2, Int}}`: Vector containing all the active children where the first index is the child's level and the second the child's id. 
"""
function get_inactive_active_children(active_elements::HierarchicalActiveInfo, element_id::Int, level::Int, two_scale_operators::Vector{T}) where {T<:AbstractTwoScaleOperator}
    active_children = NTuple{2, Int}[]

    current_element_ids = [element_id]
    current_level = level
    all_active_check = false
    while !all_active_check
        all_active_check = true
        inactive_children = Int[]
        for curr_element_id ∈ current_element_ids 
            children = get_element_children(two_scale_operators[current_level], curr_element_id)
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

@doc raw"""
    get_multilevel_information(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Computes which active elements are multilevel elements, i.e. elements where basis from multiple levels have non-empty support, as well as which basis from coarser levels are active on those elements.

# Arguments

- `spaces::Vector{AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}}`: finite element spaces at each level. 
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the finite element spaces at each level.
- `active_elements::HierarchicalActiveInfo`: active elements on each level.
- `active_basis::HierarchicalActiveInfo`: active basis on each level.

# Returns
- `multilevel_information::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}`: information about multilevel elements. The key's two indices indicate the multilevel element's level and id and the and the key's value is a vector of tuples where the indices are the basis level and id (from coarser levels), respectively.
"""
function get_multilevel_information(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    L = get_num_levels(active_elements)
    multilevel_information = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    # Above, first tuple is level and id of ml element, second tuple
    # is level and id of ml basis in that element
    for level ∈ 1:L
        level_active_elements = [get_level_active(active_elements, level)[2]]
        _, level_active_basis = get_level_active(active_basis, level)

        for basis ∈ level_active_basis
            support = get_support(spaces[level], basis)

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

function get_active_basis_matrix(fe_spaces, element, level, active_basis)
    full_coeffs, full_indices = get_extraction(fe_spaces[level], element)
    local_active_indices = findall(x -> x in get_level_active(active_basis, level)[2], full_indices)

    return Matrix{Float64}(LinearAlgebra.I, size(full_coeffs))[:, local_active_indices], local_active_indices
end

function get_multilevel_basis_evaluation(fe_spaces, two_scale_operators, active_basis, basis_level, basis_id, element_level, element_id, truncated::Bool)
    local_subdiv_matrix = LinearAlgebra.I
    current_child_element = element_id

    for level ∈ element_level:-1:basis_level+1
        current_parent_element = get_coarser_element(two_scale_operators[level-1], current_child_element)

        current_subdiv_matrix = get_local_subdiv_matrix(two_scale_operators[level-1], current_parent_element, current_child_element)

        if truncated
            _, full_level_indices = get_extraction(fe_spaces[level], current_child_element)
            active_indices = findall(x -> x in get_level_active(active_basis, level)[2], full_level_indices)
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

function get_refinement_data(active_basis_matrix, local_active_indices, fe_spaces, two_scale_operators, active_basis, element_id, element_level, multilevel_information, truncated)
    active_basis_size = size(active_basis_matrix)
    multilevel_basis_length = length(multilevel_information[(element_level, element_id)])

    refinement_matrix = zeros(active_basis_size .+ (0, multilevel_basis_length))

    refinement_matrix[1:active_basis_size[1],1:active_basis_size[2]] .= active_basis_matrix

    multilevel_basis_hspace_indices = Vector{Int}(undef, multilevel_basis_length)
    
    ml_basis_count = 1
    for (basis_level, basis_id) ∈ multilevel_information[(element_level, element_id)]
        refinement_matrix[:,active_basis_size[2]+ml_basis_count] .= get_multilevel_basis_evaluation(fe_spaces, two_scale_operators, active_basis, basis_level, basis_id, element_level, element_id, truncated)

        multilevel_basis_hspace_indices[ml_basis_count] = get_active_index(active_basis, basis_id, basis_level)

        ml_basis_count += 1
    end
    if truncated
        refinement_matrix = truncate_refinement_matrix!(refinement_matrix, local_active_indices)
    end

    return refinement_matrix, multilevel_basis_hspace_indices
end

@doc raw"""
    get_multilevel_extraction(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo, truncated::Bool) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Computes which elements are multilevel elements, i.e. elements for which basis from multiple levels have non-emtpy support, as well as their extraction coefficients matrices and active basis indices.

The extraction coefficients depend on whether the hierarchical space is `truncated` or not.

# Arguments
- `spaces::Vector{AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}}`: finite element spaces at each level. 
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the finite element spaces at each level.
- `active_elements::HierarchicalActiveInfo`: active elements on each level.
- `active_basis::HierarchicalActiveInfo`: active basis on each level.
- `truncated`: flag for a truncated hierarchical space.

# Returns 
- `multilevel_elements::SparseArrays.SparseVector{Int, Int}`: elements where basis from multiple levels
    have non-empty support.
- `multilevel_extraction_coeffs::Vector{Matrix{Float64}}`: extraction coefficients of active basis in `multilevel_elements`.
- `multilevel_basis_indices::Vector{Vector{Int}}`: indices of active basis in `multilevel_elements`.
"""
function get_multilevel_extraction(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo, truncated::Bool) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    multilevel_information = get_multilevel_information(spaces, two_scale_operators, active_elements, active_basis)
    
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
        active_basis_matrix, local_active_indices = get_active_basis_matrix(spaces, element, level, active_basis)
        refinement_matrix, multilevel_basis_hspace_indices = get_refinement_data(active_basis_matrix, local_active_indices, spaces, two_scale_operators, active_basis, element, level, multilevel_information, truncated)

        element_coeffs, element_basis_indices = get_extraction(spaces[level], element)
        element_hspace_basis_indices = map(x -> get_active_index(active_basis, x, level), element_basis_indices[local_active_indices])

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

function get_local_basis(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, element::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    element_level = get_active_level(hierarchical_space.active_elements, element)
    element_id = get_active_id(hierarchical_space.active_elements, element)

    return get_local_basis(hierarchical_space.spaces[element_level], element_id, xi, nderivatives)
end

function get_extraction(hierarchical_space::HierarchicalFiniteElementSpace{n, S, T}, element::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    if hierarchical_space.multilevel_elements[element] == 0
        element_level = get_active_level(hierarchical_space.active_elements, element)
        element_id = get_active_id(hierarchical_space.active_elements, element)

        coeffs, level_basis_indices = get_extraction(hierarchical_space.spaces[element_level], element_id)

        # Convert level space basis indices to hierarchical space basis indices
        basis_indices = get_active_indices(hierarchical_space.active_basis, level_basis_indices, element_level)
    else
        multilevel_idx = hierarchical_space.multilevel_elements[element]
        coeffs = hierarchical_space.multilevel_extraction_coeffs[multilevel_idx]
        basis_indices = hierarchical_space.multilevel_basis_indices[multilevel_idx]
    end

    return coeffs, basis_indices
end

# Useful for L-chain

function get_level_inactive_domain(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    inactive_basis = setdiff(1:get_num_elements(hspace.spaces[level]), get_level_active(hspace.active_elements, level)[2])
    if level > 1
        inactive_basis = setdiff(inactive_basis, get_level_active(hspace.active_elements, level-1)[2])
    end

    return inactive_basis
end

function get_basis_contained_in_next_level(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    basis_contained_in_next_level = Int[]
    inactive_domain = get_level_inactive_domain(hspace, level)
    for basis ∈ setdiff(1:get_num_basis(hspace.spaces[level]), get_level_active(hspace.active_basis, level)[2])
        basis_support = get_support(hspace.spaces[level], basis)
        support_in_omega, _ = Mesh.check_contained(basis_support, inactive_domain)
        if support_in_omega
            append!(basis_contained_in_next_level, basis)
        end
    end

    return basis_contained_in_next_level
end

function get_level_domain(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
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

# Boundary methods

function get_boundary_dof_indices(hspace::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

    L = get_num_levels(hspace)

    boundary_dof_indices = Vector{Int}(undef, 0)
    for level ∈ 1:L
        level_boundary_dof_indices = get_boundary_dof_indices(hspace.spaces[level])
        active_hspace_indices, active_level_indices = get_level_active(hspace.active_basis, level)

        append!(boundary_dof_indices, active_hspace_indices[findall(id -> id ∈ level_boundary_dof_indices, active_level_indices)])
    end

    return boundary_dof_indices
end

