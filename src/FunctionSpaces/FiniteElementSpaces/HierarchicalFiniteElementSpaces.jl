"""
This (sub-)module provides functionality related with hierarchical refinment.

The exported names are:
"""

include("HierarchicalRelations.jl")

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
    two_scale_operators::Vector{TwoScaleOperator}
    active_elements::HierarchicalActiveInfo
    active_functions::HierarchicalActiveInfo

    function HierarchicalFiniteElementSpace(spaces::Vector{T}, two_scale_operators::Vector{TwoScaleOperator}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo) where {n, T <: AbstractFiniteElementSpace{n}}
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
        
        new{n}(spaces, two_scale_operators, active_elements, active_functions)
    end
end

include("HierarchicalRelations.jl")

# Getters for HierarchicalFiniteElementSpace

"""
    get_n_active(active_info::HierarchicalActiveInfo)

Returns the number of active objects in `active_info`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: information about active objects.
# Returns
- `::Int`: number of active objects.
"""
function get_n_active(active_info::HierarchicalActiveInfo)
    return length(active_info.ids)
end

"""
    get_n_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of active elements in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
# Returns
- `::Int`: number of active elements.
"""
function get_n_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return get_n_active(hierarchical_space.active_elements)
end

"""
    get_n_functions(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of active functions in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
# Returns
- `::Int`: number of active functions.
"""
function get_n_functions(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return get_n_active(hierarchical_space.active_functions)
end

"""
    get_n_levels(active_info::HierarchicalActiveInfo)

Returns the number of levels in `active_info`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: information about active objects.
# Returns
- `::Int`: number of levels.
"""
function get_n_levels(active_info::HierarchicalActiveInfo)
    return length(active_info.levels)-1
end

"""
    get_n_levels(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of levels in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
# Returns
- `::Int`: number of levels.
"""
function get_n_levels(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return get_n_levels(hierarchical_space.active_elements)
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
function get_space(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return hierarchical_space.spaces[index]
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

function get_active_indices(active_info::HierarchicalActiveInfo, ids::UnitRange{Int}, level::Int)
    ids_range = active_info.levels[level]+1:active_info.levels[level+1]

    return ids_range[active_info.ids[ids_range] .∈ [ids]]
end

function get_extraction(hierarchical_space::HierarchicalFiniteElementSpace, element::Int)
    level = get_element_level(hierarchical_space, element)
    element_id = hierarchical_space.active_elements.ids[element]
    return get_extraction(hierarchical_space.spaces[level], element_id)
end

function evaluate(hierarchical_space::HierarchicalFiniteElementSpace, element::Int, xi::Vector{Float64}, nderivatives::Int)
    level = get_element_level(hierarchical_space, element)
    element_id = hierarchical_space.active_elements.ids[element]
    return evaluate(hierarchical_space.spaces[level], element_id, xi, nderivatives)
end

# Updaters for the Hierarcical Active information

function remove_active!(active_info::HierarchicalActiveInfo, indices::Vector{Int}, level::Int)
    indices = sort!(unique(indices))
    deleteat!(active_info.ids, indices)
    active_info.levels[level+1] -= length(indices)

    return active_info
end

function add_active!(active_info::HierarchicalActiveInfo, ids::Vector{Int}, level::Int)
    ids = sort!(unique(ids))
    if level <= get_n_levels(active_info)
        index = active_info.levels[level+1]+1
        active_info.ids = [active_info.ids[1:index-1]; ids; active_info.ids[index:end]]
        levels[level+1] += length(ids)
    else
        append!(active_info.ids, ids)
        append!(active_info.levels, active_info.levels[end]+length(ids))
    end

    return active_info
end

# Hierarchical space constructor

function get_hierarchical_space(fe_spaces::Vector{T}, two_scale_operators::Vector{TwoScaleOperator}, refined_domains::HierarchicalActiveInfo, nsubdivisions::Vector{Int}) where {T<:AbstractFiniteElementSpace{n} where n}
    L = length(fe_spaces)

    # Initialize active functions and elements
    active_elements = HierarchicalActiveInfo(collect(1:get_num_elements(fe_spaces[1])),[0, get_num_elements(fe_spaces[1])])
    active_functions = HierarchicalActiveInfo(collect(1:get_dim(fe_spaces[1])),[0, get_dim(fe_spaces[1])])

    for level in 1:L-1 # Loop over levels
        _, next_level_domain = get_level_active(refined_domains, level+1)

        element_indices_to_remove = Int[]
        elements_to_add = Int[]
        function_indices_to_remove = Int[]
        functions_to_add = Int[]

        index, Ni = get_level_active(active_functions, level)
        for i in 1:length(Ni) # Loop over active functions in level
            support_check, support, finer_support = check_support(fe_spaces[level], Ni[i], next_level_domain, nsubdivisions[level]) # gets support in both levels

            if support_check
                append!(element_indices_to_remove, get_active_indices(active_elements, support, level))
                append!(elements_to_add, finer_support)
                append!(function_indices_to_remove, index[i])
                append!(functions_to_add, collect(get_finer_basis_id(Ni[i], two_scale_operators[level]))[1])
            end
        end

        remove_active!(active_elements, element_indices_to_remove, level)
        add_active!(active_elements, elements_to_add, level+1)
        remove_active!(active_functions, function_indices_to_remove, level)
        add_active!(active_functions, functions_to_add, level+1)
    end

    return HierarchicalFiniteElementSpace(fe_spaces, two_scale_operators, active_elements, active_functions)
end