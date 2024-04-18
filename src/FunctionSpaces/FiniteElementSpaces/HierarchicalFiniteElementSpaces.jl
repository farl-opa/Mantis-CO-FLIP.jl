"""
This (sub-)module provides functionality related with hierarchical refinment.

The exported names are:
"""

include("HierarchicalRelations.jl")


"""
This (sub-)module provides functionality related with hierarchical refinment.

The exported names are:
"""

module HierarchicalFiniteElementSpaces

import .. FiniteElementSpaces

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
    struct HierarchicalFiniteElementSpace{n} <: FiniteElementSpaces.AbstractFiniteElementSpace{n}

A hierarchical space that is built from a nested hierarchy of `n`–variate function spaces. 

# Fields
- `spaces::Vector{FiniteElementSpaces.AbstractFiniteElementSpace{n}} `: collection of `n`–variate 
function spaces.
- `two_scale_operators::Vector{FiniteElementSpaces.TwoScaleOperator}`: collection of two–scale operators
relating each consequtive pair of function spaces.
- `active_elements::HierarchicalActiveInfo`: information about the active elements in each level.
- `active_functions::HierarchicalActiveInfo`: information about the active functions in each level.
"""
struct HierarchicalFiniteElementSpace{n} <: FiniteElementSpaces.AbstractFiniteElementSpace{n}
    spaces::Vector{FiniteElementSpaces.AbstractFiniteElementSpace{n}} 
    two_scale_operators::Vector{FiniteElementSpaces.TwoScaleOperator}
    active_elements::HierarchicalActiveInfo
    active_functions::HierarchicalActiveInfo

    function HierarchicalFiniteElementSpace(spaces::Vector{T}, two_scale_operators::Vector{FiniteElementSpaces.TwoScaleOperator}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo) where {n, T <: FiniteElementSpaces.AbstractFiniteElementSpace{n}}
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
    return get_n_active(hierarchical_space.elements)
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
    return get_n_active(hierarchical_space.functions)
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
    return get_n_levels(hierarchical_space.elements)
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
- `::FiniteElementSpaces.AbstractFiniteElementSpace{n}`: function space at `level`.
"""
function get_space(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return hierarchical_space.spaces[index]
end

"""
    get_level_active(active_info::HierarchicalActiveInfo, level::Int)

Returns the active objects of `active_info` at `level`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: information about active objects.
- `level::Int`: refinement level.
# Returns
- `::@view Vector{Int}`: active objects at `level`.
"""
function get_level_active(active_info::HierarchicalActiveInfo, level::Int)
    return @view active_info.ids[active_info.levels[level]+1:active_info.levels[level+1]]
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

end