"""
    HierarchicalActiveInfo

A structure that contains hierarchical information of active objects.

# Fields
- `ids::Vector{Int}`: The index of the object in its corresponding structure.
- `levels::Vector{Int}`: Level information of active objects.

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

"""
    get_finer_active_elements(active_elements::HierarchicalActiveInfo, two_scale_operators::Vector{O}, element_id::Int, level::Int) where {O<:AbstractTwoScaleOperator}

Retrieve the finer active elements for a given element at a specific level.

# Arguments
- `active_elements::HierarchicalActiveInfo`: Information about active elements.
- `two_scale_operators::Vector{O}`: Collection of two-scale operators.
- `element_id::Int`: ID of the element.
- `level::Int`: Level of the element.

# Returns
- `::HierarchicalActiveInfo`: Hierarchical information of finer active elements.
"""
function get_finer_active_elements(active_elements::HierarchicalActiveInfo, two_scale_operators::Vector{O}, element_id::Int, level::Int) where {O<:AbstractTwoScaleOperator}
    # Initialize vectors to store finer element ids and their levels
    element_ids = Int[]
    levels = zeros(Int, level+1)

    # Create a HierarchicalActiveInfo object to store finer active elements
    finer_active_elements = HierarchicalActiveInfo(element_ids, levels)

    # Check if the element is active at the given level
    if element_id ∈ get_level_active(active_elements, level)[2]
        add_active!(finer_active_elements, element_id, level)
        return finer_active_elements
    end
    
    # Initialize the list of elements to process
    elements = [element_id]
    next_elements = Int[]
    
    # Iterate through levels to find finer elements
    for l ∈ level:get_num_levels(active_elements)-1
        for element ∈ elements, finer_element ∈ get_finer_elements(two_scale_operators[l], element)
            if finer_element ∈ get_level_active(active_elements, l+1)[2]
                add_active!(finer_active_elements, finer_element, l+1)
            else
                append!(next_elements, finer_element)
            end
        end
        elements = next_elements
        next_elements = Int[]
    end

    return finer_active_elements
end

"""
    HierarchicalFiniteElementSpace{n} <: AbstractFiniteElementSpace{n}

A hierarchical space that is built from a nested hierarchy of `n`–variate function spaces.

# Fields
- `spaces::Vector{AbstractFiniteElementSpace{n}}`: Collection of `n`–variate function spaces.
- `two_scale_operators::Vector{O} where {O<:AbstractTwoScaleOperator}`: Collection of two–scale operators
  relating each consecutive pair of function spaces.
- `active_elements::HierarchicalActiveInfo`: Information about the active elements in each level.
- `active_functions::HierarchicalActiveInfo`: Information about the active functions in each level.
- `multilevel_elements::SparseArrays.SparseVector{Int, Int}`: Sparse vector representing multilevel elements.
- `multilevel_extraction_coeffs::Vector{Array{Float64, 2}}`: Extraction coefficients for multilevel basis.
- `multilevel_basis_indices::Vector{Vector{Int}}`: Indices of multilevel basis functions.
"""
struct HierarchicalFiniteElementSpace{n} <: AbstractFiniteElementSpace{n}
    spaces::Vector{AbstractFiniteElementSpace{n}} 
    two_scale_operators::Vector{O} where {O<:AbstractTwoScaleOperator}
    active_elements::HierarchicalActiveInfo
    active_functions::HierarchicalActiveInfo
    multilevel_elements::SparseArrays.SparseVector{Int, Int}
    multilevel_extraction_coeffs::Vector{Array{Float64, 2}}
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

    function HierarchicalFiniteElementSpace(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, refined_domains::HierarchicalActiveInfo) where {T<:AbstractFiniteElementSpace{n} where n, O<:AbstractTwoScaleOperator}
        # Compute active elements and functions based on refined domains
        active_elements, active_functions = get_active_objects(fe_spaces, two_scale_operators, refined_domains)
        
        # Compute multilevel extraction information
        multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices = get_multilevel_extraction(fe_spaces, two_scale_operators, active_elements, active_functions)

        return HierarchicalFiniteElementSpace(fe_spaces, two_scale_operators, active_elements, active_functions, multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices)
    end
end

"""
    get_n_active(active_info::HierarchicalActiveInfo)

Returns the number of active objects in `active_info`.

# Arguments 
- `active_info::HierarchicalActiveInfo`: Information about active objects.

# Returns
- `::Int`: Number of active objects.
"""
function get_n_active(active_info::HierarchicalActiveInfo)
    return length(active_info.ids)
end

"""
    get_num_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of active elements in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.

# Returns
- `::Int`: Number of active elements.
"""
function get_num_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return get_n_active(hierarchical_space.active_elements)
end

"""
    get_dim(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of active functions in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.

# Returns
- `::Int`: Number of active functions.
"""
function get_dim(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return get_n_active(hierarchical_space.active_functions)
end

"""
    get_max_local_dim(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the maximum local dimension of the hierarchical space.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.

# Returns
- `::Int`: Maximum local dimension.
"""
function get_max_local_dim(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
    return 27
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
    get_num_levels(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of levels in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.

# Returns
- `::Int`: Number of levels.
"""
function get_num_levels(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
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
    get_element_level(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}

Returns the level of the element given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `index::Int`: Index of the active element.

# Returns
- `::Int`: Level of the element.
"""
function get_element_level(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return get_active_level(hierarchical_space.active_elements, index)
end

"""
    get_function_level(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}

Returns the level of the function given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `index::Int`: Index of the active function.

# Returns
- `::Int`: Level of the function.
"""
function get_function_level(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return get_active_level(hierarchical_space.active_functions, index)
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
    get_element_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}

Returns the corresponding id of the element given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `index::Int`: Index of the active element.

# Returns
- `::Int`: ID of the active element.
"""
function get_element_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return get_active_id(hierarchical_space.active_elements, index)
end

"""
    get_function_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}

Returns the corresponding id of the function given by `index`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `index::Int`: Index of the active function.

# Returns
- `::Int`: ID of the active function.
"""
function get_function_id(hierarchical_space::HierarchicalFiniteElementSpace{n}, index::Int) where {n}
    return get_active_id(hierarchical_space.active_functions, index)
end

"""
    get_space(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}

Returns the function space at the specified `level` from the hierarchical space.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `level::Int`: Refinement level.

# Returns
- `::AbstractFiniteElementSpace{n}`: Function space at the specified `level`.

# Notes
- This function assumes that the spaces are stored in an array-like structure within the `hierarchical_space`.
- The `level` is used as an index to access the corresponding space.
"""
function get_space(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    # Access and return the space at the specified level
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
    get_level_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}

Returns the active elements of `hierarchical_space` at the specified `level`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `level::Int`: Refinement level.

# Returns
- `::Tuple{UnitRange{Int}, SubArray{Int,1}}`: A tuple containing:
  1. A `UnitRange` of indices for the active elements at the specified level.
  2. A view of the active element IDs for the specified level.

# Notes
- This function delegates to `get_level_active` using the `active_elements` field of the hierarchical space.
"""
function get_level_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    # Delegate to get_level_active using the active_elements field
    return get_level_active(hierarchical_space.active_elements, level)
end

"""
    get_level_functions(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}

Returns the active functions of `hierarchical_space` at the specified `level`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
- `level::Int`: Refinement level.

# Returns
- `::Tuple{UnitRange{Int}, SubArray{Int,1}}`: A tuple containing:
  1. A `UnitRange` of indices for the active functions at the specified level.
  2. A view of the active function IDs for the specified level.

# Notes
- This function delegates to `get_level_active` using the `active_functions` field of the hierarchical space.
"""
function get_level_functions(hierarchical_space::HierarchicalFiniteElementSpace{n}, level::Int) where {n}
    # Delegate to get_level_active using the active_functions field
    return get_level_active(hierarchical_space.active_functions, level)
end

"""
    get_active_indices(active_info::HierarchicalActiveInfo, ids, level::Int)

Returns the indices of active objects at the specified `level` that match the given `ids`.

# Arguments
- `active_info::HierarchicalActiveInfo`: Information about active objects.
- `ids`: A collection of IDs to match against.
- `level::Int`: Refinement level.

# Returns
- `::Vector{Int}`: Indices of active objects at the specified level that match the given IDs.

# Notes
- This function first determines the range of indices for the given level.
- It then filters these indices based on whether their corresponding IDs are in the provided `ids` collection.
"""
function get_active_indices(active_info::HierarchicalActiveInfo, ids, level::Int)
    # Determine the range of indices for the given level
    ids_range = active_info.levels[level]+1:active_info.levels[level+1]

    # Filter and return the indices whose IDs are in the provided ids collection
    return ids_range[active_info.ids[ids_range] .∈ [ids]]
end

"""
    get_active_extraction(hierarchical_space::HierarchicalFiniteElementSpace, element::Int)

Returns the active extraction coefficients and basis indices for a given element in the hierarchical space.

# Arguments
- `hierarchical_space::HierarchicalFiniteElementSpace`: The hierarchical finite element space.
- `element::Int`: The element index.

# Returns
- `::Tuple{Matrix{Float64}, Vector{Int}}`: A tuple containing:
  1. The active extraction coefficients.
  2. The corresponding basis indices.

# Notes
- This function first determines the level and ID of the given element.
- It then delegates to a more specific `get_active_extraction` method using this information.
"""
function get_active_extraction(hierarchical_space::HierarchicalFiniteElementSpace, element::Int)
    # Determine the level and ID of the given element
    level = get_element_level(hierarchical_space, element)
    element_id = get_element_id(hierarchical_space, element)

    # Delegate to the specific get_active_extraction method
    return get_active_extraction(hierarchical_space.spaces[level], level, element_id, hierarchical_space.active_functions)
end

"""
    get_active_extraction(space::S, level::Int, element_id::Int, active_functions::HierarchicalActiveInfo) where {S<:AbstractFiniteElementSpace}

Returns the active extraction coefficients and basis indices for a given element in a specific space.

# Arguments
- `space::S`: The finite element space.
- `level::Int`: The refinement level.
- `element_id::Int`: The element ID.
- `active_functions::HierarchicalActiveInfo`: Information about active functions.

# Returns
- `::Tuple{SubArray{Float64,2}, Vector{Int}}`: A tuple containing:
  1. A view of the active extraction coefficients.
  2. The corresponding active basis indices.

# Notes
- This function first gets the full extraction coefficients and basis indices for the element.
- It then filters these based on the active functions at the given level.
"""
function get_active_extraction(space::S, level::Int, element_id::Int, active_functions::HierarchicalActiveInfo) where {S<:AbstractFiniteElementSpace}
    # Get full extraction coefficients and basis indices
    coeffs, basis_indices = get_extraction(space, element_id)

    # Determine which basis indices are active at this level
    active_indices = findall(x -> x ∈ get_level_active(active_functions, level)[2], basis_indices)

    # Return views of the active coefficients and basis indices
    return @views coeffs[:,active_indices], basis_indices[active_indices]
end

"""
    get_extraction(hierarchical_space::HierarchicalFiniteElementSpace, element::Int)

Returns the extraction coefficients and basis indices for a given element in the hierarchical space.

# Arguments
- `hierarchical_space::HierarchicalFiniteElementSpace`: The hierarchical finite element space.
- `element::Int`: The element index.

# Returns
- `::Tuple{Matrix{Float64}, Vector{Int}}`: A tuple containing:
  1. The extraction coefficients.
  2. The corresponding basis indices.

# Notes
- This function handles both single-level and multi-level elements.
- For single-level elements, it uses `get_active_extraction`.
- For multi-level elements, it directly accesses pre-computed data.
"""
function get_extraction(hierarchical_space::HierarchicalFiniteElementSpace, element::Int)
    if length(hierarchical_space.multilevel_elements) == 0 || hierarchical_space.multilevel_elements[element] == 0
        # Handle single-level elements
        coeffs, basis_indices = get_active_extraction(hierarchical_space, element)
        level = get_element_level(hierarchical_space, element)
        basis_indices = get_active_indices(hierarchical_space.active_functions, basis_indices, level)
        
        return coeffs, basis_indices
    else
        # Handle multi-level elements
        multilevel_idx = hierarchical_space.multilevel_elements[element]
        coeffs = hierarchical_space.multilevel_extraction_coeffs[multilevel_idx]
        basis_indices = hierarchical_space.multilevel_basis_indices[multilevel_idx]
        
        return coeffs, basis_indices
    end
end

"""
    get_local_basis(hierarchical_space::HierarchicalFiniteElementSpace, element::Int, xi::Union{Vector{Float64}, NTuple{n, Vector{Float64}}}, nderivatives::Int) where {n}

Computes the local basis functions and their derivatives for a given element at specified parametric coordinates.

# Arguments
- `hierarchical_space::HierarchicalFiniteElementSpace`: The hierarchical finite element space.
- `element::Int`: The element index.
- `xi::Union{Vector{Float64}, NTuple{n, Vector{Float64}}}`: Parametric coordinates.
- `nderivatives::Int`: Number of derivatives to compute.

# Returns
- The local basis functions and their derivatives (return type depends on the specific space implementation).

# Notes
- This function first determines the level and ID of the given element.
- It then delegates to the `get_local_basis` method of the specific space at that level.
"""
function get_local_basis(hierarchical_space::HierarchicalFiniteElementSpace, element::Int, xi::Union{Vector{Float64}, NTuple{n, Vector{Float64}}}, nderivatives::Int) where {n}
    # Determine the level and ID of the given element
    level = get_element_level(hierarchical_space, element)
    element_id = get_element_id(hierarchical_space, element)
    
    # Get the space at the determined level
    level_space = get_space(hierarchical_space, level)

    # Delegate to the get_local_basis method of the specific space
    return get_local_basis(level_space, element_id, xi, nderivatives)
end


# Updaters for the Hierarcical Active information

"""
    remove_active!(active_info::HierarchicalActiveInfo, indices::Vector{Int}, level::Int)

Remove active objects at the specified `level` from the `HierarchicalActiveInfo` structure.

# Arguments
- `active_info::HierarchicalActiveInfo`: The hierarchical active information structure.
- `indices::Vector{Int}`: Indices of objects to remove.
- `level::Int`: The level from which to remove objects.

# Returns
- `::HierarchicalActiveInfo`: The modified `active_info` structure.

# Notes
- This function modifies the `active_info` in-place.
- The `indices` are sorted and made unique before processing.
"""
function remove_active!(active_info::HierarchicalActiveInfo, indices::Vector{Int}, level::Int)
    # Sort and remove duplicates from indices
    indices = sort!(unique(indices))
    
    # Remove the specified ids
    deleteat!(active_info.ids, indices)
    
    # Update the level information
    active_info.levels[level+1] -= length(indices)

    return active_info
end

"""
    add_active!(active_info::HierarchicalActiveInfo, id::Int, level::Int)

Add a single active object at the specified `level` to the `HierarchicalActiveInfo` structure.

# Arguments
- `active_info::HierarchicalActiveInfo`: The hierarchical active information structure.
- `id::Int`: The id of the object to add.
- `level::Int`: The level at which to add the object.

# Returns
- `::HierarchicalActiveInfo`: The modified `active_info` structure.

# Notes
- This function modifies the `active_info` in-place.
- If the specified level is greater than the current number of levels, new levels are added as needed.
"""
function add_active!(active_info::HierarchicalActiveInfo, id::Int, level::Int)
    current_levels = get_num_levels(active_info) 
    
    if level <= current_levels
        # Insert the new id at the appropriate position
        index = active_info.levels[level+1] + 1
        insert!(active_info.ids, index, id)
        active_info.levels[level+1] += 1
    else
        # Add new levels if necessary
        append!(active_info.ids, id)
        append!(active_info.levels, zeros(Int, level - current_levels - 1))
        append!(active_info.levels, active_info.levels[end] + 1)
    end

    return active_info
end

"""
    add_active!(active_info::HierarchicalActiveInfo, ids::Vector{Int}, level::Int)

Add multiple active objects at the specified `level` to the `HierarchicalActiveInfo` structure.

# Arguments
- `active_info::HierarchicalActiveInfo`: The hierarchical active information structure.
- `ids::Vector{Int}`: The ids of the objects to add.
- `level::Int`: The level at which to add the objects.

# Returns
- `::HierarchicalActiveInfo`: The modified `active_info` structure.

# Notes
- This function modifies the `active_info` in-place.
- The `ids` are sorted and made unique before processing.
- If the specified level is greater than the current number of levels, a new level is added.
"""
function add_active!(active_info::HierarchicalActiveInfo, ids::Vector{Int}, level::Int)
    # Sort and remove duplicates from ids
    ids = sort!(unique(ids))
    
    if level <= get_num_levels(active_info)
        # Insert the new ids at the appropriate position
        index = active_info.levels[level+1] + 1
        active_info.ids = [active_info.ids[1:index-1]; ids; active_info.ids[index:end]]
        active_info.levels[level+1] += length(ids)
    else
        # Add a new level if necessary
        append!(active_info.ids, ids)
        append!(active_info.levels, active_info.levels[end] + length(ids))
    end

    return active_info
end

"""
    add_multilevel_elements!(columns::Vector{Int}, active_elements::HierarchicalActiveInfo, finer_elements::HierarchicalActiveInfo)

Add multilevel elements to the `columns` vector based on the `active_elements` and `finer_elements`.

# Arguments
- `columns::Vector{Int}`: The vector to which multilevel elements will be added.
- `active_elements::HierarchicalActiveInfo`: The active elements information.
- `finer_elements::HierarchicalActiveInfo`: The finer elements information.

# Returns
- `::Vector{Int}`: The modified `columns` vector.

# Notes
- This function modifies the `columns` vector in-place.
- It iterates through all levels of `finer_elements` and adds the corresponding active indices.
"""
function add_multilevel_elements!(columns::Vector{Int}, active_elements::HierarchicalActiveInfo, finer_elements::HierarchicalActiveInfo)
    for level ∈ 1:get_num_levels(finer_elements)
        # Get active indices for the current level
        finer_element_indices = get_active_indices(active_elements, get_level_active(finer_elements, level)[2], level)
        append!(columns, finer_element_indices)
    end

    return columns
end

"""
    add_multilevel_functions!(multilevel_basis_indices::Dict{Tuple{Int, Int}, Vector{Int}}, finer_elements::HierarchicalActiveInfo, Ni::Int)

Add multilevel functions to the `multilevel_basis_indices` dictionary based on the `finer_elements`.

# Arguments
- `multilevel_basis_indices::Dict{Tuple{Int, Int}, Vector{Int}}`: Dictionary to store multilevel basis indices.
- `finer_elements::HierarchicalActiveInfo`: The finer elements information.
- `Ni::Int`: The function index to be added.

# Returns
- `::Dict{Tuple{Int, Int}, Vector{Int}}`: The modified `multilevel_basis_indices` dictionary.

# Notes
- This function modifies the `multilevel_basis_indices` dictionary in-place.
- It iterates through all levels and active elements of `finer_elements`, adding `Ni` to the corresponding entries.
"""
function add_multilevel_functions!(multilevel_basis_indices::Dict{Tuple{Int, Int}, Vector{Int}}, finer_elements::HierarchicalActiveInfo, Ni::Int)
    for level ∈ 1:get_num_levels(finer_elements)
        for element ∈ get_level_active(finer_elements, level)[2]
            if haskey(multilevel_basis_indices, (level, element))
                push!(multilevel_basis_indices[(level, element)], Ni)
            else
                multilevel_basis_indices[(level, element)] = [Ni]
            end
        end
    end

    return multilevel_basis_indices
end

# getters for hierarchical space constructor

"""
    get_active_objects(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, refined_domains::HierarchicalActiveInfo) where {O<:AbstractTwoScaleOperator, T<:AbstractFiniteElementSpace{n} where n}

Determine the active elements and functions in a hierarchical finite element space.

# Arguments
- `fe_spaces::Vector{T}`: Vector of finite element spaces.
- `two_scale_operators::Vector{O}`: Vector of two-scale operators.
- `refined_domains::HierarchicalActiveInfo`: Information about refined domains.

# Returns
- `::Tuple{HierarchicalActiveInfo, HierarchicalActiveInfo}`: A tuple containing:
  1. The active elements information.
  2. The active functions information.

# Notes
- This function initializes the active elements and functions based on the first finite element space.
- It iterates through each level, updating the active elements and functions based on the support of functions in the next level.
"""
function get_active_objects(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, refined_domains::HierarchicalActiveInfo) where {O<:AbstractTwoScaleOperator, T<:AbstractFiniteElementSpace{n} where n}
    L = get_num_levels(refined_domains)

    # Initialize active functions and elements
    active_elements = HierarchicalActiveInfo(collect(1:get_num_elements(fe_spaces[1])), [0, get_num_elements(fe_spaces[1])])
    active_functions = HierarchicalActiveInfo(collect(1:get_dim(fe_spaces[1])), [0, get_dim(fe_spaces[1])])

    for level in 1:L-1 # Loop over levels
        _, next_level_domain = get_level_active(refined_domains, level+1)

        element_indices_to_remove = Int[]
        elements_to_add = Int[]
        function_indices_to_remove = Int[]
        functions_to_add = Int[]

        index, Ni = get_level_active(active_functions, level)
        for i in 1:length(Ni) # Loop over active functions in level
            support_check, support, finer_support = check_support(fe_spaces[level], Ni[i], next_level_domain, two_scale_operators[level]) # gets support in both levels
            if support_check
                append!(element_indices_to_remove, get_active_indices(active_elements, support, level))
                append!(elements_to_add, finer_support)
                append!(function_indices_to_remove, index[i])
                append!(functions_to_add, get_finer_basis_id(two_scale_operators[level], Ni[i]))
            end
        end

        remove_active!(active_elements, element_indices_to_remove, level)
        add_active!(active_elements, elements_to_add, level+1)
        remove_active!(active_functions, function_indices_to_remove, level)
        add_active!(active_functions, functions_to_add, level+1)
    end

    return active_elements, active_functions
end

"""
    get_multilevel_elements(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo) where {T<:AbstractFiniteElementSpace{n} where n, O<:AbstractTwoScaleOperator}

Determine the multilevel elements and their corresponding basis indices.

# Arguments
- `fe_spaces::Vector{T}`: Vector of finite element spaces.
- `two_scale_operators::Vector{O}`: Vector of two-scale operators.
- `active_elements::HierarchicalActiveInfo`: Information about active elements.
- `active_functions::HierarchicalActiveInfo`: Information about active functions.

# Returns
- `::Tuple{SparseVector{Int}, Dict{Tuple{Int, Int}, Vector{Int}}}`: A tuple containing:
  1. A sparse vector representing multilevel elements.
  2. A dictionary mapping element levels and IDs to basis indices.

# Notes
- This function iterates through each level, updating the multilevel elements and basis indices based on the support of functions.
"""
function get_multilevel_elements(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo) where {T<:AbstractFiniteElementSpace{n} where n, O<:AbstractTwoScaleOperator}
    L = get_num_levels(active_elements)

    el_columns = Int[]
    multilevel_basis_indices_dic = Dict{Tuple{Int, Int}, Vector{Int}}()

    for level in 1:L-1
        _, current_domain = get_level_active(active_elements, level)

        index, N = get_level_active(active_functions, level)
        for i ∈ 1:length(N) # Loop over active functions in level
            elements = FunctionSpaces.get_support(fe_spaces[level], N[i]) # support of the basis function
            
            _, element_checks = Mesh.check_contained(elements, current_domain) # Check which elements are active

            for element ∈ elements[map(!,element_checks)] # loop over deactivated elements in the support
                finer_active_elements = get_finer_active_elements(active_elements, two_scale_operators, element, level)
                add_multilevel_elements!(el_columns, active_elements, finer_active_elements)
                add_multilevel_functions!(multilevel_basis_indices_dic, finer_active_elements, index[i])
            end
        end
    end

    sort!(unique!(el_columns))

    multilevel_elements = SparseArrays.sparsevec(el_columns, 1:length(el_columns), get_n_active(active_elements))

    return multilevel_elements, multilevel_basis_indices_dic
end

"""
    get_multilevel_extraction(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo) where {T<:AbstractFiniteElementSpace{n} where n, O<:AbstractTwoScaleOperator}

Compute the multilevel extraction coefficients and basis indices.

# Arguments
- `fe_spaces::Vector{T}`: Vector of finite element spaces.
- `two_scale_operators::Vector{O}`: Vector of two-scale operators.
- `active_elements::HierarchicalActiveInfo`: Information about active elements.
- `active_functions::HierarchicalActiveInfo`: Information about active functions.

# Returns
- `::Tuple{SparseVector{Int}, Vector{Array{Float64, 2}}, Vector{Vector{Int}}}`: A tuple containing:
  1. A sparse vector representing multilevel elements.
  2. A vector of arrays representing multilevel extraction coefficients.
  3. A vector of vectors representing multilevel basis indices.

# Notes
- This function iterates through each multilevel element, computing the extraction coefficients and basis indices.
- It uses the multilevel elements and functions to build the extraction operators for each multilevel element.
"""
function get_multilevel_extraction(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo) where {T<:AbstractFiniteElementSpace{n} where n, O<:AbstractTwoScaleOperator}
    multilevel_elements, multilevel_basis_indices_dic = get_multilevel_elements(fe_spaces, two_scale_operators, active_elements, active_functions)

    n_multilevel_els = length(multilevel_elements.nzind)
    multilevel_extraction_coeffs = Vector{Array{Float64, 2}}(undef, n_multilevel_els)
    multilevel_basis_indices = Vector{Vector{Int}}(undef, n_multilevel_els)

    # Use multilevel elements and functions on those elements to get new extraction coeffs.
    for element ∈ multilevel_elements.nzind # loop over multilevel elements
        element_index = multilevel_elements[element]

        element_level = get_active_level(active_elements, element)
        element_id = get_active_id(active_elements, element)

        # Copy extraction of current element
        multilevel_extraction_coeffs[element_index], multilevel_basis_indices[element_index] = get_active_extraction(fe_spaces[element_level], element_level, element_id, active_functions)

        # Update basis indices
        multilevel_basis_indices[element_index] = get_active_indices(active_functions, multilevel_basis_indices[element_index], element_level)

        for Ni ∈ multilevel_basis_indices_dic[(element_level, element_id)] # Loop over active functions on the element
            basis_level = get_active_level(active_functions, Ni)
            basis_id = get_active_id(active_functions, Ni)
            basis_element = get_coarser_element(two_scale_operators, basis_level, element_id, element_level)

            # Get evaluation of function on the element
            Ni_eval = get_finer_extraction_coeffs(fe_spaces[basis_level], two_scale_operators, basis_element, basis_level, element_id, element_level, basis_id)

            # Add evaluation to extraction coeffs of that element
            multilevel_extraction_coeffs[element_index] = hcat(multilevel_extraction_coeffs[element_index], Ni_eval)
        end
        append!(multilevel_basis_indices[element_index], multilevel_basis_indices_dic[(element_level, element_id)])
    end

    return multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices
end

