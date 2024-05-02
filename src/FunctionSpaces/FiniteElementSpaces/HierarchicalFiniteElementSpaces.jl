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
    multilevel_elements::SparseArrays.SparseVector{Int, Int}
    multilevel_extraction_coeffs::Vector{Array{Float64, 2}}
    multilevel_basis_indices::Dict{Tuple{Int, Int}, Vector{Int}}

    function HierarchicalFiniteElementSpace(spaces::Vector{T}, two_scale_operators::Vector{TwoScaleOperator}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo, multilevel_elements::SparseArrays.SparseVector{Int, Int}, multilevel_extraction_coeffs::Vector{Array{Float64, 2}}, multilevel_basis_indices::Dict{Tuple{Int, Int}, Vector{Int}}) where {n, T <: AbstractFiniteElementSpace{n}}
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
end

# Getters for HierarchicalActiveInfo

function get_finer_active_elements(active_elements::HierarchicalActiveInfo, two_scale_operators::Vector{TwoScaleOperator}, element_id::Int, level::Int)
    element_ids = Int[]
    levels = zeros(Int, level+1)

    finer_active_elements = HierarchicalActiveInfo(element_ids, levels)

    if element_id ∈ get_level_active(active_elements, level)[2]
        add_active!(finer_active_elements, element_id, level)
        return finer_active_elements
    end
    
    elements = [element_id]
    next_elements = Int[]
    for l ∈ level:get_n_levels(active_elements)-1
        for element ∈ elements
            for finer_element ∈ two_scale_operators[l].coarse_to_fine_elements[element]
                if finer_element ∈ get_level_active(active_elements, l+1)[2]
                    add_active!(finer_active_elements, finer_element, l+1)
                end
            end
            append!(next_elements, two_scale_operators[l].coarse_to_fine_elements[element])
        end
        elements = next_elements
        next_elements = Int[]
    end

    return finer_active_elements
end

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
    get_num_elements(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}

Returns the number of active elements in `hierarchical_space`.

# Arguments 
- `hierarchical_space::HierarchicalFiniteElementSpace{n}`: Hierarchical function space.
# Returns
- `::Int`: number of active elements.
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
- `::Int`: number of active functions.
"""
function get_dim(hierarchical_space::HierarchicalFiniteElementSpace{n}) where {n}
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

function get_active_indices(active_info::HierarchicalActiveInfo, ids, level::Int)
    ids_range = active_info.levels[level]+1:active_info.levels[level+1]

    return ids_range[active_info.ids[ids_range] .∈ [ids]]
end

function get_active_extraction(hierarchical_space::HierarchicalFiniteElementSpace, element::Int)
    level = get_element_level(hierarchical_space, element)
    element_id = get_element_id(hierarchical_space, element)

    coeffs, basis_indices = get_extraction(get_space(hierarchical_space, level), element_id)

    active_indices = findall(x -> x ∈ get_level_active(hierarchical_space.active_functions, level)[2], basis_indices)

    return @views coeffs[:,active_indices], basis_indices[active_indices]
end


function get_active_extraction(space::S, level::Int, element_id::Int, active_functions::HierarchicalActiveInfo) where {S<:AbstractFiniteElementSpace{1}}

    coeffs, basis_indices = get_extraction(space, element_id)

    active_indices = findall(x -> x ∈ get_level_active(active_functions, level)[2], basis_indices)

    return @views coeffs[:,active_indices], basis_indices[active_indices]
end

function get_extraction(hierarchical_space::HierarchicalFiniteElementSpace, element::Int)
    level = get_element_level(hierarchical_space, element)
    element_id = get_element_id(hierarchical_space, element)

    level_space = get_space(hierarchical_space, level)

    coeffs, basis_indices = get_active_extraction(hierarchical_space, element)
    basis_indices = get_active_indices(hierarchical_space.active_functions, basis_indices, level)

    if length(hierarchical_space.multilevel_elements) == 0 || hierarchical_space.multilevel_elements[element] == 0
        return coeffs, basis_indices
    else
        coeffs = hierarchical_space.multilevel_extraction_coeffs[hierarchical_space.multilevel_elements[element]]
        
        return coeffs, append!(basis_indices, hierarchical_space.multilevel_basis_indices[(level, element_id)])
    end
end

function get_local_basis(hierarchical_space::HierarchicalFiniteElementSpace, element::Int, xi::Vector{Float64}, nderivatives::Int)
    level = get_element_level(hierarchical_space, element)
    element_id = get_element_id(hierarchical_space, element)
    level_space = get_space(hierarchical_space, level)

    return get_local_basis(level_space, element_id, xi, nderivatives)
end

function evaluate(hierarchical_space::HierarchicalFiniteElementSpace, element::Int, xi::Vector{Float64}, nderivatives::Int)
    extraction_coefficients, basis_indices = get_extraction(hierarchical_space, element)
    local_basis = get_local_basis(hierarchical_space, element, xi, nderivatives)

    for r = 0:nderivatives
        local_basis[r] = @views local_basis[r] * extraction_coefficients
    end

    return local_basis, basis_indices
end

# Updaters for the Hierarcical Active information

function remove_active!(active_info::HierarchicalActiveInfo, indices::Vector{Int}, level::Int)
    indices = sort!(unique(indices))
    deleteat!(active_info.ids, indices)
    active_info.levels[level+1] -= length(indices)

    return active_info
end

function add_active!(active_info::HierarchicalActiveInfo, id::Int, level::Int)
    current_levels = get_n_levels(active_info) 
    if level <= current_levels
        index = active_info.levels[level+1]+1
        insert!(active_info.ids, index, id)
        active_info.levels[level+1] += 1
    else
        append!(active_info.ids, id)
        append!(active_info.levels, zeros(Int, level-current_levels-1))
        append!(active_info.levels, active_info.levels[end]+1)
    end

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

function add_multilevel_elements!(columns::Vector{Int}, active_elements::HierarchicalActiveInfo, finer_elements::HierarchicalActiveInfo)
    for level ∈ 1:get_n_levels(finer_elements)
        finer_element_indices = get_active_indices(active_elements, get_level_active(finer_elements, level)[2], level)
        append!(columns, finer_element_indices)
    end

    return columns
end

function add_multilevel_functions!(multilevel_basis_indices::Dict{Tuple{Int, Int}, Vector{Int}}, finer_elements::HierarchicalActiveInfo, Ni::Int)
    for level ∈ 1:get_n_levels(finer_elements)
        for element ∈ get_level_active(finer_elements, level)[2]
            if haskey(multilevel_basis_indices, (level, element))
                append!(multilevel_basis_indices[(level, element)], Ni)
            else
                multilevel_basis_indices[(level, element)] = [Ni]
            end
        end
    end

    return multilevel_basis_indices
end

# Hierarchical space constructor

function get_active_objects(fe_spaces::Vector{T}, two_scale_operators::Vector{TwoScaleOperator}, refined_domains::HierarchicalActiveInfo) where {T<:AbstractFiniteElementSpace{n} where n}
    L = get_n_levels(refined_domains)

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
            support_check, support, finer_support = check_support(fe_spaces[level], Ni[i], next_level_domain, two_scale_operators[level]) # gets support in both levels
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

    return active_elements, active_functions
end

function get_multilevel_elements(fe_spaces, two_scale_operators, active_elements, active_functions)
    L = get_n_levels(active_elements)

    el_columns = Int[]
    multilevel_basis_indices = Dict{Tuple{Int, Int}, Vector{Int}}()

    for level in 1:L-1
        _, current_domain = get_level_active(active_elements, level)

        index, N = get_level_active(active_functions, level)
        for i ∈ 1:length(N) # Loop over active functions in level
            elements = FunctionSpaces.get_support(fe_spaces[level], N[i]) # support of the basis function
            
            _, element_checks =  Mesh.check_contained(elements, current_domain) # Check which elements are active

            for element ∈ elements[map(!,element_checks)] # loop over deactivated elements in the support
                finer_active_elements = get_finer_active_elements(active_elements, two_scale_operators, element, level)
                add_multilevel_elements!(el_columns, active_elements, finer_active_elements)
                add_multilevel_functions!(multilevel_basis_indices, finer_active_elements, index[i])
            end
        end
    end

    sort!(unique!(el_columns))

    multilevel_elements = SparseArrays.sparsevec(el_columns, 1:length(el_columns), get_n_active(active_elements))

    return multilevel_elements, multilevel_basis_indices
end

function get_multilevel_extraction(fe_spaces::Vector{T}, two_scale_operators::Vector{TwoScaleOperator}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo) where {T<:AbstractFiniteElementSpace{n} where n}
    multilevel_elements, multilevel_basis_indices = get_multilevel_elements(fe_spaces, two_scale_operators, active_elements, active_functions)

    multilevel_extraction_coeffs = Vector{Array{Float64, 2}}(undef,  length(multilevel_elements.nzind))

    # use multilevel elements and functions on those elements to get new extraction coeffs. a loop will go over every multilevel element, then another loop will go over every function on that multilevele element and add the function to the extraction operator of that multilevel element.
    
    for element ∈ multilevel_elements.nzind # loop over multilevel elements
        element_index = multilevel_elements[element]

        element_level = get_active_level(active_elements, element)
        element_id = get_active_id(active_elements, element)

        multilevel_extraction_coeffs[element_index] = get_active_extraction(fe_spaces[element_level], element_level, element_id, active_functions)[1]#copy extraction coeff of that element

        for Ni ∈ multilevel_basis_indices[(element_level, element_id)] # Loop over active functions on the element
            basis_level = get_active_level(active_functions, Ni)
            basis_id = get_active_id(active_functions, Ni)
            basis_element = get_coarser_element(two_scale_operators, basis_level, element_id, element_level)

            Ni_eval = get_finer_extraction_coeffs(fe_spaces[basis_level], two_scale_operators, basis_element, basis_level, element_id, element_level, basis_id) # Get evaluation of function on the element

            multilevel_extraction_coeffs[element_index] = hcat(multilevel_extraction_coeffs[element_index], Ni_eval)# Add evaluation to extraction coeffs of that element
        end
    end

    return multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices
end

function get_hierarchical_space(fe_spaces::Vector{T}, two_scale_operators::Vector{TwoScaleOperator}, refined_domains::HierarchicalActiveInfo) where {T<:AbstractFiniteElementSpace{n} where n}
    active_elements, active_functions = get_active_objects(fe_spaces, two_scale_operators, refined_domains)
    multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices = get_multilevel_extraction(fe_spaces, two_scale_operators, active_elements, active_functions)

    return HierarchicalFiniteElementSpace(fe_spaces, two_scale_operators, active_elements, active_functions, multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices)
end