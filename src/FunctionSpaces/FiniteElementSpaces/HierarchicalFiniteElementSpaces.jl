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

function get_finer_active_elements(active_elements::HierarchicalActiveInfo, two_scale_operators::Vector{O}, element_id::Int, level::Int) where {O<:AbstractTwoScaleOperator}
    element_ids = Int[]
    levels = zeros(Int, level+1)

    finer_active_elements = HierarchicalActiveInfo(element_ids, levels)

    if element_id ∈ get_level_active(active_elements, level)[2]
        add_active!(finer_active_elements, element_id, level)
        return finer_active_elements
    end
    
    elements = [element_id]
    next_elements = Int[]
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

function get_active_indices(active_info::HierarchicalActiveInfo, ids, level::Int)
    ids_range = active_info.levels[level]+1:active_info.levels[level+1]

    return @view ids_range[active_info.ids[ids_range] .∈ [ids]]
end

function get_active_extraction(hierarchical_space::HierarchicalFiniteElementSpace{n}, element::Int) where {n}
    level = get_element_level(hierarchical_space, element)
    element_id = get_element_id(hierarchical_space, element)

    coeffs, basis_indices, _ = get_active_extraction(hierarchical_space.spaces[level], level, element_id, hierarchical_space.active_functions)
    return coeffs, basis_indices
end

function get_active_extraction(space::S, level::Int, element_id::Int, active_functions::HierarchicalActiveInfo) where {S <: AbstractFiniteElementSpace{n} where {n}}

    coeffs, basis_indices = get_extraction(space, element_id)

    active_indices = findall(x -> x ∈ get_level_active(active_functions, level)[2], basis_indices)
    active_basis_ids = basis_indices[active_indices]

    basis_indices = get_active_indices(active_functions, active_basis_ids, level)

    return coeffs, basis_indices, active_indices
end

function get_extraction(hierarchical_space::HierarchicalFiniteElementSpace{n}, element::Int) where {n}

    if length(hierarchical_space.multilevel_elements) == 0 || hierarchical_space.multilevel_elements[element] == 0
        coeffs, basis_indices = get_active_extraction(hierarchical_space, element)
    else
        multilevel_idx = hierarchical_space.multilevel_elements[element]
        coeffs = hierarchical_space.multilevel_extraction_coeffs[multilevel_idx]
        basis_indices = hierarchical_space.multilevel_basis_indices[multilevel_idx]
    end

    return coeffs, basis_indices
end

function get_local_basis(hierarchical_space::HierarchicalFiniteElementSpace{n}, element::Int, xi::Union{Vector{Float64}, NTuple{n, Vector{Float64}}}, nderivatives::Int) where {n}
    level = get_element_level(hierarchical_space, element)
    element_id = get_element_id(hierarchical_space, element)
    level_space = get_space(hierarchical_space, level)

    return get_local_basis(level_space, element_id, xi, nderivatives)
end

function get_coarse_refinement_matrix(size_extraction_coeffs::Tuple{Int, Int}, active_indices::Vector{Int})
    return Matrix{Float64}(LinearAlgebra.I, size_extraction_coeffs)[:,active_indices]
end

function truncate_refinement_matrix!(refinement_matrix::Union{Matrix{Float64}, SparseArrays.SparseMatrixCSC{Float64, Int}}, active_indices::Vector{Int})
    active_length = length(active_indices)
    refinement_matrix[active_indices, active_length+1:end] .= 0.0

    return refinement_matrix
end

# Updaters for the Hierarcical Active information

function remove_active!(active_info::HierarchicalActiveInfo, indices::Vector{Int}, level::Int)
    indices = sort!(unique(indices))
    deleteat!(active_info.ids, indices)
    active_info.levels[level+1] -= length(indices)

    return active_info
end

function add_active!(active_info::HierarchicalActiveInfo, id::Int, level::Int)
    current_levels = get_num_levels(active_info) 
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
    if level <= get_num_levels(active_info)
        index = active_info.levels[level+1]+1
        active_info.ids = [active_info.ids[1:index-1]; ids; active_info.ids[index:end]]
        active_info.levels[level+1:end] .+= length(ids)
    else
        append!(active_info.ids, ids)
        append!(active_info.levels, active_info.levels[end]+length(ids))
    end

    return active_info
end

function add_multilevel_elements!(columns::Vector{Int}, active_elements::HierarchicalActiveInfo, finer_elements::HierarchicalActiveInfo)
    for level ∈ 1:get_num_levels(finer_elements)
        finer_element_indices = get_active_indices(active_elements, get_level_active(finer_elements, level)[2], level)
        append!(columns, finer_element_indices)
    end

    return columns
end

function add_multilevel_functions!(multilevel_basis_indices::Dict{Tuple{Int, Int}, Vector{Int}}, finer_elements::HierarchicalActiveInfo, Ni::Int)
    for level ∈ 1:get_num_levels(finer_elements)
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

# getters for hierarchical space constructor


function get_active_objects(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, refined_domains::HierarchicalActiveInfo) where {O<:AbstractTwoScaleOperator, T<:AbstractFiniteElementSpace{n} where n}
    L = get_num_levels(refined_domains)

    # Initialize active functions and elements
    active_elements = HierarchicalActiveInfo(collect(1:get_num_elements(fe_spaces[1])),[0, get_num_elements(fe_spaces[1])])
    active_functions = HierarchicalActiveInfo(collect(1:get_dim(fe_spaces[1])),[0, get_dim(fe_spaces[1])])

    for level in 1:L-1 # Loop over levels
        _, next_level_domain = get_level_active(refined_domains, level+1)

        element_indices_to_remove = Int[]
        elements_to_add = Int[]
        function_indices_to_remove = Int[]
        functions_to_add = Int[]

        level_indices, level_basis = get_level_active(active_functions, level)
        for (i, Ni) ∈ zip(level_indices, level_basis)
            support_check, support = check_support(fe_spaces[level], Ni, next_level_domain, two_scale_operators[level]) # gets support in both levels
            if support_check
                union!(element_indices_to_remove, get_active_indices(active_elements, support, level))
                append!(function_indices_to_remove, i)
                union!(elements_to_add, get_finer_elements(two_scale_operators[level], support))
                union!(functions_to_add, get_finer_basis_id(two_scale_operators[level], Ni))
            end
        end

        # change way active functions are stored.

        remove_active!(active_elements, element_indices_to_remove, level)
        add_active!(active_elements, elements_to_add, level+1)
        remove_active!(active_functions, function_indices_to_remove, level)
        add_active!(active_functions, functions_to_add, level+1)
    end
    nothing
    return active_elements, active_functions
end

#=
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
        push!(active_elements_per_level, elements_to_add)
        push!(active_functions_per_level, functions_to_add)

    end

    active_elements = convert_element_vector_to_active_info(active_elements_per_level)
    active_functions = convert_element_vector_to_active_info(active_functions_per_level)
    
    return active_elements, active_functions
end

function get_inactive_active_children(active_elements, element_id, level, two_scale_operators)
    active_children = NTuple{2, Int}[]

    all_active_check = false
    current_element_ids = [element_id]
    while !all_active_check
        all_active_check = true
        inactive_children = Int[]
        for element_id ∈ current_element_ids 
            children = get_finer_elements(two_scale_operators[current_level], element_id)
            children_check = children .∈ get_level_active(active_elements, level+1)
            for child_id ∈ children[children_check]
                append!(active_children, (current_level+1, child_id))
            end
            append!(inactive_children, children[map(!, children_check)])
        all_active_check = all_active_check && all(children_check)
        end
        current_element_ids = inactive_children
        current_level += 1
    end

    return active_children
end

function get_multilevel_basis_indices(fe_spaces, two_scale_operators, active_elements, active_functions)
    L = get_num_levels(active_elements)
    multilevel_information = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    # Above, first tuple is level and id of ml element, second tuple
    # is level and id of ml basis in that element
    for level ∈ 1:L
        active_elements = [get_level_active(active_elements, level)[2]]
        _, active_basis = get_level_active(active_functions, level)

        for basis ∈ active_basis
            support = get_support(fe_spaces[level], basis)

            active_support_checks = support .∈ active_elements
            for inactive_element ∈ elements[map(!, active_support_checks)]
                active_children = get_inactive_active_children(active_elements, inactive_element, level, two_scale_operators)
                for (child_level, child_id) ∈ active_children
                    if haskey(multilevel_information, (level, inactive_element))
                        append!(multilevel_information[(level, inactive_element)], (child_level, child_id))
                    else
                        multilevel_information[(level, element)] = [(child_level, child_id)]
                    end
                end
            end
        end
    end

    return multilevel_information
end

function get_multilevel_extraction(fe_spaces, two_scale_operators, active_elements, active_functions, multilevel_information)
    for (level, element) ∈ keys(multilevel_information)

    end

    return multilevel_elements, multilevel_extraction_coeffs
end
=#

function get_multilevel_elements(fe_spaces, two_scale_operators, active_elements, active_functions)
    L = get_num_levels(active_elements)

    el_columns = Int[]
    multilevel_basis_indices_dic = Dict{Tuple{Int, Int}, Vector{Int}}()

    for level in 1:L-1
        _, current_domain = get_level_active(active_elements, level)

        index, N = get_level_active(active_functions, level)
        for (i, Ni) ∈ zip(index, N) # Loop over active functions in level
            elements = FunctionSpaces.get_support(fe_spaces[level], Ni) # support of the basis function
            
            _, element_checks =  Mesh.check_contained(elements, current_domain) # Check which elements are active

            for element ∈ elements[map(!,element_checks)] # loop over deactivated elements in the support
                finer_active_elements = get_finer_active_elements(active_elements, two_scale_operators, element, level)
                add_multilevel_elements!(el_columns, active_elements, finer_active_elements)
                add_multilevel_functions!(multilevel_basis_indices_dic, finer_active_elements, i)
            end
        end
    end

    sort!(unique!(el_columns))

    multilevel_elements = SparseArrays.sparsevec(el_columns, 1:length(el_columns), get_num_active(active_elements))

    return multilevel_elements, multilevel_basis_indices_dic
end


function get_multilevel_extraction(fe_spaces::Vector{T}, two_scale_operators::Vector{O}, active_elements::HierarchicalActiveInfo, active_functions::HierarchicalActiveInfo, truncated::Bool) where {T<:AbstractFiniteElementSpace{n} where n, O<:AbstractTwoScaleOperator}
    multilevel_elements, multilevel_basis_indices_dic = get_multilevel_elements(fe_spaces, two_scale_operators, active_elements, active_functions)

    n_multilevel_els = length(multilevel_elements.nzind)
    multilevel_extraction_coeffs = Vector{Array{Float64, 2}}(undef, n_multilevel_els)
    multilevel_basis_indices = Vector{Vector{Int}}(undef, n_multilevel_els)

    # use multilevel elements and functions on those elements to get new extraction coeffs. a loop will go over every multilevel element, then another loop will go over every function on that multilevele element and add the function to the extraction operator of that multilevel element.
    
    for element ∈ multilevel_elements.nzind # loop over multilevel elements
        element_index = multilevel_elements[element]

        element_level = get_active_level(active_elements, element)
        element_id = get_active_id(active_elements, element)

        multilevel_extraction_coeffs[element_index], multilevel_basis_indices[element_index], active_indices = get_active_extraction(fe_spaces[element_level], element_level, element_id, active_functions)#copy extraction of current element with active basis indices

        current_level_matrix = get_coarse_refinement_matrix(size(multilevel_extraction_coeffs[element_index]), active_indices)
        refinement_matrix = Matrix{Float64}(undef, (size(current_level_matrix)[1], length(multilevel_basis_indices_dic[(element_level, element_id)])))

        for (ml_count, Ni) ∈ enumerate(multilevel_basis_indices_dic[(element_level, element_id)]) # Loop over active functions on the element
            basis_level = get_active_level(active_functions, Ni)
            basis_id = get_active_id(active_functions, Ni)
            basis_element = get_coarser_element(two_scale_operators, basis_level, element_id, element_level)

            refinement_matrix = update_refinement_matrix!(refinement_matrix, two_scale_operators, basis_element, basis_level,basis_id, element_id, element_level, ml_count, active_functions, truncated) # Get current ml function refinement
        end
        if truncated
            refinement_matrix = truncate_refinement_matrix!(refinement_matrix, active_indices)
        end
        refinement_matrix = hcat(current_level_matrix, refinement_matrix)
        multilevel_extraction_coeffs[element_index] = multilevel_extraction_coeffs[element_index] * refinement_matrix
        append!(multilevel_basis_indices[element_index], multilevel_basis_indices_dic[(element_level, element_id)])
    end

    return multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices
end

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
