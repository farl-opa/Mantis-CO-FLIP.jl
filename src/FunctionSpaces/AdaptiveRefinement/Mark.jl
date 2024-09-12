
@doc raw"""
    get_marked_basis_support(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, marked_basis) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the union of the support of all `marked_basis` in `level` as a vector of element indices. 

# Arguments
- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical space of the marked basis.
- `level::Int`: level of the marked basis functions.
- `marked_basis`: set of marked basis functions from `level`.

# Returns
- `::Vector{Int}`: union of the support of all `marked_basis`.
"""
function get_marked_basis_support(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, marked_basis) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return Vector{Int}(unique!(vcat(get_support.((get_space(hspace, level),), marked_basis)...)))
end


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