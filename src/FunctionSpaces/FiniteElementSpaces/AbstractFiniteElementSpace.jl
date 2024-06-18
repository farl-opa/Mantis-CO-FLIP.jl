"""
    AbstractFiniteElementSpace

Supertype for all scalar finite element spaces.
"""
abstract type AbstractFiniteElementSpace{n} <: AbstractFunctionSpace end

# Getters for the function spaces
get_n(f::AbstractFiniteElementSpace{n}) where {n} = n

@doc raw"""
    evaluate(space::S, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {S<:AbstractFiniteElementSpace}

For given global element id `element_id` for a given finite element `space`, evaluate the local basis functions and return.

# Arguments 
- `space<:AbstractFiniteElementSpace`: finite element space.
- `element_id::Int`: global element id.
- `xi::Vector{Float64}`: vector of element-normalized points (i.e., in [0,1]) where basis needs to be evaluated.
- `nderivatives::Int`: number of derivatives to evaluate.

# Returns
- `::Array{Float64}`: array of evaluated global basis (size: num_eval_points x num_funcs x nderivatives+1)
- `::Vector{Int}`: vector of global basis indices (size: num_funcs).
"""
function evaluate(space::AbstractFiniteElementSpace{n}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n}
    extraction_coefficients, basis_indices = get_extraction(space, element_id)
    local_basis = get_local_basis(space, element_id, xi, nderivatives)

    for key ∈ keys(local_basis)     
        local_basis[key] = @views local_basis[key] * extraction_coefficients
    end

    return local_basis, basis_indices
end

function evaluate(space::AbstractFiniteElementSpace{n}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n}
    return evaluate(space, element_id, xi, 0)
end

function evaluate(space::AbstractFiniteElementSpace{n}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int, coeffs::Vector{Float64}) where {n}
    local_basis, basis_indices = evaluate(space, element_id, xi, nderivatives) 
    evaluation = copy(local_basis)
    
    for key ∈ keys(local_basis)
        evaluation[key] .= @views local_basis[key] * coeffs[basis_indices]
    end

    return evaluation
end