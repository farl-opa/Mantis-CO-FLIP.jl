"""
    AbstractCanonicalSpace

Supertype for all element-local bases.
"""
abstract type AbstractCanonicalSpace <: AbstractFunctionSpace end

# Listed alphabetically
include("BernsteinPolynomials.jl")
include("LagrangePolynomials.jl")
include("ECTSpaces.jl")


# # Has to be below the include statements to ensure that all evaluate 
# # methods are visible to it!
# @doc raw"""
#     (elem_loc_basis::AbstractCanonicalSpace)(xi::Vector{Float64}, args...)::Matrix{Float64}

# Call the `evaluate`-method for the given `elem_loc_basis`.

# Wrapper for all `evaluate`-methods so that all `AbstractCanonicalSpace` can 
# be called by calling the struct instead of explicitly calling the 
# `evaluate`-method. Automatically throws a MethodError if the subtype 
# does not have an evaluate function implemented. Some of these methods 
# may have additional arguments, such as the additional derivatives to 
# evaluate, so this version should allow that as well.

# See also 
# - [`evaluate(elem_loc_basis::AbstractLagrangePolynomials, ξ::Vector{Float64})`](@ref),
# - [`evaluate(elem_loc_basis::Bernstein, xi::Vector{Float64}, nderivatives::Int64)`](@ref),
# - [`evaluate(elem_loc_basis::Bernstein, xi::Vector{Float64})`](@ref),
# - [`evaluate(elem_loc_basis::Bernstein, xi::Float64)`](@ref),
# - [`evaluate(elem_loc_basis::Bernstein, xi::Float64, nderivatives::Int64)`](@ref).
# """
function (elem_loc_basis::C where {C <: AbstractCanonicalSpace})(xi::Vector{Float64}, args...)
    return evaluate(elem_loc_basis, xi, args...)
end

function get_polynomial_degree(elem_loc_basis::AbstractCanonicalSpace)
    return elem_loc_basis.p
end

"""
    _evaluate_all_at_point(canonical_space::AbstractCanonicalSpace, xi::Float64, nderivatives::Int)

Evaluates all derivatives upto order `nderivatives` for all basis functions of `canonical_space` at a given point `xi`.

# Arguments
- `canonical_space::AbstractCanonicalSpace`: A canonical space.
- `xi::Float64`: The point where all global basis functiuons are evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
# Returns
- `::SparseMatrixCSC{Float64}`: Global basis functions, size = n_dofs x nderivatives+1
"""
function _evaluate_all_at_point(canonical_space::AbstractCanonicalSpace, xi::Float64, nderivatives::Int)
    local_basis = evaluate(canonical_space, [xi], nderivatives)
    ndofs = get_polynomial_degree(canonical_space)+1
    basis_indices = collect(1:ndofs)
    I = zeros(Int, ndofs * (nderivatives + 1))
    J = zeros(Int, ndofs * (nderivatives + 1))
    V = zeros(Float64, ndofs * (nderivatives + 1))
    count = 0
    for r = 0:nderivatives
        for i = 1:ndofs
            I[count+1] = basis_indices[i]
            J[count+1] = r+1
            V[count+1] = local_basis[r+1][1][1, i]
            count += 1
        end
    end

    return SparseArrays.sparse(I,J,V,ndofs,nderivatives+1)
end