@doc raw"""
    CanonicalFiniteElementSpace <: AbstractFiniteElementSpace{1}

Wrapper that allows treating a canonical space as a finite element space.

# Fields
- `canonical_space::AbstractCanonicalSpace` : canonical space.
"""
struct CanonicalFiniteElementSpace{F} <: AbstractFiniteElementSpace{1}
    canonical_space::F
end

function get_polynomial_degree(space::CanonicalFiniteElementSpace)
    return space.canonical_space.p
end

function get_polynomial_degree(space::CanonicalFiniteElementSpace, ::Int)
    return space.canonical_space.p
end

function get_num_elements(::CanonicalFiniteElementSpace)
    return 1
end

function get_dim(space::CanonicalFiniteElementSpace)
    return space.canonical_space.p+1
end

"""
_evaluate_all_at_point(canonical_space::CanonicalFiniteElementSpace, xi::Float64, nderivatives::Int)

Evaluates all derivatives upto order `nderivatives` for all `canonical_space` basis functions at a given point `xi`.

# Arguments
- `canonical_space::AbstractCanonicalSpace`: A canonical function space.
- `xi::Float64`: The point where all global basis functiuons are evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
# Returns
- `::SparseMatrixCSC{Float64}`: Global basis functions, size = n_dofs x nderivatives+1
"""
function _evaluate_all_at_point(space::CanonicalFiniteElementSpace, xi::Float64, nderivatives::Int)
    local_basis = evaluate(space.canonical_space, [xi], nderivatives)
    ndofs = get_dim(space)
    basis_indices = 1:ndofs
    nloc = length(basis_indices)
    I = zeros(Int, nloc * (nderivatives + 1))
    J = zeros(Int, nloc * (nderivatives + 1))
    V = zeros(Float64, nloc * (nderivatives + 1))
    count = 0
    for r = 0:nderivatives
        for i = 1:nloc
            I[count+1] = basis_indices[i]
            J[count+1] = r+1
            V[count+1] = local_basis[r][1, i]
            count += 1
        end
    end
    return SparseArrays.sparse(I,J,V,ndofs,nderivatives+1)
end

function _evaluate_all_at_point(space::CanonicalFiniteElementSpace, ::Int, xi::Float64, nderivatives::Int)
    return _evaluate_all_at_point(space, xi, nderivatives)
end

function get_local_basis(space::CanonicalFiniteElementSpace,::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int)
    return evaluate(space.canonical_space, xi[1], nderivatives)
end

function get_basis_indices(space::CanonicalFiniteElementSpace,::Int)
    return collect(1:get_dim(space))
end

function get_max_local_dim(space::CanonicalFiniteElementSpace)
    return get_dim(space)
end

function get_extraction(space::CanonicalFiniteElementSpace,::Int)
    basis_indices = get_basis_indices(space, 1)
    nbasis = length(basis_indices)
    return Matrix(LinearAlgebra.I, nbasis, nbasis), basis_indices
end

function evaluate(space::CanonicalFiniteElementSpace, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int)
    local_basis = get_local_basis(space, 1, xi, nderivatives)
    basis_indices = get_basis_indices(space, 1)

    return local_basis, basis_indices
end