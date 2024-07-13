@doc raw"""
    CanonicalFiniteElementSpace <: AbstractFiniteElementSpace{1}

Wrapper that allows treating a canonical space as a finite element space.

# Fields
- `canonical_space::AbstractCanonicalSpace` : canonical space.
"""
struct CanonicalFiniteElementSpace{C} <: AbstractFiniteElementSpace{1}
    canonical_space::C
    boundary_dof_indices::Vector{Int}

    function CanonicalFiniteElementSpace(canonical_space::C) where C <: AbstractCanonicalSpace
        new{C}(canonical_space, [1, canonical_space.p+1])
    end
end

function get_polynomial_degree(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return space.canonical_space.p
end

function get_polynomial_degree(space::CanonicalFiniteElementSpace{C}, ::Int) where {C<: AbstractCanonicalSpace}
    return space.canonical_space.p
end

function get_num_elements(::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return 1
end

function get_dim(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return space.canonical_space.p+1
end

"""
_evaluate_all_at_point(canonical_space::CanonicalFiniteElementSpace{C}, xi::Float64, nderivatives::Int)

Returns the local basis functions for a `CanonicalFiniteElementSpace`.

# Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the local basis functions.
- `::Int`: Unused argument (placeholder).
- `xi::NTuple{1,Vector{Float64}}`: The coordinates at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to apply.

# Return Value
The local basis functions evaluated at `xi` with `nderivatives` derivatives.
"""
function _evaluate_all_at_point(space::CanonicalFiniteElementSpace{C}, xi::Float64, nderivatives::Int) where {C<: AbstractCanonicalSpace}
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
            V[count+1] = local_basis[r+1][1][1, i]
            count += 1
        end
    end
    return SparseArrays.sparse(I,J,V,ndofs,nderivatives+1)
end

function _evaluate_all_at_point(space::CanonicalFiniteElementSpace{C}, ::Int, xi::Float64, nderivatives::Int) where {C<: AbstractCanonicalSpace}
    return _evaluate_all_at_point(space, xi, nderivatives)
end

function get_local_basis(space::CanonicalFiniteElementSpace{C},::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int) where {C<: AbstractCanonicalSpace}
    return evaluate(space.canonical_space, xi[1], nderivatives)
end

function get_basis_indices(space::CanonicalFiniteElementSpace{C},::Int) where {C<: AbstractCanonicalSpace}
    return collect(1:get_dim(space))
end

function get_max_local_dim(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return get_dim(space)
end

function set_boundary_dof_indices(space::CanonicalFiniteElementSpace{C}, indices::Vector{Int}) where {C<: AbstractCanonicalSpace}
    space.boundary_dof_indices = indices
    return nothing
end

function get_boundary_dof_indices(space::CanonicalFiniteElementSpace{C}) where {C<: AbstractCanonicalSpace}
    return space.boundary_dof_indices
end

function get_extraction(space::CanonicalFiniteElementSpace{C},::Int) where {C<: AbstractCanonicalSpace}
    basis_indices = get_basis_indices(space, 1)
    nbasis = length(basis_indices)
    return Matrix(LinearAlgebra.I, nbasis, nbasis), basis_indices
end

function evaluate(space::CanonicalFiniteElementSpace{C}, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int) where {C<: AbstractCanonicalSpace}
    local_basis = get_local_basis(space, 1, xi, nderivatives)
    basis_indices = get_basis_indices(space, 1)

    return local_basis, basis_indices
end