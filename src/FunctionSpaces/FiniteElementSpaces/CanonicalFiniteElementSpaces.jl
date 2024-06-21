@doc raw"""
    CanonicalFiniteElementSpace <: AbstractFiniteElementSpace{1}

Wrapper that allows treating a canonical space as a finite element space.

# Fields
- `canonical_space::AbstractCanonicalSpace` : canonical space.
"""
struct CanonicalFiniteElementSpace{F} <: AbstractFiniteElementSpace{1}
    canonical_space::F
    boundary_dof_indices::Vector{Int}

    function CanonicalFiniteElementSpace(canonical_space::F) where F <: AbstractCanonicalSpace
        new{F}(canonical_space, [1, canonical_space.p+1])
    end
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

function get_local_basis(space::CanonicalFiniteElementSpace,::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int)
    return evaluate(space.canonical_space, xi[1], nderivatives)
end

function get_basis_indices(space::CanonicalFiniteElementSpace,::Int)
    return collect(1:get_dim(space))
end

function get_max_local_dim(space::CanonicalFiniteElementSpace)
    return get_dim(space)
end

function set_boundary_dof_indices(space::CanonicalFiniteElementSpace, indices::Vector{Int})
    space.boundary_dof_indices = indices
    return nothing
end

function get_boundary_dof_indices(space::CanonicalFiniteElementSpace)
    return space.boundary_dof_indices
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