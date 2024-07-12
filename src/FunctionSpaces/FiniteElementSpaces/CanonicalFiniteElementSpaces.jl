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

@doc raw"""
    get_polynomial_degree(space::CanonicalFiniteElementSpace)

# Arguments
- `space::CanonicalFiniteElementSpace`: The CanonicalFiniteElementSpace object for which to retrieve the polynomial degree.

# Return Value
The polynomial degree p of the CanonicalFiniteElementSpace.
"""
function get_polynomial_degree(space::CanonicalFiniteElementSpace)
    return space.canonical_space.p
end

@doc raw"""
    get_polynomial_degree(space::CanonicalFiniteElementSpace, ::Int)

# Arguments    
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the polynomial degree.
- `::Int`: Unused argument (placeholder).
    
# Return Value
- The polynomial degree `p` of the `CanonicalFiniteElementSpace`.
"""
function get_polynomial_degree(space::CanonicalFiniteElementSpace, ::Int)
    return space.canonical_space.p
end

@doc raw"""
    get_num_elements(::CanonicalFiniteElementSpace)

Returns the number of elements in a `CanonicalFiniteElementSpace`.
    
# Arguments
- `::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the number of elements.

# Return Value
The number of elements, which is always 1 for this function.
"""
function get_num_elements(::CanonicalFiniteElementSpace)
    return 1
end

@doc raw"""
    get_dim(space::CanonicalFiniteElementSpace)

Returns the dimension of a `CanonicalFiniteElementSpace`.
    
# Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the dimension.

# Return Value
The dimension `p+1` of the `CanonicalFiniteElementSpace`, where `p` is the polynomial degree.
"""
function get_dim(space::CanonicalFiniteElementSpace)
    return space.canonical_space.p+1
end

@doc raw"""
    get_local_basis(space::CanonicalFiniteElementSpace, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int)

Returns the local basis functions for a `CanonicalFiniteElementSpace`.

# Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the local basis functions.
- `::Int`: Unused argument (placeholder).
- `xi::NTuple{1,Vector{Float64}}`: The coordinates at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to apply.

# Return Value
The local basis functions evaluated at `xi` with `nderivatives` derivatives.
"""
function get_local_basis(space::CanonicalFiniteElementSpace,::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int)
    return evaluate(space.canonical_space, xi[1], nderivatives)
end

@doc raw"""
    get_basis_indices(space::CanonicalFiniteElementSpace, ::Int)

Returns the basis indices for a `CanonicalFiniteElementSpace`.

## Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the basis indices.
- `::Int`: Unused argument (placeholder).

# Return Value
The basis indices, which are the indices of the basis functions.
"""
function get_basis_indices(space::CanonicalFiniteElementSpace,::Int)
    return collect(1:get_dim(space))
end

@doc raw"""
    get_max_local_dim(space::CanonicalFiniteElementSpace)
    
Returns the maximum local dimension of a `CanonicalFiniteElementSpace`.

# Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the maximum local dimension.

# Return Value
The maximum local dimension, which is the dimension of the `CanonicalFiniteElementSpace`.
"""
function get_max_local_dim(space::CanonicalFiniteElementSpace)
    return get_dim(space)
end

@doc raw"""
    set_boundary_dof_indices(space::CanonicalFiniteElementSpace, indices::Vector{Int})

Sets the boundary degrees of freedom indices for a `CanonicalFiniteElementSpace`.
    
# Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to set the boundary degrees of freedom indices.
- `indices::Vector{Int}`: The indices of the boundary degrees of freedom.

# Return Value
`nothing`.
"""
function set_boundary_dof_indices(space::CanonicalFiniteElementSpace, indices::Vector{Int})
    space.boundary_dof_indices = indices
    return nothing
end

@doc raw"""

    get_boundary_dof_indices(space::CanonicalFiniteElementSpace)

Returns the boundary degrees of freedom indices for a `CanonicalFiniteElementSpace`.

# Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the boundary degrees of freedom indices.

# Return Value
The boundary degrees of freedom indices.

"""
function get_boundary_dof_indices(space::CanonicalFiniteElementSpace)
    return space.boundary_dof_indices
end

@doc raw"""
    get_extraction(space::CanonicalFiniteElementSpace, ::Int)

Returns the extraction matrix and basis indices for a `CanonicalFiniteElementSpace`.

# Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to retrieve the extraction matrix and basis indices.
- `::Int`: Unused argument (placeholder).

# Return Value
A tuple containing:
- The extraction matrix, which is an identity matrix.
- The basis indices.
"""
function get_extraction(space::CanonicalFiniteElementSpace,::Int)
    basis_indices = get_basis_indices(space, 1)
    nbasis = length(basis_indices)
    return Matrix(LinearAlgebra.I, nbasis, nbasis), basis_indices
end

@doc raw"""
    evaluate(space::CanonicalFiniteElementSpace, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int)

Evaluates the basis functions for a `CanonicalFiniteElementSpace`.
    
# Arguments
- `space::CanonicalFiniteElementSpace`: The `CanonicalFiniteElementSpace` object for which to evaluate the basis functions.
- `::Int`: Unused argument (placeholder).
- `xi::NTuple{1,Vector{Float64}}`: The coordinates at which to evaluate the basis functions.
- `nderivatives::Int`: The number of derivatives to apply.

# Return Value
A tuple containing:
- The basis functions evaluated at `xi` with `nderivatives` derivatives.
- The basis indices.
"""
function evaluate(space::CanonicalFiniteElementSpace, ::Int, xi::NTuple{1,Vector{Float64}}, nderivatives::Int)
    local_basis = get_local_basis(space, 1, xi, nderivatives)
    basis_indices = get_basis_indices(space, 1)

    return local_basis, basis_indices
end