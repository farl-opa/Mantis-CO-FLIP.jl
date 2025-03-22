
"""
    get_extraction_operator(space::AbstractFESpace)

Returns the extraction operator of the given space.

!!! warning "Not all AbstractFESpaces have an extraction operator."
    Having an explicitly defined extraction operator in not required for building a finite
    element space. If such no extraction operator is defined, an error is thrown saying
    that the type does not have field 'extraction_op'.

# Arguments
- `space::AbstractFESpace`: A finite element space.

# Returns
- `::ExtractionOperator`: The extraction operator.
"""
function get_extraction_operator(space::AbstractFESpace)
    return space.extraction_op
end

"""
    get_extraction(space::AbstractFESpace, element_id::Int)

Returns the extraction coefficients and indices of the supported basis functions on the
given element. If no extraction operator is defined for `space`, and there is no specific
`get_extraction` method for `space`, an error is thrown saying that the type does not have
field 'extraction_op'.

# Arguments
- `space::AbstractFESpace`: A finite element space.
- `element_id::Int`: The identifier of the element.

# Returns
- `::Matrix{Float64}`: The extraction coefficients.
- `::Vector{Int}`: The (global) basis functions supported on this element.
"""
function get_extraction(space::AbstractFESpace, element_id::Int)
    return get_extraction(get_extraction_operator(space), element_id)
end

"""
    get_basis_indices(space::AbstractFESpace, element_id::Int)

Get the global indices of the basis functions of `space` on element `element_id`.

# Arguments
- `space::AbstractFESpace`: Finite element space.
- `element_id::Int`: Indendifier of the element.

# Returns
- `::Vector{Int}`: Global indices of the basis functions.
"""
function get_basis_indices(space::AbstractFESpace, element_id::Int)
    return get_basis_indices(get_extraction_operator(space), element_id)
end

"""
    get_num_basis(space::AbstractFESpace)

Returns the number of basis functions of the finite element space `space`.

# Arguments
- `space::AbstractFESpace`: Finite element space.

# Returns
- `::Int`: Number of basis functions spanning the finite element space.
"""
function get_num_basis(space::AbstractFESpace)
    return get_num_basis(get_extraction_operator(space))
end

"""
    get_num_basis(space::AbstractFESpace, element_id::Int)

Get the number of basis functions of the finite element space `space` supported on element
`element_id`.

# Arguments
- `space::AbstractFESpace`: Finite element space.
- `element_id::Int`: Indentifier of the element.

# Returns
- `::Int`: Number of basis functions supported on the given element.
"""
function get_num_basis(space::AbstractFESpace, element_id::Int)
    return get_num_basis(get_extraction_operator(space), element_id)
end

"""
    get_dof_partition(space::AbstractFESpace{manifold_dim, 1}) where {manifold_dim}

Retrieve and return the degrees of freedom (d.o.f.s) partition for `space`.

# Arguments
- `space::AbstractFESpace{manifold_dim, 1}`: Single-component finite element space.

# Returns
- `::Vector{Vector{Vector{Int}}}`: Nested d.o.f. partition. The first level of nesting
    corresponds to the patch. The second level corresponds to the division (see ... for its
    definition). The third level corresponds to the individual d.o.f.s.
"""
function get_dof_partition(space::AbstractFESpace{manifold_dim, 1}) where {manifold_dim}
    return space.dof_partition
end

"""
    get_max_local_dim(space::AbstractFESpace)

Compute an upper bound of the element-local dimension of `space`. Note that this is not
necessarily a tight upper bound.

# Arguments
- `space::AbstractFESpace`: A finite element space.

# Returns
- `::Int`: The element-local upper bound.
"""
function get_max_local_dim(space::AbstractFESpace)
    return error("get_max_local_dim not implemented for $(typeof(space))")
end

"""
    get_num_elements(space::AbstractFESpace)

Returns the number of elements in the underlying partition.

# Arguments
- `bspline::BSplineSpace`: A finite element space.

# Returns
- `::Int`: The number of elements.
"""
function get_num_elements(space::AbstractFESpace)
    return get_num_elements(get_extraction_operator(space))
end

"""
    get_local_basis(
        space::AbstractFESpace{manifold_dim, num_components},
        element_id::Int,
        xi::NTuple{manifold_dim,Vector{Float64}},
        nderivatives::Int,
    ) where {manifold_dim, num_components}

Evaluates the local basis functions used to construct the finite element space `space`.

# Arguments
- `space::AbstractFESpace{manifold_dim, num_components}`: A finite element space.
- `element_id::Int`: The indentifier of the element.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: The coordinates at which to evaluate the
    basis functions. These coordinates are in the **canonical** domain, and thus always lie
    in the interval `[0, 1]`. The coordinates have to be given per dimension. Multi-
    dimensional spaces are evaluated on the tensor product of the given coordinates.
- `nderivatives::Int`: The number of derivatives to compute.

# Returns
- `::Vector{Vector{Matrix{Float64}}}`: A nested vector structure containing the local basis
    functions and their derivatives. The first level of nesting corresponds to the order of
    the derivatives. The second level corresponds to a specific derivative. The matrix
    contains the actual evaluations. See [`get_derivative_idx(der_key::Vector{Int})`](@ref)
    for more details on the order in which the derivatives are stored.
"""
function get_local_basis(
    space::AbstractFESpace{manifold_dim, num_components},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int,
) where {manifold_dim, num_components}
    return error("get_local_basis not implemented for $(typeof(space))")
end

"""
    evaluate(
        space::AbstractFESpace{manifold_dim, num_components},
        element_id::Int,
        xi::NTuple{manifold_dim,Vector{Float64}},
        nderivatives::Int=0
    ) where {manifold_dim, num_components}

Evaluate the basis functions of the finite element space `space` at the point `xi` on the
element with identifier `element_id` up to order `nderivatives`.

All of our function spaces are built in the parameteric domain. For example, in 1D, let the
mapping from canonical coordinates to the `i`-th parametric element be
`Φᵢ: [0, 1] -> [aᵢ, bᵢ]` then, for a function `f` defined on the parametric patch, the
evaluate method actually returns the values/derivatives of `f∘Φᵢ`. Consider the function
`f₀ := f∘Φᵢ`. Then, `FunctionSpaces.evaluate(f)` returns
`evaluations::Vector{Vector{Vector{Matrix{Float64}}}` where `evaluations[i][j][k][a,b]` is
the evaluation of:
- the `j`-th mixed derivative ...
- of order `i-1` ...
- for the `b`-th basis function ...
- of the `k`-th component ...
- at the `a`-th evaluation point ...
- for `f₀`.
See [`get_derivative_idx(der_key::Vector{Int})`](@ref) for more details on the order in
which all the mixed derivatives of order `i-1` are stored.

# Arguments
- `space::AbstractFESpace{manifold_dim, num_components}`: Finite element space
- `element_id::Int`: Index of the element
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Point on the element
- `nderivatives::Int=0`: Order of the derivatives. Default is 0 (i.e., function evaluation).

# Returns
- `evaluation::Vector{Vector{Matrix{Float64}}}`: Values of the basis functions.
- `basis_indices::Vector{Int}`: Global indices of the basis functions.
"""
function evaluate(
    space::AbstractFESpace{manifold_dim, num_components},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int=0,
) where {manifold_dim, num_components}
    extraction_coefficients, basis_indices = get_extraction(space, element_id)
    local_basis = get_local_basis(space, element_id, xi, nderivatives)
    evaluation = Vector{Vector{Matrix{Float64}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        n_ders = length(local_basis[j + 1])
        evaluation[j + 1] = Vector{Matrix{Float64}}(undef, n_ders)
    end

    for j in 0:nderivatives
        for k in 1:length(local_basis[j + 1])
            if isassigned(local_basis[j + 1], k)
                evaluation[j + 1][k] = @views local_basis[j + 1][k] *
                    extraction_coefficients
            end
        end
    end

    return evaluation, basis_indices
end

function evaluate(
    space::AbstractFESpace{manifold_dim, num_components},
    element_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
    nderivatives::Int,
    coeffs::Vector{Float64},
) where {manifold_dim, num_components}
    basis_eval, basis_indices = evaluate(space, element_id, xi, nderivatives)
    evaluation = Vector{Vector{Vector{Float64}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        n_ders = length(basis_eval[j + 1])
        evaluation[j + 1] = Vector{Vector{Float64}}(undef, n_ders)
    end

    for j in 0:nderivatives
        for k in 1:length(basis_eval[j + 1])
            if isassigned(basis_eval[j + 1], k)
                evaluation[j + 1][k] = @views basis_eval[j + 1][k] * coeffs[basis_indices]
            end
        end
    end

    return evaluation
end

"""
    _evaluate_all_at_point(
        fem_space::AbstractFESpace{1, num_components},
        element_id::Int,
        xi::Float64,
        nderivatives::Int,
    ) where {num_components}

Evaluates all derivatives up to order `nderivatives` for all basis functions of `fem_space`
at a given point `xi` in the element `element_id`.

# Arguments
- `fem_space::AbstractFESpace{1, num_components}`: A univariate FEM space.
- `element_id::Int`: The id of the element.
- `xi::Float64`: The point where all global basis functiuons are evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.

# Returns
- `::SparseMatrixCSC{Float64}`: Global basis functions, size = n_dofs x nderivatives+1
"""
function _evaluate_all_at_point(
    fem_space::AbstractFESpace{1, num_components},
    element_id::Int,
    xi::Float64,
    nderivatives::Int,
) where {num_components}
    basis_eval, basis_indices = evaluate(fem_space, element_id, ([xi],), nderivatives)
    nloc = length(basis_indices)
    ndofs = get_num_basis(fem_space)
    I = zeros(Int, nloc * (nderivatives + 1))
    J = zeros(Int, nloc * (nderivatives + 1))
    V = zeros(Float64, nloc * (nderivatives + 1))
    count = 0
    for r in 0:nderivatives
        for i in 1:nloc
            I[count + 1] = basis_indices[i]
            J[count + 1] = r + 1
            V[count + 1] = basis_eval[r + 1][1][1, i]
            count += 1
        end
    end

    return SparseArrays.sparse(I, J, V, ndofs, nderivatives + 1)
end

"""
    _compute_parametric_geometry_coeffs(fem_space::F) where {
        manifold_dim,
        num_components,
        F<:FunctionSpaces.AbstractFESpace{manifold_dim, num_components}
    }

Compute the parametric geometry coefficients for a given finite element space.

This function calculates the coefficients necessary for representing the parametric geometry
of a finite element space.

# Arguments
- `fem_space::F`: The finite element space for which to compute the parametric geometry
    coefficients.

# Returns
- `coeffs::Matrix{Float64}`: The coefficients necessary for representing the parametric
    geometry. The size of the matrix is (num_basis x manifold_dim).
"""
function _compute_parametric_geometry_coeffs(
    fem_space::F
) where {manifold_dim, num_components, F <: AbstractFESpace{manifold_dim, num_components}}
    function _compute_element_xi(xi_canonical_prod, element_vertices)
        element_xi = Matrix{Float64}(undef, size(xi_canonical_prod))

        # affine transformation x ↦ a+x(b-a)
        for dim in axes(xi_canonical_prod, 2)
            element_xi[:, dim] = @. element_vertices[dim][1] +
                xi_canonical_prod[:, dim] *
                                    (element_vertices[dim][2] - element_vertices[dim][1])
        end

        return element_xi
    end
    degrees = get_polynomial_degree(fem_space)

    num_xi_per_dim = degrees .+ 1
    num_xi = prod(num_xi_per_dim)

    xi_canonical = tuple(collect.(range(0, 1, num_xi_per_dim[i]) for i in 1:manifold_dim)...)
    xi_canonical_prod = Matrix{Float64}(undef, num_xi, manifold_dim)
    for (id, point) in enumerate(Iterators.product(xi_canonical...))
        xi_canonical_prod[id, :] .= point
    end

    num_elements = get_num_elements(fem_space)

    points = Matrix{Float64}(undef, num_elements * num_xi, manifold_dim)
    A = zeros(size(points, 1), get_num_basis(fem_space))

    for element_id in 1:num_elements
        element_vertices = get_element_vertices(fem_space, element_id)
        element_xi = _compute_element_xi(xi_canonical_prod, element_vertices)

        curr_points_ids = ((element_id - 1) * num_xi + 1):(element_id * num_xi)
        points[curr_points_ids, :] = element_xi
        eval = evaluate(fem_space, element_id, xi_canonical)
        A[curr_points_ids, eval[2]] .= eval[1][1][1]
    end

    coeffs = A \ points

    return coeffs
end

include("FiniteElementSpacesHelpers.jl")
# Extraction operator: functionaly for extraction operator
include("ExtractionOperator.jl")
# rational version of finite element spaces
include("RationalFESpaces.jl")
# univariate function spaces
include("UnivariateSplines/UnivariateSplines.jl")
# composite function spaces
include("MultiPatchSpaces/UnstructuredSpaces.jl")
include("TensorProductSpaces/TensorProductSpaces.jl")
# two scale relations
include("TwoScaleRelations/AbstractTwoScaleRelations.jl")
# hierarchical function spaces
include("Hierarchical/Hierarchical.jl")
# multi-valued function spaces
include("MultiComponentSpaces/MultiComponentSpaces.jl")
