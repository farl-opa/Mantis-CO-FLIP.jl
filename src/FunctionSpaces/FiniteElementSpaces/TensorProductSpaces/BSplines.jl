# Specialisations for b-spline tensor product spaces. For the definition of b-spline spaces,
# see the `BSplines.jl`-file in the `UnivariateSplines`-directory.

"""
    get_greville_points(tp_space::TensorProductSpace{manifold_dim, T}) where {
        manifold_dim, num_spaces, T <: NTuple{num_spaces, BSplineSpace}
    }

Compute the Greville points for a `TensorProductSpace` composed of `BSplineSpace`s.

# Arguments
- `tp_space::TensorProductSpace{manifold_dim, T}`: The tensor product space containing
    `BSplineSpace`s.

# Returns
- `::NTuple{manifold_dim, Vector{Float64}}`: A tuple of vectors containing the Greville
    points for each dimension of the tensor product space.
"""
function get_greville_points(
    tp_space::TensorProductSpace{manifold_dim, num_components, num_patches, T}
) where {
    manifold_dim,
    num_components,
    num_patches,
    num_spaces,
    T <: NTuple{num_spaces, BSplineSpace},
}
    space_points = map(get_greville_points, tp_space.fem_spaces)

    greville_points = Vector{Vector{Float64}}(undef, manifold_dim)

    manifold_dims = map(get_manifold_dim, tp_space.fem_spaces)
    cumsum_dims = [0; cumsum(manifold_dims)...]
    for space_id in 1:num_spaces
        for dim in 1:manifold_dims[space_id]
            greville_points[dim+cumsum_dims[space_id]] = space_points[space_id][dim]
        end
    end

    return tuple(greville_points...)
end
