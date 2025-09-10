"""
    PointSet{manifold_dim, T} <: AbstractPoints{manifold_dim}

Represents a set of points in `manifold_dim` dimensions.

# Fields
- `point_set::Vector{NTuple{manifold_dim, T}}`: The vector containing each point in the
  point set.
"""
struct PointSet{manifold_dim, T} <: AbstractPoints{manifold_dim}
    point_set::Vector{NTuple{manifold_dim, T}}

    function PointSet(
        point_set::Vector{NTuple{manifold_dim, T}}
    ) where {manifold_dim, T <: Number}
        return new{manifold_dim, T}(point_set)
    end
end

"""
    get_point_set(points::PointSet)

Returns the `Vector` containing each point in `points`.
"""
get_point_set(points::PointSet) = points.point_set
get_num_points(points::PointSet) = length(get_point_set(points))

function Base.getindex(points::PointSet, i::Int)
    return get_point_set(points)[i]
end

function get_constituent_points(points::PointSet{manifold_dim}) where {manifold_dim}
    point_set = get_point_set(points)
    num_points = get_num_points(points)
    const_points = ntuple(
        dim -> [point_set[point][dim] for point in 1:num_points], manifold_dim
    )

    return const_points
end
