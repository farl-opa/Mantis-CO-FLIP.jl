struct SignedDistanceBoundary{manifold_dim, F} <: AbstractImmersedBoundary{manifold_dim}
    signed_distance::F

    function SignedDistanceBoundary(manifold_dim, signed_distance::F) where {F}
        new{manifold_dim, F}(signed_distance)
    end
end

function get_distance(
    boundary::SignedDistanceBoundary{manifold_dim, F}, points::AbstractArray
) where {manifold_dim, F}
    return boundary.signed_distance(points)
end

function get_indicator(
    boundary::SignedDistanceBoundary{manifold_dim, F}, points::AbstractArray
) where {manifold_dim, F}
    return get_distance(boundary, points) .<= 0.0
end

function get_indicator_and_distance(
    boundary::SignedDistanceBoundary{manifold_dim, F}, points::AbstractArray
) where {manifold_dim, F}
    distance = get_distance(boundary, points)
    return distance .<= 0.0, distance
end
