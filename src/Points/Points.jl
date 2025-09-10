"""
    module Points

Contains all definitions of points used to evaluate geometries, function spaces, forms and
any other objects.
"""
module Points

############################################################################################
#                                      Abstract Types                                      #
############################################################################################

"""
    AbstractPoints{manifold_dim}

Supertype for all evaluable points.

# Type parameters
- `manifold_dim`: Dimension of the manifold where the points are evaluated.
"""
abstract type AbstractPoints{manifold_dim} end

############################################################################################
#                                    Abstract Methods                                      #
############################################################################################

"""
    get_manifold_dim(points::AbstractPoints{manifold_dim}) where {manifold_dim}

Returns the manifold dimension of the evaluable `points`.
"""
get_manifold_dim(::AbstractPoints{manifold_dim}) where {manifold_dim} = manifold_dim

"""
    get_num_points(points::P) where {manifold_dim, P <: AbstractPoints{manifold_dim}}

Returns the number of of evaluable `points` in the given point structure.
"""
function get_num_points(::P) where {P <: AbstractPoints}
    throw(MethodError(get_num_points, (P,)))
end

"""
    get_constituent_points(points::P) where {P <: AbstractPoints}

Returns the constituent points of `points` per manifold dimension.
"""
function get_constituent_points(::P) where {P <: AbstractPoints}
    throw(MethodError(get_constituent_points, (P,)))
end

Base.firstindex(points::AbstractPoints) = 1
Base.lastindex(points::AbstractPoints) = get_num_points(points)
Base.keys(points::AbstractPoints) = firstindex(points):lastindex(points)
Base.length(points::AbstractPoints) = get_num_points(points)

function Base.iterate(points::AbstractPoints)
    i = firstindex(points)

    return getindex(points, i), i
end

function Base.iterate(points::AbstractPoints, i::Int)
    i += 1
    i > lastindex(points) && return nothing

    return getindex(points, i), i
end

############################################################################################
#                                         Includes                                         #
############################################################################################

include("./CartesianPoints.jl")
include("./PointSet.jl")

end
