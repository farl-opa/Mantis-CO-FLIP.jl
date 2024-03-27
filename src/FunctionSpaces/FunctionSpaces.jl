"""
This (sub-)module provides a collection of scalar function spaces.

The exported names are:
"""
module FunctionSpaces

import .. Mesh

"""
    AbstractFunctionSpace

Supertype for all scalar function spaces.
"""
abstract type AbstractFunctionSpace{n, k} end

include("SplineSpaces.jl")
include("ExtractionCoefficients.jl")

# Getter for the function spaces

# B-Spline getters
function get_space_dim(bspline::BSplineSpace{n, k}, d::Int) where {n, k}
    return (length(bspline.patch.breakpoints[d])-1) * bspline.polynomial_degree[d] - sum(bspline.regularity[d]) + 1
end

function get_space_dim(bspline::BSplineSpace{n, k}) where {n, k}
    return NTuple{n, Int}( get_space_dim(bspline, d) for d in 1:1:n)
end

end