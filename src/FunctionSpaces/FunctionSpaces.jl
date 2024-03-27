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
abstract type AbstractFunctionSpace{n} end

include("SplineSpaces.jl")
include("ExtractionCoefficients.jl")

# Getter for the function spaces

#=
# B-Spline getters
function get_space_dim(bspline::BSplineSpace)
    return (length(bspline.patch.breakpoints)-1) * bspline.knot_vector.polynomial_degree - sum(bspline.regularity) + 1
end
=#

end