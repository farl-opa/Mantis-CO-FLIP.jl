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

end