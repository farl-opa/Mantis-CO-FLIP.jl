"""
This (sub-)module provides a collection of function spaces.
The exported names are:
"""
module FunctionSpaces

import .. Mesh
import SparseArrays
import Memoization

"""
    AbstractFunctionSpace

Supertype for all function spaces.
"""
abstract type AbstractFunctionSpace end

# core functionality
include("CanonicalSpaces/CanonicalSpaces.jl")  # Creates Module CanonicalSpaces
include("FiniteElementSpaces/FiniteElementSpaces.jl")  # Creates Module FiniteElementSpaces
#include("AdaptiveRefinement/AdaptiveRefinement.jl")

# helper functions for convenience
include("./FunctionSpaceHelpers.jl")

end