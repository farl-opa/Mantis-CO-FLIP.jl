"""
This (sub-)module provides a collection of function spaces.
The exported names are:
"""
module FunctionSpaces

import .. Mesh

import Combinatorics
import LinearAlgebra
import Memoization
import PolynomialBases
import SparseArrays
import ToeplitzMatrices

"""
    AbstractFunctionSpace

Supertype for all function spaces.
"""
abstract type AbstractFunctionSpace end

"""
    AbstractCanonicalSpace <: AbstractFunctionSpace

Supertype for all element-local bases. These spaces are only defined in the canonical domain
and are used to define the shape functions on the reference element.
"""
abstract type AbstractCanonicalSpace <: AbstractFunctionSpace end

abstract type AbstractECTSpaces <: AbstractCanonicalSpace end
abstract type AbstractLagrangePolynomials <: AbstractCanonicalSpace end
abstract type AbstractEdgePolynomials <: AbstractCanonicalSpace end

include("CanonicalSpaces/CanonicalSpaces.jl")

"""
    AbstractFESpace{manifold_dim, image_dim}

Supertype for all scalar finite element spaces.
"""
abstract type AbstractFESpace{manifold_dim, image_dim} <: AbstractFunctionSpace end

function get_manifold_dim(::AbstractFESpace{manifold_dim, image_dim}) where {
    manifold_dim, image_dim
}
    return manifold_dim
end
function get_image_dim(::AbstractFESpace{manifold_dim, image_dim}) where {
    manifold_dim, image_dim
}
    return image_dim
end


abstract type AbstractMultiComponentSpace{manifold_dim, image_dim, num_components} <:
    AbstractFESpace{manifold_dim, image_dim}
end


include("FiniteElementSpaces/FiniteElementSpaces.jl")
#include("AdaptiveRefinement/AdaptiveRefinement.jl")

# helper functions for convenience
include("./FunctionSpaceHelpers.jl")

end
