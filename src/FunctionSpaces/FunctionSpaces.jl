"""
This (sub-)module provides a collection of function spaces.
The exported names are:
"""
module FunctionSpaces

import Graphs
import LinearAlgebra
import Memoization
import PolynomialBases
import SparseArrays
import ToeplitzMatrices

using ..GeneralHelpers
import ..Mesh

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
    AbstractFESpace{manifold_dim, num_components} <: AbstractFunctionSpace

Supertype for all scalar finite element spaces.

# Type parameters
- `manifold_dim::Int`: Dimension of the manifold.
- `num_components::Int`: Number of (output) components of the function space.
"""
abstract type AbstractFESpace{manifold_dim, num_components} <: AbstractFunctionSpace end

function get_manifold_dim(
    ::AbstractFESpace{manifold_dim, num_components}
) where {manifold_dim, num_components}
    return manifold_dim
end
function get_num_components(
    ::AbstractFESpace{manifold_dim, num_components}
) where {manifold_dim, num_components}
    return num_components
end

include("FiniteElementSpaces/FiniteElementSpaces.jl")
include("AdaptiveRefinement/AdaptiveRefinement.jl")

# helper functions for convenience
include("./FunctionSpaceHelpers.jl")

end
