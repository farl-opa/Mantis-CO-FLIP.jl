"""
    AbstractMultiValuedFiniteElementSpace

Supertype for all multi-valued finite element spaces, i.e. spaces that return multiple values for each evaluation point. 
"""
abstract type AbstractMultiValuedFiniteElementSpace{manifold_dim, num_components} <: AbstractFunctionSpace end

# Getters for the function spaces
get_manifold_dim(_::AbstractMultiValuedFiniteElementSpace{manifold_dim,num_components}) where {manifold_dim, num_components} = manifold_dim

get_num_components(_::AbstractMultiValuedFiniteElementSpace{manifold_dim,num_components}) where {manifold_dim, num_components} = num_components

include("DirectSumSpace.jl")