"""
This module provides a collection of polynomial bases.

The exported names are:
"""
module Polynomials

# Write your package code here.

abstract type GenericPolynomials end

include("NodalPolynomials.jl")


end