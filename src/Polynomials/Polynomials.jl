"""
This (sub-)module provides a collection of polynomial bases.

The exported names are:
"""
module Polynomials




"""
    Polynomial

Supertype for all polynomials.
"""
abstract type AbstractPolynomials end



include("NodalPolynomials.jl")
include("BernsteinPolynomial.jl")


# Ensures that all polynomials can be evaluated by called an instance. 
# Automatically throws a MethodError if the subtype does not have an 
# evaluate function implemented. Some of these methods may have 
# additional arguments, such as the additional derivatives to evaluate, 
# so this version should allow that as well.
function (polynomial::AbstractPolynomials)(xi::Vector{Float64}, args...)::Matrix{Float64}
    return evaluate(polynomial, xi, args...)
end



end