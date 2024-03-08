"""
This (sub-)module provides a collection of local bases function.

The exported names are:
"""
module Bases

"""
    AbstractBases

Supertype for all bases.
"""
abstract type AbstractBases end

include("BSplines.jl")

"""
    evalute(E::Array{Float64,3}, el::Int, B::Array{Float64,3})::Matrix{Float64}

Evaluates the local basis functions on element `el` using the extraction coefficients `E` and reference functions `B`.

# Arguments
- `E::Array{Float64,3}`: extraction coefficients for all elements.
- `el::Int`: element in which to compute the basis functions.
- `B::Array{Float64,3}`: reference functions to be used.
# Returns
- `N::Array{Float64,3}`: local evaluation of basis functions.
"""
function evalute(E::Array{Float64,3}, el::Int, B::Array{Float64,3})::Matrix{Float64}
    nx = size(B)[1]
    p = size(E)[2]-1

    N = zeros(nx, p+1)

    for i=1:p+1
        for j=1:p+1
            @. N[:,i] += E[j,i,el] * (@view B[:,j,1])
        end
    end

    return N
end


end