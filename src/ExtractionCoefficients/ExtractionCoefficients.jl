"""
This (sub-)module provides the extraction coefficients.

The exported names are:
"""

module ExtractionCoefficients

include("BezierExtraction.jl")


"""
    evalute_basis(E::Array{Float64,3}, el::Int, B::Array{Float64,3})::Matrix{Float64}

Evaluates the local basis functions on element `el` using the extraction coefficients `E` and reference functions `B`.

# Arguments
- `E::Array{Float64,3}`: extraction coefficients for all elements.
- `el::Int`: element in which to compute the basis functions.
- `B::Array{Float64,3}`: reference functions to be used.
# Returns
- `N::Array{Float64,3}`: local evaluation of basis functions.
"""
function evalute_basis(E::Array{Float64,3}, el::Int, B::Array{Float64,3})::Matrix{Float64}
    nx = size(B)[1]
    p = size(E)[2]-1

    N = zeros(nx, p+1)

    for i=1:p+1
        for j=1:p+1
            @. N[:,i] += E[el,j,i] * (@view B[:,j,1])
        end
    end

    return N
end

end