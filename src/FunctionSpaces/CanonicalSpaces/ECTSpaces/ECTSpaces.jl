
"""
    evaluate(ect_space::AbstractECTSpaces, ξ::Vector{Float64}, nderivatives::Int64)

Compute derivatives up to order `nderivatives` for all basis functions of degree `p` at `ξ`
for ``\\xi \\in [0.0, 1.0]``.

# Arguments
- `ect_space::AbstractECTSpaces`: ECT section space.
- `ξ::Vector{Float64}`: vector of evaluation points ``\\in [0.0, 1.0]``.
- `nderivatives::Int64`: maximum order of derivatives to be computed (nderivatives
    ``\\leq p``).

See also [`evaluate(ect_space::AbstractECTSpaces, ξ::Float64, nderivatives::Int64)`](@ref).
"""
@Memoization.memoize function evaluate(
    ect_space::AbstractECTSpaces, ξ::Vector{Float64}, nderivatives::Int
)
    neval = length(ξ)
    # allocate space for derivatives
    # - ders[j+1][1] contains the matrix of evaluations of the j-th derivative
    ders = Vector{Vector{Matrix{Float64}}}(undef, nderivatives + 1)
    for j = 0:nderivatives
        ders[j+1] = Vector{Matrix{Float64}}(undef, 1)
        ders[j+1][1] = zeros(Float64, neval, ect_space.p + 1)
    end
    # loop over the evaluation points and evaluate all derivatives at each point
    for i = 1:neval
        tmp = _evaluate(ect_space, ξ[i], nderivatives)
        for j = 0:nderivatives
            ders[j+1][1][i,:] .= tmp[1,:,j+1]
        end
    end
    return ders
end

include("GeneralizedTrigonometric.jl")
include("GeneralizedExponential.jl")
include("Tchebycheff.jl")
