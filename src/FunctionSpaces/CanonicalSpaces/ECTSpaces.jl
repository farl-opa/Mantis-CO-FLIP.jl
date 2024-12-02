import SparseArrays, LinearAlgebra

abstract type AbstractECTSpaces <: AbstractCanonicalSpace end

@doc raw"""
    evaluate(ect_space::AbstractECTSpaces, ξ::Vector{Float64}, nderivatives::Int64)

Compute derivatives up to order `nderivatives` for all basis functions of degree `p` at `ξ` for ``\xi \in [0.0, 1.0]``. 

# Arguments
- `ect_space::AbstractECTSpaces`: ECT section space.
- `ξ::Vector{Float64}`: vector of evaluation points ``\in [0.0, 1.0]``.
- `nderivatives::Int64`: maximum order of derivatives to be computed (nderivatives ``\leq p``).

See also [`evaluate(ect_space::AbstractECTSpaces, ξ::Float64, nderivatives::Int64)`](@ref).
"""
@Memoization.memoize function evaluate(ect_space::AbstractECTSpaces, ξ::Vector{Float64}, nderivatives::Int)
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
        tmp = _evaluate(ect_space, ξ[i],nderivatives)
        for j = 0:nderivatives
            ders[j+1][1][i,:] .= tmp[1,:,j+1]
        end
    end
    return ders
end

"""
    build_two_scale_matrix(ect_space::AbstractECTSpaces, num_sub_elements::Int)

Uniformly subdivides the ECT space into `num_sub_elements` sub-elements. It is assumed that `num_sub_elements` is a power of 2, else the method throws an argument error. It returns a global subdivision matrix that maps the global basis functions of the ECT space to the global basis functions of the subspaces.

# Arguments
- `ect_space::AbstractECTSpaces`: A ect space.
- `num_sub_elements::Int`: The number of subspaces to divide the EC T space into.

# Returns
- `::SparseMatrixCSC{Float64}`: A global subdivision matrix that maps the global basis functions of the ECT space to the global basis functions of the subspaces.
"""
function build_two_scale_matrix(ect_space::AbstractECTSpaces, num_sub_elements::Int)
    num_ref = log2(num_sub_elements)
    if num_sub_elements < 2 || !isapprox(num_ref-round(num_ref), 0.0, atol=1e-12)
        throw(ArgumentError("Number of subdivisions should be a power of 2 and greater than 1"))
    end
    p = get_polynomial_degree(ect_space)
    num_ref = Int(num_ref)
    
    # first, build the subdivision matrix for num_sub_elements = 2
    bisection_matrix = zeros(Float64, 2 * (p+1), p+1)
    # evaluate all ECT basis functions at left and right endpoints
    eval_L = _evaluate_all_at_point(ect_space, 0.0, p-1)
    # fill the subdivision matrix by solving endpoint interpolation problems
    for i = 1:p
        if i == 1
            bisection_matrix[1, i] = 1.0
        else
            for j = 1:i-1
                bisection_matrix[i,j] = (eval_L[j, i]/2^(i-1) - sum(bisection_matrix[1:i-1, j] .* eval_L[1:i-1, i]))/eval_L[i, i]
            end
            bisection_matrix[i,i] = 1.0 - sum(bisection_matrix[i, 1:i-1])
        end
    end
    # evaluate at midpoint for the central degree of freedom
    eval_M = _evaluate_all_at_point(ect_space, 0.5, 0)
    bisection_matrix[p+1, 1:p+1] = eval_M
    # fill the rest of the matrix in symmetrically
    bisection_matrix[p+2:end, :] = bisection_matrix[p+1:-1:1, p+1:-1:1]
    # convert to a sparse matrix
    bisection_matrix = SparseArrays.sparse(bisection_matrix)

    # now, build the subdivision matrix for num_sub_elements > 2
    subdivision_matrix = SparseArrays.sparse(Matrix(LinearAlgebra.I, p+1, p+1))
    for i = 1:num_ref
        subdivision_matrix = SparseArrays.blockdiag([bisection_matrix for i = 1:2^(i-1)]...) * subdivision_matrix
    end
    
    return subdivision_matrix
end

include("ECTSpaces/GeneralizedTrigonometric.jl")
include("ECTSpaces/GeneralizedExponential.jl")