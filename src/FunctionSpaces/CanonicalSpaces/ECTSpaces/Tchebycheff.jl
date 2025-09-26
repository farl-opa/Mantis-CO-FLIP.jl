"""
    TBpatchTcheb

A TB-spline patch of degree `p` based on a linear differential operator with constant coefficients.
"""
struct Tchebycheff <: AbstractECTSpaces
    p::Int
    roots::Matrix{Float64} # (real, imag) pairs of roots
    root_mult::Vector{Int} # multiplicities of roots
    root_type::Vector{Int} # types of roots: 0=zero, 1=real, 2=imaginary, 3=complex
    l::Float64
    mu::Vector{Int} # cumulative dimension
    C::Matrix{Float64} # representation matrix
    endpoint_tol::Float64

    function Tchebycheff(
        p::Int,
        roots::Matrix{Float64} = [0.0 0.0],
        l::Float64 = 1.0,
        root_mult::Vector{Int} = ones(Int, size(roots, 1)),
        endpoint_tol::Float64 = 1e-12,
    )
        if any(root_mult .< 1)
            throw(ArgumentError("All multiplicities must be positive integers."))
        end
        if size(roots, 1) != length(root_mult)
            throw(ArgumentError("The number of roots must match the number of multiplicities."))
        end

        # make all imaginary parts non-negative
        roots[:, 2] .= abs.(roots[:, 2])
        # total multiplicity: each real root counts once, each complex root counts twice
        m = sum(((roots[:,2] .!= 0) .+ 1) .* root_mult)
        if p+1 != m
            if p+1 > m
                roots = vcat(roots, [0.0 0.0])
                push!(root_mult, p+1 - m)
            else
                throw(ArgumentError("The local dimension, $(p+1), must be greater than or equal to the number of roots (counting multiplicities), $(m)."))
            end
        end

        # find unique roots and their indices
        roots_i = unique(i -> roots[i, :], 1:size(roots,1))
        roots_unq = roots[roots_i, :]
        # find multiplicities of unique roots
        root_mult_unq = similar(root_mult[roots_i])
        count = 0
        for ru in eachrow(roots_unq)
            root_mult_unq[count+1] = sum(root_mult[findall(r -> all(r .== ru), eachrow(roots))])
            count += 1
        end
        # store types of roots
        root_type = zeros(Int, length(root_mult_unq))
        root_type[findall(roots_unq[:,1] .!= 0.0)] .= 1
        root_type[findall(roots_unq[:,2] .!= 0.0)] .= 2
        root_type[findall(all(roots_unq .!= 0, dims=2))] .= 3

        # number of basis functions for each unique root
        mu = [0, cumsum(((roots_unq[:, 2] .!= 0) .+ 1) .* root_mult_unq)...]

        return new(
            p,
            roots_unq,
            root_mult_unq,
            root_type,
            l,
            mu,
            tcheb_representation(p, roots_unq, root_mult_unq, root_type, l, mu),
            endpoint_tol
        )
    end
end

# """
#     TB_evaluation_all(P, xx; closed=true)

# Evaluate all TB-splines at points `xx`.
# """
# function TB_evaluation_all(P::TBpatchTcheb, xx::Vector{Float64}; closed::Bool=true)
#     tol = 1e-12
#     M = zeros(p+1, length(xx))
#     j = (xx .>= P.xx[1]) .& (xx .< P.xx[end])
#     if closed
#         j[abs.(xx .- P.xx[end]) .< tol] .= true
#     end
#     if any(j)
#         xl = xx[j] .- P.xx[1]
#         E = zeros(p+1, length(xl))
#         m = Int(maximum(P.W[:, 4])) - 1
#         X = vcat(ones(1, length(xl)), cumprod.((1.0 ./ (1:m))' .* xl, dims=1))
#         for i in 1:length(root_mult)
#             r_ind = Int(root_type[i])
#             sel = mu[i]+1:mu[i+1]
#             if r_ind == 0
#                 Ei = X[1:Int(root_mult[i]), :]
#             elseif r_ind == 1
#                 ewxl = exp(roots[i, 1] .* xl)
#                 Ei = X[1:Int(root_mult[i]), :] .* ewxl'
#             elseif r_ind == 2
#                 cwxl = cos(roots[i, 2] .* xl)
#                 swxl = sin(roots[i, 2] .* xl)
#                 Ei = vcat(X[1:Int(root_mult[i]), :] .* cwxl', X[1:Int(root_mult[i]), :] .* swxl')
#             elseif r_ind == 3
#                 ewxl = exp(roots[i, 1] .* xl)
#                 ecwxl = ewxl .* cos(roots[i, 2] .* xl)
#                 eswxl = ewxl .* sin(roots[i, 2] .* xl)
#                 Ei = vcat(X[1:Int(root_mult[i]), :] .* ecwxl', X[1:Int(root_mult[i]), :] .* eswxl')
#             end
#             E[sel, :] = Ei
#         end
#         M[:, j] = P.C * E
#     end
#     return M
# end

function tcheb_representation(p, roots, root_mult, root_type, l, mu)
    M0 = zeros(p+1, p+1)
    M1 = zeros(p+1, p+1)
    m = maximum(root_mult) - 1
    X = vcat([1.0], cumprod(l ./ (1:m)))
    I = 1.0LinearAlgebra.I
    for i in eachindex(root_mult)
        r_ind = root_type[i]
        idx = (mu[i]+1):mu[i+1]
        if r_ind == 0
            M0i = Matrix(I, root_mult[i], p+1)
            M1i = Matrix(ToeplitzMatrices.Toeplitz(X[1:root_mult[i]], [1.0; zeros(p)]))
        elseif r_ind == 1
            ewl = exp(roots[i, 1] * l)
            M0i = Matrix(I, root_mult[i], p+1)
            M1i = zeros(root_mult[i], p+1)
            M1i[:, 1] .= X[1:root_mult[i]] * ewl
            for k in 1:p
                ki = min(k, root_mult[i]) - 1
                M0i[1:ki+1, k+1] .= vcat(0.0, M0i[1:ki, k]) + roots[i, 1] * M0i[1:ki+1, k]
                M1i[:, k+1] .= vcat(0.0, M1i[1:end-1, k]) + roots[i, 1] * M1i[:, k]
            end
        elseif r_ind == 2
            cwl = cos(roots[i, 2] * l)
            swl = sin(roots[i, 2] * l)
            n4 = root_mult[i]
            M0ci = Matrix(I, n4, p+1)
            M0si = zeros(n4, p+1)
            M1ci = zeros(n4, p+1)
            M1si = zeros(n4, p+1)
            M1ci[:, 1] .= X[1:n4] * cwl
            M1si[:, 1] .= X[1:n4] * swl
            for k in 1:p
                ki = min(k, n4) - 1
                M0ci[1:ki+1, k+1] .= vcat(0.0, M0ci[1:ki, k]) - roots[i, 2] * M0si[1:ki+1, k]
                M0si[1:ki+1, k+1] .= vcat(0.0, M0si[1:ki, k]) + roots[i, 2] * M0ci[1:ki+1, k]
                M1ci[:, k+1] .= vcat(0.0, M1ci[1:end-1, k]) - roots[i, 2] * M1si[:, k]
                M1si[:, k+1] .= vcat(0.0, M1si[1:end-1, k]) + roots[i, 2] * M1ci[:, k]
            end
            M0i = vcat(M0ci, M0si)
            M1i = vcat(M1ci, M1si)
        elseif r_ind == 3
            ecwl = exp(roots[i, 1] * l) * cos(roots[i, 2] * l)
            eswl = exp(roots[i, 1] * l) * sin(roots[i, 2] * l)
            n4 = root_mult[i]
            M0ci = Matrix(I, n4, p+1)
            M0si = zeros(n4, p+1)
            M1ci = zeros(n4, p+1)
            M1si = zeros(n4, p+1)
            M1ci[:, 1] .= X[1:n4] * ecwl
            M1si[:, 1] .= X[1:n4] * eswl
            for k in 1:p
                ki = min(k, n4) - 1
                M0ci[1:ki+1, k+1] .= vcat(0.0, M0ci[1:ki, k]) +
                    roots[i, 1] * M0ci[1:ki+1, k] - roots[i, 2] * M0si[1:ki+1, k]
                M0si[1:ki+1, k+1] .= vcat(0.0, M0si[1:ki, k]) +
                    roots[i, 1] * M0si[1:ki+1, k] + roots[i, 2] * M0ci[1:ki+1, k]
                M1ci[:, k+1] .= vcat(0.0, M1ci[1:end-1, k]) +
                    roots[i, 1] * M1ci[:, k] - roots[i, 2] * M1si[:, k]
                M1si[:, k+1] .= vcat(0.0, M1si[1:end-1, k]) +
                    roots[i, 1] * M1si[:, k] + roots[i, 2] * M1ci[:, k]
            end
            M0i = vcat(M0ci, M0si)
            M1i = vcat(M1ci, M1si)
        end
        M0[idx, :] .= M0i
        M1[idx, :] .= M1i
    end
    cs = zeros(p+1)
    C = zeros(p+1, p+1)
    # Solve for last row
    C[p+1, :] = (Matrix(I, 1, p+1)) / hcat(M1[:, 1], M0[:, 1:p])
    for i in 2:p
        cs .+= C[p+1-i+2, :]
        cc = zeros(1, p+1)
        cc[i] = reduce(+, -cs .* M1[:, i])
        C[p+1-i+1, :] = cc / hcat(M1[:, 1:i], M0[:, 1:p+1-i])
    end
    C[1, :] = (Matrix(I, 1, p+1)) / hcat(M0[:, 1], M1[:, 1:p])

    return C
end

function get_derivative_space(ect_space::Tchebycheff)
    zero_i = findall(all(ect_space.roots .== 0.0, dims=2))
    if length(zero_i) == 0
        throw(ArgumentError("The zero root must exist for the input Tchebycheff space."))
    end
    if ect_space.root_mult[zero_i[1]] == 1
        idx = setdiff(1:length(ect_space.root_mult), zero_i)
        new_roots = ect_space.roots[idx, :]
        new_root_mult = ect_space.root_mult[idx]
        new_root_type = ect_space.root_type[idx]
    else
        new_roots = ect_space.roots
        new_root_mult = copy(ect_space.root_mult)
        new_root_mult[zero_i[1]] -= 1
        new_root_type = copy(ect_space.root_type)
    end
    return Tchebycheff(
        ect_space.p-1, new_roots, ect_space.l, new_root_mult
    )
end

function get_bisected_canonical_space(ect_space::Tchebycheff)
    return Tchebycheff(ect_space.p, ect_space.roots, ect_space.l/2, ect_space.root_mult)
end

function get_finer_canonical_space(ect_space::Tchebycheff, num_sub_elements::Int)
    num_ref = log2(num_sub_elements)
    if num_sub_elements < 2 || !isapprox(num_ref - round(num_ref), 0.0; atol=1e-12)
        throw(
            ArgumentError(
                "Number of subdivisions should be a power of 2 and greater than 1"
            ),
        )
    end

    return Tchebycheff(
        ect_space.p, ect_space.roots, ect_space.l/num_sub_elements, ect_space.root_mult
    )
end
