"""
    TBpatchTcheb

A TB-spline patch of degree `p` based on a linear differential operator with constant coefficients.
"""
struct Tchebycheff <: AbstractECTSpaces
    p::Int
    W::Tuple{Vector{Int}, Matrix{Float64}, Vector{Int}} # TB-spline root parameters (nroots × 2)
    l::Float64
    C::Matrix{Float64} # representation matrix
    mu::Vector{Int} # cumulative dimension
    endpoint_tol::Float64

    function Tchebycheff(
        p::Int,
        w::Matrix{Float64} = [0.0 0.0],
        l::Float64 = 1.0,
        mult::Vector{Int} = ones(Int, size(w, 1)),
        endpoint_tol::Float64 = 1e-12,
    )
        if any(mult .< 1)
            throw(ArgumentError("All multiplicities must be positive integers."))
        end
        if size(w, 1) != length(mult)
            throw(ArgumentError("The number of roots must match the number of multiplicities."))
        end

        # make all imaginary parts non-negative
        w[:, 2] .= abs.(w[:, 2])
        # total multiplicity: each real root counts once, each complex root counts twice
        m = sum(((w[:,2] .!= 0) .+ 1) .* mult)
        if p+1 != m
            if p+1 > m
                w = vcat(w, [0.0 0.0])
                push!(mult, p+1 - m)
            else
                throw(ArgumentError("The local dimension, $(p+1), must be greater than or equal to the number of roots (counting multiplicities), $(m)."))
            end
        end

        # find unique roots and their indices
        wi = unique(i -> w[i, :], 1:size(w,1))
        wu = w[wi, :]
        # find multiplicities of unique roots
        multu = similar(mult[wi])
        for ru in eachrow(wu)
            multu = sum(mult[findall(r -> all(r .== ru), eachrow(w))])
        end
        # store types of roots
        t = zeros(Int, length(multu))
        t[wu[:,1] .!=0] .= 1
        t[wu[:,2] .!=0] .= 2
        t[findall(i -> all(wu[i, :] .!= 0), 1:size(wu, 1))] .= 3
        # store all root information in a tuple
        # (type, real part, imaginary part, multiplicity)
        W = (t, wu, multu)

        # number of basis functions for each unique root
        mu = [0, cumsum(((wu[:, 2] .!= 0) + 1) .* multu)...]

        return new(
            p, W, l, tcheb_representation(p, W, l, mu), mu, endpoint_tol
        )
    end
end

"""
    TB_evaluation_all(P, xx; closed=true)

Evaluate all TB-splines at points `xx`.
"""
function TB_evaluation_all(P::TBpatchTcheb, xx::Vector{Float64}; closed::Bool=true)
    tol = 1e-12
    M = zeros(P.n, length(xx))
    j = (xx .>= P.xx[1]) .& (xx .< P.xx[end])
    if closed
        j[abs.(xx .- P.xx[end]) .< tol] .= true
    end
    if any(j)
        xl = xx[j] .- P.xx[1]
        E = zeros(P.n, length(xl))
        m = Int(maximum(P.W[:, 4])) - 1
        X = vcat(ones(1, length(xl)), cumprod.((1.0 ./ (1:m))' .* xl, dims=1))
        for i in 1:size(P.W, 1)
            kind = Int(P.W[i, 1])
            sel = P.mu[i]+1:P.mu[i+1]
            if kind == 0
                Ei = X[1:Int(P.W[i, 4]), :]
            elseif kind == 1
                ewxl = exp(P.W[i, 2] .* xl)
                Ei = X[1:Int(P.W[i, 4]), :] .* ewxl'
            elseif kind == 2
                cwxl = cos(P.W[i, 3] .* xl)
                swxl = sin(P.W[i, 3] .* xl)
                Ei = vcat(X[1:Int(P.W[i, 4]), :] .* cwxl', X[1:Int(P.W[i, 4]), :] .* swxl')
            elseif kind == 3
                ewxl = exp(P.W[i, 2] .* xl)
                ecwxl = ewxl .* cos(P.W[i, 3] .* xl)
                eswxl = ewxl .* sin(P.W[i, 3] .* xl)
                Ei = vcat(X[1:Int(P.W[i, 4]), :] .* ecwxl', X[1:Int(P.W[i, 4]), :] .* eswxl')
            end
            E[sel, :] = Ei
        end
        M[:, j] = P.C * E
    end
    return M
end

function tcheb_representation(p, W, l, mu)
    # compute the representation matrix of the TB-spline basis
    n = p + 1
    C = Matrix{Float64}(I, n, n)
    for k in 1:p
        # construct the k-th order divided difference matrix
        D = zeros(n - k, n - k + 1)
        for i in 1:(n - k)
            D[i, i] = -1 / l
            D[i, i + 1] = 1 / l
        end
        # construct the diagonal matrix of root evaluations
        V = zeros(n - k, n - k)
        for i in 1:(n - k)
            x = -k * l
            for j in 1:size(W, 1)
                kind = Int(W[j, 1])
                re = W[j, 2]
                im = W[j, 3]
                mult = Int(W[j, 4])
                if i > mu[j] && i <= mu[j + 1]
                    if kind == 0
                        V[i, i] = re
                    elseif kind == 1
                        V[i, i] = re
                    elseif kind == 2
                        V[i, i] = re + im * im / re
                    elseif kind == 3
                        V[i, i] = re + im * im / re
                    end
                    break
                end
            end
        end
        # update the representation matrix
        C = D * C[1:end - 1, :] - V * C[1:end - 1, :]
    end
    return C
end
