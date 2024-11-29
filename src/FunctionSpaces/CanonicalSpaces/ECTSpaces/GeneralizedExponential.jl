@doc raw"""
    struct GeneralizedExponential <: AbstractECTSpaces

Concrete type for Generalized Exponential section space spanned by `<1, x, ..., x^(p-2), exp(wx), exp(-wx)>`, equivalently `<1, x, ..., x^(p-2), cosh(wx), sinh(wx)>`, on `[0,1]`.

# Fields
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.
- `C::Matrix{Float64}`: representation matrix for the local basis.
"""
struct GeneralizedExponential <: AbstractECTSpaces
    p::Int
    w::Float64
    t::Bool
    m::Int
    C::Matrix{Float64}
    endpoint_tol::Float64
    function GeneralizedExponential(p::Int)
        w = 1.0
        t = false
        m = 10
        GeneralizedExponential(p, w, t, m)
    end

    function GeneralizedExponential(p::Int, w::Float64)
        t = abs(w) >= 3.0
        m = 10
        GeneralizedExponential(p, w, t, m)
    end

    function GeneralizedExponential(p::Int, w::Float64, t::Bool)
        m = 10
        GeneralizedExponential(p, w, t, m)
    end

    function GeneralizedExponential(p::Int, w::Float64, t::Bool, m::Int)
        endpoint_tol = 1e-12
        new(p, w, t, m, gexp_representation(p, w, t, m),endpoint_tol)
     end
end

function _evaluate(gexp::GeneralizedExponential, xi::Float64, nderivatives::Int)
    tol = gexp.endpoint_tol
    M = zeros(Float64, 1, gexp.p+1, nderivatives+1)
    j = xi >= 0.0 && xi < 1.0
    if abs(xi-1.0) < tol
       j = true
    end
    if j
        if gexp.t
            for r = 0:nderivatives
                k = min(r, gexp.p-1)
                wxl = gexp.w * xi
                E = [1.0; cumprod((1.0 ./ (1:gexp.p-k)) * wxl)]
                E[gexp.p-k, :] .= exp(wxl);
                E[gexp.p+1-k, :] .= (-1)^r * exp(-wxl);
                M[1, :, r+1] = (gexp.w^r) * (gexp.C[:,k+1:end] * E)
            end
        else
            for r = 0:nderivatives
                k = min(r, gexp.p-1)
                ww = [1; cumprod(repeat([gexp.w * gexp.w], gexp.m))]
                Ef = [1.0; cumprod((1.0 ./ (1:gexp.p-k+2*gexp.m)) * xi)]
                E = Ef[1:gexp.p+1-k]
                E[gexp.p-k, :] = Ef[gexp.p-k:2:end, :]' * ww
                E[gexp.p-k+1, :] = Ef[gexp.p-k+1:2:end, :]' * ww
                M[1, :, r+1] = gexp.C[:,k+1:end] * E
            end
        end
    end
    return M
end

@doc raw"""
    evaluate(gexp::GeneralizedExponential, ξ::Vector{Float64})

Compute all basis function values at `ξ` in ``[0.0, 1.0]``.

# Arguments
- `gexp::GeneralizedExponential`:  Generalized Exponential section space.
- `xi::Vector{Float64}`: vector of evaluation points ``\in [0.0, 1.0]``.

See also [`evaluate(gexp::GeneralizedExponential, xi::Vector{Float64}, nderivatives::Int64)`](@ref).
"""
function evaluate(gexp::GeneralizedExponential, xi::Vector{Float64})
    return evaluate(gexp, xi, 0)
end

function evaluate(gexp::GeneralizedExponential, xi::Float64)
    return evaluate(gexp, [xi], 0)
end

@doc raw"""
    gexp_representation(p::Int, w::Float64, t::Bool, m::Int)

Build representation matrix for Generalized Exponential section space of degree `p`, weight `w` and `m` terms.

# Arguments
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.

# Returns:
- `C::Matrix{Float64}`: representation matrix for the local basis.
"""

import LinearAlgebra, ToeplitzMatrices

function gexp_representation(p::Int, w::Float64, t::Bool, m::Int)
    
    I = Matrix(1.0LinearAlgebra.I, p+1, p+1)
    if t
        ew = exp(w)
        ewm = exp(-w)
        M0 = I[:,:]
        M0[p, :] .= 1
        M0[p+1, 1:2:p+1] .= 1
        M0[p+1, 2:2:p+1] .= -1
        M1 = Matrix(ToeplitzMatrices.Toeplitz([1; cumprod(w ./ (1:p))], I[:,1]))
        M1[p, :] .= ew;
        M1[p+1, 1:2:p+1] .= ewm;
        M1[p+1, 2:2:p+1] .= -ewm;

    else
        M0 = I[:,:]
        ww = [1 cumprod(repeat([w * w], 1, m), dims=2)]
        M = ToeplitzMatrices.Toeplitz([1; cumprod(1.0 ./ (1:p+2*m))], I[:,1])
        M1 = M[1:p+1, :]
        M1[p, :] = ww * M[p:2:end, :]
        M1[p+1, :] = ww * M[p+1:2:end, :]
    end
    cs = zeros(Float64, p+1)
    C = zeros(Float64, p+1, p+1)
    C[p+1, :] =  [M1[:, 1] M0[:, 1:p]]' \ I[:,1]
    for i = 2:p
        cs = cs + C[p+1-i+2, :]
        cc = zeros(Float64, p+1)
        cc[i] = -cs' * M1[:, i]
        C[p+1-i+1, :] .= [M1[:, 1:i] M0[:, 1:p+1-i]]' \ cc
    end
    C[1, :] .= [M0[:, 1] M1[:, 1:p]]' \ I[:,1]

    return C
end

"""
    get_derivative_space(ect_space::GeneralizedExponential)

Get the space of one degree lower than the input space.

# Arguments
- `ect_space::GeneralizedExponential`: A ect space.

# Returns
- `::GeneralizedExponential`: A ect space of one degree lower than the input space.
"""
function get_derivative_space(ect_space::GeneralizedExponential)
    return GeneralizedExponential(ect_space.p-1, ect_space.w, ect_space.t, ect_space.m)
end