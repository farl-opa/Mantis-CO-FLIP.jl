"""
    GeneralizedTrigonometric <: AbstractECTSpaces

Concrete type for Generalized Trignometric section space spanned by
``<1, x, ..., x^(p-2), cos(wx), sin(wx)>`` on ``[0,1]``.

# Fields
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.
- `C::Matrix{Float64}`: representation matrix for the local basis.
"""
struct GeneralizedTrigonometric <: AbstractECTSpaces
    p::Int
    w::Float64
    t::Bool
    m::Int
    C::Matrix{Float64}
    endpoint_tol::Float64
    function GeneralizedTrigonometric(p::Int)
        w = 1.0
        t = false
        m = 10
        return GeneralizedTrigonometric(p, w, t, m)
    end

    function GeneralizedTrigonometric(p::Int, w::Float64)
        t = abs(w) >= 3.0
        m = 10
        return GeneralizedTrigonometric(p, w, t, m)
    end

    function GeneralizedTrigonometric(p::Int, w::Float64, t::Bool)
        m = 10
        return GeneralizedTrigonometric(p, w, t, m)
    end

    function GeneralizedTrigonometric(p::Int, w::Float64, t::Bool, m::Int)
        endpoint_tol = 1e-12
        return new(p, w, t, m, gtrig_representation(p, w, t, m), endpoint_tol)
    end
end

function _evaluate(gtrig::GeneralizedTrigonometric, xi::Float64, nderivatives::Int)
    tol = gtrig.endpoint_tol
    M = zeros(Float64, 1, gtrig.p + 1, nderivatives + 1)
    j = xi >= 0.0 && xi < 1.0
    if abs(xi - 1.0) < tol
        j = true
    end
    if j
        if gtrig.t
            for r in 0:nderivatives
                k = min(r, gtrig.p - 1)
                wxl = gtrig.w * xi
                E = [1.0; cumprod((1.0 ./ (1:(gtrig.p - k))) * wxl)]
                if mod(r, 4) == 0
                    E[gtrig.p - k] = cos(wxl)
                    E[gtrig.p + 1 - k] = sin(wxl)
                elseif mod(r, 4) == 1
                    E[gtrig.p - k] = -sin(wxl)
                    E[gtrig.p + 1 - k] = cos(wxl)
                elseif mod(r, 4) == 2
                    E[gtrig.p - k] = -cos(wxl)
                    E[gtrig.p + 1 - k] = -sin(wxl)
                elseif mod(r, 4) == 3
                    E[gtrig.p - k] = sin(wxl)
                    E[gtrig.p + 1 - k] = -cos(wxl)
                end
                M[1, :, r + 1] = (gtrig.w^r) * (gtrig.C[:, (k + 1):end] * E)
            end
        else
            for r in 0:nderivatives
                k = min(r, gtrig.p - 1)
                ww = [1; cumprod(repeat([-gtrig.w * gtrig.w], gtrig.m))]
                Ef = [1.0; cumprod((1.0 ./ (1:(gtrig.p - k + 2 * gtrig.m))) * xi)]
                E = Ef[1:(gtrig.p + 1 - k)]
                E[gtrig.p - k, :] = Ef[(gtrig.p - k):2:end, :]' * ww
                E[gtrig.p - k + 1, :] = Ef[(gtrig.p - k + 1):2:end, :]' * ww
                M[1, :, r + 1] = gtrig.C[:, (k + 1):end] * E
            end
        end
    end
    return M
end

"""
    evaluate(gtrig::GeneralizedTrigonometric, ξ::Vector{Float64})

Compute all basis function values at `ξ` in ``[0.0, 1.0]``.

# Arguments
- `gtrig::GeneralizedTrigonometric`:  Generalized Trigonometric section space.
- `xi::Vector{Float64}`: vector of evaluation points in ``[0.0, 1.0]``.

See also [`evaluate(gtrig::GeneralizedTrigonometric, xi::Vector{Float64}, nderivatives::Int64)`](@ref).
"""
function evaluate(gtrig::GeneralizedTrigonometric, xi::Vector{Float64})
    return evaluate(gtrig, xi, 0)
end

function evaluate(gtrig::GeneralizedTrigonometric, xi::Float64)
    return evaluate(gtrig, [xi], 0)
end

"""
    gtrig_representation(p::Int, w::Float64, t::Bool, m::Int)

Build representation matrix for Generalized Trignometric section space of degree `p`,
weight `w` and `m` terms.

# Arguments
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.

# Returns:
- `C::Matrix{Float64}`: representation matrix for the local basis.
"""
function gtrig_representation(p::Int, w::Float64, t::Bool, m::Int)
    I = Matrix(1.0LinearAlgebra.I, p + 1, p + 1)
    if t
        cw = cos(w)
        sw = sin(w)
        M0 = I[:, :]
        M0[[p, p + 1], [p, p + 1]] .= 0.0
        M0[p, 1:4:end] .= 1
        M0[p, 3:4:end] .= -1
        M0[p + 1, 2:4:end] .= 1
        M0[p + 1, 4:4:end] .= -1
        M1 = Matrix(ToeplitzMatrices.Toeplitz([1; cumprod(w ./ (1:p))], I[:, 1]))
        M1[p, 1:4:end] .= cw
        M1[p, 2:4:end] .= -sw
        M1[p, 3:4:end] .= -cw
        M1[p, 4:4:end] .= sw
        M1[p + 1, 1:4:end] .= sw
        M1[p + 1, 2:4:end] .= cw
        M1[p + 1, 3:4:end] .= -sw
        M1[p + 1, 4:4:end] .= -cw
    else
        M0 = I[:, :]
        ww = [1 cumprod(repeat([-w * w], 1, m); dims=2)]
        M = ToeplitzMatrices.Toeplitz([1; cumprod(1.0 ./ (1:(p + 2 * m)))], I[:, 1])
        M1 = M[1:(p + 1), :]
        M1[p, :] = ww * M[p:2:end, :]
        M1[p + 1, :] = ww * M[(p + 1):2:end, :]
    end
    cs = zeros(Float64, p + 1)
    C = zeros(Float64, p + 1, p + 1)
    C[p + 1, :] = [M1[:, 1] M0[:, 1:p]]' \ [1.0; zeros(p)]
    for i in 2:p
        cs = cs + C[p + 1 - i + 2, :]
        cc = zeros(Float64, p + 1)
        cc[i] = -cs' * M1[:, i]
        C[p + 1 - i + 1, :] = [M1[:, 1:i] M0[:, 1:(p + 1 - i)]]' \ cc
    end
    C[1, :] = [M0[:, 1] M1[:, 1:p]]' \ [1.0; zeros(p)]

    return C
end

"""
    get_derivative_space(ect_space::GeneralizedTrigonometric)

Get the space of one degree lower than the input space.

# Arguments
- `ect_space::GeneralizedTrigonometric`: A ect space.

# Returns
- `::GeneralizedTrigonometric`: A ect space of one degree lower than the input space.
"""
function get_derivative_space(ect_space::GeneralizedTrigonometric)
    return GeneralizedTrigonometric(ect_space.p - 1, ect_space.w, ect_space.t, ect_space.m)
end

"""
    get_bisected_canonical_space(ect_space::GeneralizedTrigonometric)

Bisect the canonical space by dividing the weight in half.

# Arguments
- `ect_space::GeneralizedTrigonometric`: A ect space.

# Returns
- `::GeneralizedTrigonometric`: A ect space with the weight divided by 2.
"""
function get_bisected_canonical_space(ect_space::GeneralizedTrigonometric)
    return GeneralizedTrigonometric(ect_space.p, ect_space.w / 2, ect_space.t, ect_space.m)
end

"""
    get_finer_canonical_space(ect_space::GeneralizedTrigonometric, num_sub_elements::Int)

For number of sub-elements which is powers of 2, bisect the canonical space by dividing the
weight in half for each power.

# Arguments
- `ect_space::GeneralizedTrigonometric`: A ect space.
- `num_sub_elements::Int`: Number of sub-elements to be created.

# Returns
- `::GeneralizedTrigonometric`: A ect space with the subdivided weight.
"""
function get_finer_canonical_space(
    ect_space::GeneralizedTrigonometric, num_sub_elements::Int
)
    num_ref = log2(num_sub_elements)
    if num_sub_elements < 2 || !isapprox(num_ref - round(num_ref), 0.0; atol=1e-12)
        throw(
            ArgumentError(
                "Number of subdivisions should be a power of 2 and greater than 1"
            ),
        )
    end

    return GeneralizedTrigonometric(
        ect_space.p, ect_space.w / num_sub_elements, ect_space.t, ect_space.m
    )
end
