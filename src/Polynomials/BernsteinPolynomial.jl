
"""
    polynomial_bernstein(p::Int, l::Int, x::Float64)::Float64

Compute the value of the `l`-th Bernstein polynomial of degree `p` at `x`. 

Evaluates the `l`-th Bernstein polynomial of degree `p` using the 
definition at `x`, for ``x \\in [0.0, 1.0]``. Up to `p=3`, the formulas 
are written out. Beyond that the definition is used, as is the build-in 
`binomial`-function.

# Arguments
- `p::Int`: degree of the polynomial (``p \\geq 0``).
- `l::Int`: number of the polynomial (``0 \\leq l \\leq p``).
- `x::Float64`: location to evaluate the polynomial at (``x \\in [0.0, 1.0]``).
"""
function polynomial_bernstein(p::Int, l::Int, x::Float64)::Float64
    if p < 0
        msg = "The Bernstein polynomial must be of degree at least 0."
        throw(ArgumentError(msg))
    elseif l > p
        msg = "Only the polynomials for l = 0, 1, ..., $p exist for p = $p."
        msg2 = " You asked for l = $l."
        throw(ArgumentError(msg*msg2))
    elseif x < 0.0 || x > 1.0
        msg = "x = $x is outside the interval [0.0, 1.0]."
        throw(ArgumentError(msg))
    end

    if p == 0
        return 1.0

    elseif p == 1
        if l == 0
            return 1.0 - x
        else
            return x
        end

    elseif p == 2
        if l == 0
            return (1.0 - x)^2
        elseif l == 1
            return 2.0*x*(1.0 - x)
        else
            return x^2
        end

    elseif p == 3
        if l == 0
            return (1.0-x)^3
        elseif l == 1
            return 3.0*x*(1.0 - x)^2
        elseif l == 2
            return 3.0*x^2*(1.0 - x)
        else
            return x^3
        end

    else
        return binomial(p, l) * x^l * (1.0 - x)^(p-l)
    end
end

