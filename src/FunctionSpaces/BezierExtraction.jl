"""
    create_identity(nel::Int, d::Int)::Array{Float64, 3}

Creates a `(d, d, ni)` array where each slice along the last dimension is a `(d, d)` identity matrix.

# Arguments 
- `ni::Int`: number of identity matrices in the array.
- `d::Int`: size of the identity matrices in the array.

# Returns
- `Id::Array{Float64, 3}`: array of identity matrices with size (d, d, ni).
"""
function create_identity(ni::Int, d::Int)
    Id = zeros((d,d,ni))
    for i in 1:ni
        for j in 1:d
            Id[j,j,i] = 1.0
        end
    end

    return Id
end

"""
    extract_bezier_representation(knot_vector::KnotVector, p::Int)

Computes the extraction coefficients of the B-Spline basis functions defined 
by the knot vector `knot_vector` and polynomial degree `p`.

`E[:, j, el]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el`.

# Arguments
- `knot_vector::KnotVector`: knot vector definied the extraction coefficients.
- `p::Int`: piece-wise degree of the basis functions (``p \\geq 0``).

# Returns
- `E::Array{Float64, 3}`: a `(p+1, p+1, nel)` matrix with the extraction coefficients of the b-splines basis functions on every element.
"""
function extract_bezier_representation(knot_vector::KnotVector)
    # number of elements
    nel = length(knot_vector.breakpoints)-1

    # first extraction matrix
    E = create_identity(nel, knot_vector.polynomial_degree+1)
    # this is where knot-insertion coefficients are saved
    alphas = zeros(max(knot_vector.polynomial_degree - 1, 0))

    for el in 1:1:nel
        mult = knot_vector.multiplicity[el+1]

        if mult < knot_vector.polynomial_degree
            numer = knot_vector.breakpoints[el+1] - knot_vector.breakpoints[el]
            r = knot_vector.polynomial_degree - mult

            for j in knot_vector.polynomial_degree-1:-1:mult
                idx = el+1+floor(Int, j/mult)

                if idx > nel+1
                    alphas[j-mult+1] = numer / (knot_vector.breakpoints[end] - knot_vector.breakpoints[el])
                    continue
                end

                alphas[j-mult+1] = numer / (knot_vector.breakpoints[idx] - knot_vector.breakpoints[el])
            end

            for j in 1:r
                s = mult + j - 1
                for k in knot_vector.polynomial_degree:-1:s+1
                    alpha = alphas[k-s]
                    E[k+1, :, el] .=  (@view E[k+1, :, el]).* alpha  .+  (@view E[k, :, el]) .* (1.0-alpha)
                end

                save = r - j + 1
                if el < nel
                    E[save, save:save+j, el+1] .= (@view E[knot_vector.polynomial_degree+1, knot_vector.polynomial_degree-j+1:knot_vector.polynomial_degree+1, el])
                end
            end
        end
    end

    return E
end

"""
    extract_bezier_representation(bspline::BSplineSpace{n,k}, d) where {n, k}

Computes the extraction coefficients of `n`-dimensional `k`-form B-Spline basis functions on dimension `d`. Defined by the elements in `bspline.patch`, with piece-wise polynomial degree `bspline.polynomial_degree` and C^`bspline.regularity`continuity at the interfaces between elements.\n 
`E[:,j, el]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el` in dimension `d`.

# Arguments
- `bspline::BSplineSpace{n,k}`: B-Spline.
- `d::Int`: Dimension.
# Returns
- `E::Array{Float64, 3}`: a `(p+1, p+1, nel)` matrix with the extraction coefficients of the b-splines basis functions on every element.
"""
function extract_bezier_representation(bspline::FunctionSpaces.BSplineSpace)
        return extract_bezier_representation(bspline.knot_vector)
end