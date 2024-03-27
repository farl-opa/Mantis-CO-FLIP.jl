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
    create_uniform_knot_vector(left_endpoint::Float64, right_endpoint::Float64, nel::Int, p::Int, k::Vector{Int})

Creates a uniform knot vector corresponding to `nel*p - sum(k) + 1` 
B-splines basis functions of polynomial degree `p` and continuity `k[i]` over `nel` 
equally spaced elements in the interval between`left_endpoint` and `right_endpoint`. 
See https://en.wikipedia.org/wiki/B-spline#Definition.

# Arguments
- `left_endpoint::Float64`: first value of the knot vector.
- `right_endpoint::Float64`: last value of the knot vector.
- `nel::Int`: number of elements.
- `p::Int`: degree of the polynomial (``p \\geq 0``).
- `k::Vector{Int}`: continuity at the interfaces between elements. (`` -1 \\leq k[i] \\leq p``).

# Returns 
- `::KnotVector`: uniform knot vector.
"""
function create_uniform_knot_vector(left_endpoint::Float64, right_endpoint::Float64, nel::Int, p::Int, k::Vector{Int})
    breakpoints = collect(range(left_endpoint, right_endpoint, nel+1))
    multiplicity = ones(Int, length(breakpoints))

    multiplicity[1] = p+1
    multiplicity[end] = p+1
    for i in eachindex(k)
        multiplicity[i+1] = p - k[i]
    end
    
    return KnotVector(breakpoints, multiplicity)
end

"""
    extract_bezier_representation(left_endpoint::Float64, right_endpoint::Float64, nel::Int, p::Int, k::Int)::Array{Float64, 3}

Computes the extraction coefficients of the B-Spline basis functions defined 
between `left_endpoint` and `right_endpoint`, subdivided into `nel` elements, 
with piece-wise polynomial degree `p` and C^`k`continuity at the interfaces between elements.

`E[:, j, el]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el`.

# Arguments
- `left_endpoint::Float64`: first value of the domain.
- `right_endpoint::Float64`: last value of the domain.
- `nel::Int`: number of elements.
- `p::Int`: piece-wise degree of the basis functions (``p \\geq 0``).
- `k::Int`: continuity of the basis functions at the interfaces (``k < p``).

# Returns
- `E::Array{Float64, 3}`: a `(p+1, p+1, nel)` matrix with the extraction coefficients of the b-splines basis functions on every element.
"""
function extract_bezier_representation(left_endpoint::Float64, right_endpoint::Float64, nel::Int, p::Int, k::Vector{Int})
    # create uniform knotvector using the provided parameters
    knot_vector = create_uniform_knot_vector(left_endpoint, right_endpoint, nel, p, k)
    return extract_bezier_representation(knot_vector, p)
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
function extract_bezier_representation(knot_vector::KnotVector, p::Int)
    # number of elements
    nel = length(knot_vector.breakpoints)-1

    # first extraction matrix
    E = create_identity(nel, p+1)
    # this is where knot-insertion coefficients are saved
    alphas = zeros(max(p - 1, 0))

    for el in 1:1:nel
        mult = knot_vector.multiplicity[el+1]

        if mult < p
            numer = knot_vector.breakpoints[el+1] - knot_vector.breakpoints[el]
            r = p - mult

            for j in p-1:-1:mult
                idx = el+1+floor(Int, j/mult)

                if idx > nel+1
                    alphas[j-mult+1] = numer / (knot_vector.breakpoints[end] - knot_vector.breakpoints[el])
                    continue
                end

                alphas[j-mult+1] = numer / (knot_vector.breakpoints[idx] - knot_vector.breakpoints[el])
            end

            for j in 1:r
                s = mult + j - 1
                for k in p:-1:s+1
                    alpha = alphas[k-s]
                    E[k+1, :, el] .=  (@view E[k+1, :, el]).* alpha  .+  (@view E[k, :, el]) .* (1.0-alpha)
                end

                save = r - j + 1
                if el < nel
                    E[save, save:save+j, el+1] .= (@view E[p+1, p-j+1:p+1, el])
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
function extract_bezier_representation(bspline::FunctionSpaces.BSplineSpace{n}, d) where {n}
        return extract_bezier_representation(bspline.patch.breakpoints[d][1], 
                                  bspline.patch.breakpoints[d][end], 
                                  size(bspline.patch)[d], 
                                  bspline.polynomial_degree[d], 
                                  bspline.regularity[d])
end

"""
    extract_bezier_representation(bspline::BSplineSpace{n,k})  where {n, k}

Computes the extraction coefficients of `n`-dimensional `k`-form B-Spline basis functions. Defined by the elements in `bspline.patch`, with piece-wise polynomial degree `bspline.polynomial_degree` and C^`bspline.regularity`continuity at the interfaces between elements.\n 
`E[d][:,j, el]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el` in dimension `d`.

# Arguments
- `bspline::BSplineSpace{n,k}`: B-Spline.
# Returns
- `E::NTuple{n, Array{Float64, 3}`: a `(p+1, p+1, nel)` matrix with the extraction coefficients of the b-splines basis functions on every element.
"""
function extract_bezier_representation(bspline::FunctionSpaces.BSplineSpace{n}) where {n}
    return NTuple{n, Array{Float64, 3}}(extract_bezier_representation(bspline,d) for d in 1:1:n)
end