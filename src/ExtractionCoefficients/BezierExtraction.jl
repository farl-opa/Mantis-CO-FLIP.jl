"""
    create_identity(nel::Int, d::Int)::Array{Float64, 3}

Creates a `(d, d, n)` array where each slice along the first dimension is a `(d, d)` identity matrix.

# Arguments 
- `n::Int`: number of identity matrices in the array.
- `d::Int`: size of the identity matrices in the array.

# Returns
- `Id::Array{Float64, 3}`: array of identity matrices.
"""
function create_identity(n::Int, d::Int)
    Id = zeros((d,d,n))
    for i in 1:n
        for j in 1:d
            Id[j,j,i] = 1.0
        end
    end

    return Id
end

"""
    create_uniform_knot_vector(endpoints::Vector{Float64}, n::Int, p::Int, k::Int)Tuple{Vector{Float64}, Int64}

Creates a uniform knot vector corresponding to `n(p+1)-(n-1)(k+1)` 
B-splines basis functions of polynomial degree `p` and continuity `k` over `n` 
equally spaced elements in the interval `[endpoints[1], endpoints[2]]`. 
See https://en.wikipedia.org/wiki/B-spline#Definition.

# Arguments
- `endpoints::Vector{Float64}`: the first and last value of the knot vector.
- `n::Int`: number of elements.
- `p::Int`: degree of the polynomial (``p \\geq 0``).
- `k::Int`: continuity at the interfaces between elements. (`` -1 \\leq k \\leq p``).

# Returns 
- `knt::Vector{Float64}`: knot vector.
"""
function create_uniform_knot_vector(endpoints::Vector{Float64}, n::Int, p::Int, k::Int)::Vector{Float64}
    mesh = range(endpoints[1], endpoints[2], n+1)
    n_knt = 2*(p+1)+(p-k)*(n-1)
    knt = fill(endpoints[1], n_knt)

    curr_idx = p+1
    for i = 2:n
        knt[curr_idx+1 : curr_idx+p-k] .= mesh[i]
        curr_idx += p-k
    end
    knt[end-p:end] .= mesh[end]

    return knt
end

"""
    extract_bezier_representation(endpoints::Vector{Float64}, n::Int, p::Int, k::Int)::Array{Float64, 3}

Computes the extraction coefficients of the B-Spline basis functions defined 
between `[endpoints[1], endpoints[2]]`, subdivided into `n` elements, 
with piece-wise polynomial degree `p` and C^`k`continuity at the interfaces between elements.

`E[:, j, el]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el`.

# Arguments
- `endpoints::Vector{Float64}`: endpoints of the domain.
- `n::Int`: number of elements.
- `p::Int`: piece-wise degree of the basis functions (``p \\geq 0``).
- `k::Int`: continuity of the basis functions at the interfaces (``k < p``).

# Returns
- `E::Array{Float64, 3}`: extraction coefficients of the b-splines basis functions on every element.
"""
function extract_bezier_representation(endpoints::Vector{Float64}, nel::Int, p::Int, k::Int)
    # create uniform knotvector using the provided parameters
    knt = create_uniform_knot_vector(endpoints, nel, p, k)
    return extract_bezier_representation(knt::Vector{Float64}, p::Int)
end

function extract_bezier_representation(knt::Vector{Float64}, p::Int)
    # number of elements
    nel = length(unique(knt))-1
    # number of knots in the knotvector
    n_knt = length(knt)

    # assuming an open knot vector, knt[a] is the last repetition of the first knot
    a = p
    # next knot
    b = a + 1
    # Bezier element being processed
    nb = 1
    # first extraction matrix
    E = create_identity(nel, p+1)
    # this is where knot-insertion coefficients are saved
    alphas = zeros(max(p - 1, 0))
    while b < n_knt
        # save the index of the current knot
        i = b
        # find the last occurrence of the current knot
        while b < n_knt - 1 && knt[b+2] == knt[b+1]
            b += 1
        end
        # multiplicity of the current knot
        mult = b - i + 1
        # if multiplicity is less than DEGREE, smoothness is at least C0, and extraction may differ from an identity matrix
        if mult < p
            numer = knt[b+1] - knt[a+1]
            # smoothness of splines
            r = p - mult
            # compute linear combination coefficients
            for j in p-1:-1:mult
                alphas[j-mult+1] = numer / (knt[a+j+2] - knt[a+1])
            end
            for j in 1:r
                s = mult + j - 1
                for k in p:-1:s+1
                    alpha = alphas[k-s]
                    E[k+1,:, nb] .=  (@view E[k+1,:, nb]).* alpha  .+  (@view E[k, :, nb]) .* (1.0-alpha)
                end
                save = r - j + 1
                if b < n_knt
                    E[save, save:save+j, nb+1] .= (@view E[p+1, p-j+1:p+1, nb])
                end
            end
        end
        # increment the element index
        nb += 1
        if b < n_knt
            a = b
            b += 1
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
- `E::Array{Float64, 3}`: extraction coefficients of the b-splines basis functions on every element in dimension `d`.
"""
function extract_bezier_representation(bspline::FunctionSpaces.BSplineSpace{n,k}, d) where {n, k}
    if allunique(bspline.polynomial_degree[d]) && allunique(bspline.regularity[d])
        return extract_bezier_representation([bspline.patch.breakpoints[d][1], 
                                  bspline.patch.breakpoints[d][end]], 
                                  size(bspline.patch)[d], 
                                  bspline.polynomial_degree[d][1], 
                                  bspline.regularity[d][1])
    else
        error("Not implemented yet...") #Add more general algorithm here.
    end
end

"""
    extract_bezier_representation(bspline::BSplineSpace{n,k})  where {n, k}

Computes the extraction coefficients of `n`-dimensional `k`-form B-Spline basis functions. Defined by the elements in `bspline.patch`, with piece-wise polynomial degree `bspline.polynomial_degree` and C^`bspline.regularity`continuity at the interfaces between elements.\n 
`E[d][:,j, el]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el` in dimension `d`.

# Arguments
- `bspline::BSplineSpace{n,k}`: B-Spline.
# Returns
- `E::NTuple{n, Array{Float64, 3}`: extraction coefficients of the b-splines basis functions on every element across all dimensions.
"""
function extract_bezier_representation(bspline::FunctionSpaces.BSplineSpace{n,k}) where {n,k}
    return NTuple{n, Array{Float64, 3}}(extract_bezier_representation(bspline,d) for d in 1:1:n)
end