import ArgCheck

"""
    create_identity(n::Int, d::Int)::Array{Float64, 3}

Creates a `(n,d,d)` array where each slice along the first dimension is a `(d,d)` identity matrix.
# Arguments 
- `n::Int`: number of identity matrices in the array.
- `d::Int`: size of the identity matrices in the array.
# Returns
- `Id::Array{Float64, 3}`: array of identity matrices.
"""
function create_identity(n::Int,q::Int)::Array{Float64, 3}
    ArgCheck.@argcheck n>0
    ArgCheck.@argcheck q>0

    Id = zeros((n,q,q))
    for i=1:n
        for j=1:q
            Id[i,j,j] = 1.0
        end
    end
    return Id
end

"""
    count_knot_elements(knt::Vector{Float64})::Int

Counts the number of elements `n_els` in a knot vector `knt`.

# Arguments
- `knt::Vector{Float64}`: knot vector used for counting.
# Returns
- `n_els::Int`: number of elements in the knot vector.
"""
function count_knot_elements(knt::Vector{Float64})::Int
    ArgCheck.@argcheck all(diff(knt) .>= 0)

    n_els = 0
    for i = 1:length(knt)-1
        if  knt[i+1] - knt[i] != 0.0
            n_els += 1
        end
    end

    return n_els
end

"""
    vector_extremes_equality(vec::Vector{Number}, n::Int)::Bool

Checks whether the first and last `n` entries of `vec` are independently equal.
# Arguments
- `vec::Vector{Number}`: vector to be checked.
-  `n::Int`: number of entries to check.
# Returns
- `check::Bool`: truth value of the extreme entries being equal.
"""
function vector_extremes_equality(vec::Vector{Number}, n::Int)::Bool
    ArgCheck.@argcheck length(vec) >= 2*n

    first = all(vec[1] .== vec[2:n])
    last = all(vec[end] .== vec[end-n+1:end])

    check = first && last

    return check
end

"""
    uniform_knot_vector(endpoints::Vector{Float64}, n::Int, p::Int, k::Int)Tuple{Vector{Float64}, Int64}

Creates a uniform knot vector to be used for bezier extraction. The first and last knots, corresponding to the endpoints, are repeated `p+1` times and all other knots are repeated `p-k`.
# Arguments
- `endpoints::Vector{Float64}`: the first and last value of the knot vector.
- `n::Int`: number of elements.
- `p::Int`: degree of the polynomial (``p \\geq 0``).
- `k::Int`: continuity at the interfaces between elements. (`` -1 \\leq k \\leq p``).
# Returns 
- `knt::Vector{Float64}`: knot vector.
"""
function uniform_knot_vector(endpoints::Vector{Float64}, n::Int, p::Int, k::Int)::Vector{Float64}
    ArgCheck.@argcheck endpoints[1]<endpoints[2]
    ArgCheck.@argcheck n>0
    ArgCheck.@argcheck p>=0
    ArgCheck.@argcheck k<p

    mesh = range(endpoints[1], endpoints[2], n+1)
    n_knt = 2*(p+1)+(p-k)*(n-1)
    knt = ones(n_knt) .* endpoints[1]

    curr_idx = p+1
    for i = 2:n
        knt[curr_idx+1 : curr_idx+p-k] .= mesh[i]
        curr_idx += p-k
    end
    knt[end-p:end] .= mesh[end]

    return knt
end

"""
    bezier_extraction(brk::Vector{Float64}, p::Int, k::Int)::Array{Float64, 3}

Computes the extraction coefficients of the B-Spline basis functions based on the knot vector `knt` and degree `p`.\n 
`E[el, :, j]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el`.

# Arguments
- `knt::Vector{Float64}`: knot vector.
- `p::Int`: degree of the polynomial (``p \\geq 0``).
# Returns
- `E::Array{Float64, 3}`: extraction coefficients of the b-splines basis functions on every element.
"""
function bezier_extraction(knt::Vector{Float64}, p::Int)::Array{Float64, 3}
    # Check for argument errors
    ArgCheck.@argcheck p>=0
    ArgCheck.@argcheck vector_extremes_equality(knt, p+1)

    n = count_knot_elements(knt)
    n_knt = length(knt)

    # assuming an open knot vector, knt[a] is the last repetition of the first knot
    a = p
    # next knot
    b = a + 1
    # Bezier element being processed
    nb = 1
    # first extraction matrix
    E = create_identity(n, p+1)
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
                    E[nb,k+1, :] .=  (@view E[nb,k+1, :]).* alpha  .+  (@view E[nb,k, :]) .* (1.0-alpha)
                end
                save = r - j + 1
                if b < n_knt
                    E[nb+1,save, save:save+j] .= (@view E[nb,p+1, p-j+1:p+1])
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
    bezier_extraction(endpoints::Vector{Float64}, n::Int, p::Int, k::Int)::Array{Float64, 3}

Computes the extraction coefficients of the B-Spline basis functions defined between some `endpoints`, subdivided into `n` elements, with piece-wise degree `p` and C^`k`continuity at the interfaces between elements.\n 
`E[el, :, j]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el`.

# Arguments
- `endpoints::Vector{Float64}`: endpoints of the domain.
- `n::Int`: number of elements.
- `p::Int`: piece-wise degree of the basis functions (``p \\geq 0``).
- `k::Int`: continuity of the basis functions at the interfaces (``k < p``).
# Returns
- `E::Array{Float64, 3}`: extraction coefficients of the b-splines basis functions on every element.
"""
function bezier_extraction(endpoints::Vector{Float64}, n::Int, p::Int, k::Int)::Array{Float64, 3}
    # Check for argument errors
    ArgCheck.@argcheck p>=0
    ArgCheck.@argcheck k<p

    knt = uniform_knot_vector(endpoints, n, p, k)
    n_knt = 2*(p+1)+(p-k)*(n-1)

    # assuming an open knot vector, knt[a] is the last repetition of the first knot
    a = p
    # next knot
    b = a + 1
    # Bezier element being processed
    nb = 1
    # first extraction matrix
    E = create_identity(n, p+1)
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
                    E[nb,k+1, :] .=  (@view E[nb,k+1, :]).* alpha  .+  (@view E[nb,k, :]) .* (1.0-alpha)
                end
                save = r - j + 1
                if b < n_knt
                    E[nb+1,save, save:save+j] .= (@view E[nb,p+1, p-j+1:p+1])
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



    
