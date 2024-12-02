"""
    _integer_sums(sum_indices::Int, num_indices::Int)

This function generates all possible combinations of non-negative integers that sum up to a given value, where each combination has a specified number of elements.

# Arguments
- `sum_indices::Int`: The target sum of the integers in each combination.
- `num_indices::Int`: The number of integers in each combination.

Return Value
The function returns a vector of vectors, where each inner vector represents a combination of integers that sum up to sum_indices. If no valid combinations exist, the function returns an empty vector.
"""
@Memoization.memoize function _integer_sums(sum_indices::Int, num_indices::Int)
    solutions = Vector{Vector{Int}}(undef,0)
    if num_indices == 1
        push!(solutions, [sum_indices])
    elseif num_indices > 1
        for combo in Combinatorics.combinations(0:sum_indices+num_indices-2, num_indices-1)
            s = zeros(Int, num_indices)
            s[1] = combo[1]
            for i in 2:num_indices-1
                s[i] = combo[i] - combo[i-1] - 1
            end
            s[end] = sum_indices+num_indices-2 - combo[num_indices-1]
            push!(solutions, s)
        end
    end
    #TODO should throw argument error if num_indices < 1
    return solutions
end

"""
    ordered_to_linear_index(ord_ind::Vector{Int}, max_ind::Vector{Int})

Convert given ordered pair index to linear index.

# Arguments
- `ord_ind::Vector{Int}`: ordered index pair
- `max_ind::Vector{Int}`: maximum index in each direction

# Returns
- `::Int`: computed linear index
"""
function ordered_to_linear_index(ord_ind::Vector{Int}, max_ind::Vector{Int})
    m = length(max_ind)
    @assert length(ord_ind)==m "Dimension mismatch."
    max_ind = cumprod(max_ind)
    lin_ind = ord_ind[1]
    for i = 2:m
        lin_ind += (ord_ind[i]-1)*max_ind[i-1]
    end
    return lin_ind
end

# Additional method overloads for ordered_to_linear_index
function ordered_to_linear_index(ord_ind::NTuple{m,Int}, max_ind::NTuple{m,Int}) where {m}
    return ordered_to_linear_index(collect(ord_ind), collect(max_ind))
end

function ordered_to_linear_index(ord_ind::Vector{Int}, max_ind::NTuple{m,Int}) where {m}
    return ordered_to_linear_index(ord_ind, collect(max_ind))
end

"""
    linear_to_ordered_index(lin_ind::Int, max_ind::Vector{Int})

Convert given linear index to ordered index pair.

# Arguments
- `lin_ind::Int`: computed linear index
- `max_ind::Vector{Int}`: maximum index in each direction

# Returns
- `::Vector{Int}`: ordered index pair
"""
function linear_to_ordered_index(lin_ind::Int, max_ind::Vector{Int})
    m = length(max_ind)
    ord_ind = zeros(Int,m)
    ord_ind[1] = lin_ind - floor((lin_ind-1)/max_ind[1])*max_ind[1]
    lin_ind = (lin_ind - ord_ind[1])/max_ind[1]
    for i = 2:m
        ord_ind[i] = lin_ind - floor(lin_ind/max_ind[i])*max_ind[i] + 1
        lin_ind = (lin_ind - ord_ind[i] + 1)/max_ind[i]
    end
    @assert lin_ind==0 "Ordered pair computation failed."
    return Int.(ord_ind)
end

# Additional method overloads for linear_to_ordered_index
function linear_to_ordered_index(lin_ind::Int, max_ind::NTuple{m,Int}) where {m}
    return linear_to_ordered_index(lin_ind, collect(max_ind))
end

function linear_to_ordered_index(lin_ind::Int, max_ind::Tuple{Int,Int})
    return linear_to_ordered_index(lin_ind, collect(max_ind))
end

"""
    _get_derivative_idx(der_key::Vector{Int})

Convert the given derivative key to a linear index corresponding to its storage location.

If `local_basis` corresponds to basis evaluations for some `n`-variate function space, then its `k`-th derivatives will all be stored in the location `local_basis[k+1]`. Moreover, the `k`-th derivative corresponding to the key `(i₁,i₂,...,iₙ)` in the location `local_basis[k+1][m]` if the key is the `m`-th key returned by the function `_integer_sums(k, n)`.

# Arguments
- `der_key::Vector{Int}`: A key for the desired derivative order.
# Returns
- `::Int`: The linear index corresponding to the derivative's storage location in basis evaluations.
"""
@Memoization.memoize Dict function _get_derivative_idx(der_key::Vector{Int})
    # generate all valid derivative keys
    all_keys = _integer_sums(sum(der_key),length(der_key))
    # return the index corresponding to der_key
    count = 1
    for key in all_keys
        if der_key == key
            return count
        else
            count += 1
        end
    end
end