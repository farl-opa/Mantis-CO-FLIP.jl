import Combinatorics

"""
    TensorProductSpace{m} <: AbstractFiniteElementSpace

A tensor-product space that is built as the tensor-product of `m` input spaces. The functions in this space are `n`-variate where `n` is equal to the sum of all `n_i` such that the i-th input space is `n_i`-variate.

# Fields
- `function_spaces::NTuple{m,AbstractFiniteElementSpace} where {m}`: collection of uni or multivariate function spaces
- `data::Dict`: any relevant data that the user wants to store
"""
struct TensorProductSpace{n, F1, F2} <: AbstractFiniteElementSpace{n} #where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    function_space_1::F1
    function_space_2::F2
    data::Dict

    function TensorProductSpace(function_space_1::F1, function_space_2::F2, data::Dict) where {F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
        n = get_n(function_space_1) + get_n(function_space_2)
        new{n,F1,F2}(function_space_1, function_space_2, data)
    end

    function TensorProductSpace(function_space_1::F1, function_space_2::F2) where {F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
        return TensorProductSpace(function_space_1, function_space_2, Dict())
    end
end

function get_n(tp::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    return n
end

@doc raw"""
    get_num_elements(tp_space::TensorProductSpace)

Returns number of total number of elements for the partition over which the function space is defined.

# Arguments 
- `tp_space::TensorProductSpace`: tensor-product space

# Returns
- `::Int`: total number of elements
"""
function get_num_elements(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    return prod(_get_num_elements_per_space(tp_space))
end

function _get_num_elements_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    return (get_num_elements(tp_space.function_space_1), get_num_elements(tp_space.function_space_2))
end

"""
    get_dim(tp_space::TensorProductSpace)

Returns the dimension of the tensor-product function space `tp_space`.

# Arguments
- `tp_space::TensorProductSpace)`: Tensor-product function space.
# Returns
- `::Int`: The dimension of the space.
"""
function get_dim(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    return prod(_get_dim_per_space(tp_space))
end

function _get_dim_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    return (get_dim(tp_space.function_space_1), get_dim(tp_space.function_space_2))
end

@doc raw"""
    ordered_to_linear_index(ord_ind::Vector{Int}, max_ind::Vector{Int})

Convert given ordered pair index to linear index:
linear_index = ordered_pair[1] + (ord_ind[2]-1)*max_ind[1] + ...

# Arguments 
- `ord_ind::Vector{Int}`: ordered index pair
- `max_ind::Vector{Int}`: maximum index in each direction

# Returns
- `lin_ind::Int`: computed linear index
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

function ordered_to_linear_index(ord_ind::Tuple{m,Int}, max_ind::Tuple{m,Int}) where {m}
    return ordered_to_linear_index(collect(ord_ind), collect(max_ind))
end

function ordered_to_linear_index(ord_ind::Tuple{Int, Int}, max_ind::Tuple{Int,Int})
    return ordered_to_linear_index(collect(ord_ind), collect(max_ind))
end

function ordered_to_linear_index(ord_ind::Vector{Int}, max_ind::Tuple{Int,Int})
    return ordered_to_linear_index(ord_ind, collect(max_ind))
end

@doc raw"""
    linear_to_ordered_index(, max_ind::Vector{Int})

Convert given linear index to ordered index pair such that:
linear_index = ordered_pair[1] + (ord_ind[2]-1)*max_ind[1] + ...

# Arguments 
- `lin_ind::Int`: computed linear index
- `max_ind::Vector{Int}`: maximum index in each direction

# Returns
- `ord_ind::Vector{Int}`: ordered index pair
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

function linear_to_ordered_index(lin_ind::Int, max_ind::Tuple{m,Int}) where {m}
    return linear_to_ordered_index(lin_ind, collect(max_ind))
end

function linear_to_ordered_index(lin_ind::Int, max_ind::Tuple{Int,Int})
    return linear_to_ordered_index(lin_ind, collect(max_ind))
end

function get_support(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    max_ind_els =_get_num_elements_per_space(tp_space)

    support_per_dim = _get_support_per_dim(tp_space, basis_id)

    support = Vector{Int}(undef, prod(length.(support_per_dim)))

    idx = 1
    for id ∈ Iterators.product(support_per_dim[1], support_per_dim[2])
        support[idx] = ordered_to_linear_index(id, max_ind_els)
        idx += 1 
    end

    return support
end

function _get_support_per_dim(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    max_ind_basis = _get_dim_per_space(tp_space)
    ordered_index = linear_to_ordered_index(basis_id, max_ind_basis)

    return (get_support(tp_space.function_space_1, ordered_index[1]), get_support(tp_space.function_space_2, ordered_index[2]))
end

function get_extraction(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    max_ind_basis = _get_dim_per_space(tp_space)

    extraction_per_dim = _get_extraction_per_dim(tp_space, el_id)

    extraction_coeffs = kron(extraction_per_dim[2][1], extraction_per_dim[1][1])
    basis_indices = Vector{Int}(undef, prod(map(extraction -> length(extraction[2]), extraction_per_dim)))

    idx = 1
    for basis ∈ Iterators.product(extraction_per_dim[1][2], extraction_per_dim[2][2])
        basis_indices[idx] = ordered_to_linear_index(basis, max_ind_basis)
        idx += 1
    end

    return extraction_coeffs, basis_indices
end

function _get_extraction_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    max_ind_el = _get_num_elements_per_space(tp_space)

    ordered_index = linear_to_ordered_index(el_id, max_ind_el)

    return (get_extraction(tp_space.function_space_1,ordered_index[1]), get_extraction(tp_space.function_space_2, ordered_index[2]))
end

function get_local_basis(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    n1 = get_n(tp_space.function_space_1)
    n2 = get_n(tp_space.function_space_2)

    local_basis_per_dim = _get_local_basis_per_dim(tp_space, el_id, xi, nderivatives)

    der_keys = _integer_sums(nderivatives, n)
    local_basis = Dict{NTuple{n,Int}, Matrix{Float64}}(i => Matrix{Float64}(undef,1,1) for i in der_keys)

    # kronecker product of constituent basis functions
    for key in keys(local_basis)
        tmp = kron(local_basis_per_dim[2][key[n1+1:n]...], local_basis_per_dim[1][key[1:n1]...])
        local_basis[key] = tmp
    end

    return local_basis
end

function _get_local_basis_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    max_ind_el = _get_num_elements_per_space(tp_space)

    n1 = get_n(tp_space.function_space_1)
    n2 = get_n(tp_space.function_space_2)

    ordered_index = linear_to_ordered_index(el_id, max_ind_el)

    length(xi[1:n1]) == 1 ? xi_1 = xi[1:n1][1] : xi_1 = xi[1:n1]
    length(xi[n1+1:n]) == 1 ? xi_2 = xi[n1+1:n][1] : xi_2 = xi[n1+1:n]

    return (get_local_basis(tp_space.function_space_1, ordered_index[1], xi_1, nderivatives), get_local_basis(tp_space.function_space_2, ordered_index[2], xi_2, nderivatives))
end

function _integer_sums(n, k)
    if k == 1
        solutions = [n]
    elseif k > 1
        solutions = []
        for combo in Combinatorics.combinations(0:n+k-2, k-1)
            s = (combo[1],)
            for i in 2:k-1
                s = (s..., combo[i] - combo[i-1] - 1)
            end
            s = (s..., n+k-2 - combo[k-1])
            push!(solutions, s)
        end
    end
    return solutions
end