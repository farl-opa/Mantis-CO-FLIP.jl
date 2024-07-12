import Combinatorics

"""
    TensorProductSpace{n, F1, F2} <: AbstractFiniteElementSpace{n}

A tensor-product space that is built as the tensor-product of two input spaces.
The functions in this space are `n`-variate where `n` is equal to the sum of the
dimensions of the two input spaces.

# Fields
- `function_space_1::F1`: First input function space
- `function_space_2::F2`: Second input function space
- `boundary_dof_indices::Vector{Int}`: Indices of boundary degrees of freedom
- `data::Dict`: Any relevant data that the user wants to store
"""
struct TensorProductSpace{n, F1, F2} <: AbstractFiniteElementSpace{n}
    function_space_1::F1
    function_space_2::F2
    boundary_dof_indices::Vector{Int}
    data::Dict

    function TensorProductSpace(function_space_1::F1, function_space_2::F2, data::Dict) where {F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
        n = get_n(function_space_1) + get_n(function_space_2)
        new{n,F1,F2}(function_space_1, function_space_2, Vector{Int}(undef,0), data)
    end

    function TensorProductSpace(function_space_1::F1, function_space_2::F2) where {F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
        return TensorProductSpace(function_space_1, function_space_2, Dict())
    end
end

"""
    get_n(tp::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Get the total dimension of the tensor-product space.

# Arguments
- `tp::TensorProductSpace{n, F1, F2}`: The tensor-product space

# Returns
- `::Int`: The total dimension of the space
"""
function get_n(tp::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    return n
end

"""
    get_num_elements(tp_space::TensorProductSpace)

Returns the total number of elements for the partition over which the function space is defined.

# Arguments
- `tp_space::TensorProductSpace`: tensor-product space

# Returns
- `::Int`: total number of elements
"""
function get_num_elements(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    # Multiply the number of elements in each constituent space
    return prod(_get_num_elements_per_space(tp_space))
end

"""
    _get_num_elements_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Helper function to get the number of elements for each constituent space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: tensor-product space

# Returns
- `::Tuple{Int,Int}`: number of elements in each constituent space
"""
function _get_num_elements_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    return (get_num_elements(tp_space.function_space_1), get_num_elements(tp_space.function_space_2))
end

"""
    get_dim(tp_space::TensorProductSpace)

Returns the dimension of the tensor-product function space.

# Arguments
- `tp_space::TensorProductSpace`: Tensor-product function space

# Returns
- `::Int`: The dimension of the space
"""
function get_dim(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    # Multiply the dimensions of each constituent space
    return prod(_get_dim_per_space(tp_space))
end

"""
    _get_dim_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Helper function to get the dimension of each constituent space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: tensor-product space

# Returns
- `::Tuple{Int,Int}`: dimensions of each constituent space
"""
function _get_dim_per_space(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    return (get_dim(tp_space.function_space_1), get_dim(tp_space.function_space_2))
end

"""
    get_max_local_dim(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Get the maximum local dimension of the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: tensor-product space

# Returns
- `::Int`: maximum local dimension
"""
function get_max_local_dim(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    # Multiply the maximum local dimensions of each constituent space
    return get_max_local_dim(tp_space.function_space_1) * get_max_local_dim(tp_space.function_space_2)
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
function ordered_to_linear_index(ord_ind::Tuple{m,Int}, max_ind::Tuple{m,Int}) where {m}
    return ordered_to_linear_index(collect(ord_ind), collect(max_ind))
end

function ordered_to_linear_index(ord_ind::Tuple{Int, Int}, max_ind::Tuple{Int,Int})
    return ordered_to_linear_index(collect(ord_ind), collect(max_ind))
end

function ordered_to_linear_index(ord_ind::Vector{Int}, max_ind::Tuple{Int,Int})
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
function linear_to_ordered_index(lin_ind::Int, max_ind::Tuple{m,Int}) where {m}
    return linear_to_ordered_index(lin_ind, collect(max_ind))
end

function linear_to_ordered_index(lin_ind::Int, max_ind::Tuple{Int,Int})
    return linear_to_ordered_index(lin_ind, collect(max_ind))
end

"""
    get_boundary_dof_indices(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}

Get the indices of boundary degrees of freedom for the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: tensor-product space

# Returns
- `::Vector{Int}`: indices of boundary degrees of freedom
"""
function get_boundary_dof_indices(tp_space::TensorProductSpace{n, F1, F2}) where {n, F1, F2}
    if length(tp_space.boundary_dof_indices)==0
        # Use default tensor-product dofs
        boundary_dofs_1, boundary_dofs_2 = _get_boundary_dof_indices_per_dim(tp_space)
        tp_dim = _get_dim_per_space(tp_space)
        
        # Calculate total number of boundary dofs
        n_boundary_dofs = tp_dim[1] * length(boundary_dofs_2) + tp_dim[2] * length(boundary_dofs_1) - length(boundary_dofs_1) * length(boundary_dofs_2)
        
        boundary_dofs = Vector{Int}(undef,n_boundary_dofs)
        
        # Compute boundaries aligned with function space 1
        idx = 1
        for id ∈ Iterators.product(1:tp_dim[1], boundary_dofs_2)
            boundary_dofs[idx] = ordered_to_linear_index(id, tp_dim)
            idx += 1
        end
        
        # Compute boundaries aligned with function space 2
        for id ∈ Iterators.product(boundary_dofs_1, 1:tp_dim[2])
            if !any(boundary_dofs_2 .== id[2])
                boundary_dofs[idx] = ordered_to_linear_index(id, tp_dim)
                idx += 1
            end
        end
        
        return boundary_dofs
    else
        return tp_space.boundary_dof_indices
    end
end

"""
    get_support(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1, F2}

Get the support of a basis function in the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: tensor-product space
- `basis_id::Int`: ID of the basis function

# Returns
- `::Vector{Int}`: support of the basis function
"""
function get_support(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1, F2}
    max_ind_els = _get_num_elements_per_space(tp_space)
    support_per_dim = _get_support_per_dim(tp_space, basis_id)
    
    support = Vector{Int}(undef, prod(length.(support_per_dim)))
    idx = 1
    for id ∈ Iterators.product(support_per_dim[1], support_per_dim[2])
        support[idx] = ordered_to_linear_index(id, max_ind_els)
        idx += 1
    end
    
    return support
end

"""
    _get_support_per_dim(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1, F2}

Helper function to get the support of a basis function in each constituent space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: tensor-product space
- `basis_id::Int`: ID of the basis function

# Returns
- `::Tuple{Vector{Int},Vector{Int}}`: support in each constituent space
"""
function _get_support_per_dim(tp_space::TensorProductSpace{n, F1, F2}, basis_id::Int) where {n, F1, F2}
    max_ind_basis = _get_dim_per_space(tp_space)
    ordered_index = linear_to_ordered_index(basis_id, max_ind_basis)
    return (get_support(tp_space.function_space_1, ordered_index[1]), 
            get_support(tp_space.function_space_2, ordered_index[2]))
end

"""
    get_extraction(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}

Get the extraction operator for an element in the tensor-product space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: tensor-product space
- `el_id::Int`: ID of the element

# Returns
- `::Tuple{Matrix{Float64},Vector{Int}}`: extraction coefficients and basis indices
"""
function get_extraction(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}
    max_ind_basis = _get_dim_per_space(tp_space)
    extraction_per_dim = _get_extraction_per_dim(tp_space, el_id)
    
    # Compute Kronecker product of extraction coefficients
    extraction_coeffs = kron(extraction_per_dim[2][1], extraction_per_dim[1][1])
    
    # Compute basis indices
    basis_indices = Vector{Int}(undef, prod(map(extraction -> length(extraction[2]), extraction_per_dim)))
    idx = 1
    for basis ∈ Iterators.product(extraction_per_dim[1][2], extraction_per_dim[2][2])
        basis_indices[idx] = ordered_to_linear_index(basis, max_ind_basis)
        idx += 1
    end
    
    return extraction_coeffs, basis_indices
end

"""
    _get_extraction_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}

Helper function to get the extraction operator for each constituent space.

# Arguments
- `tp_space::TensorProductSpace{n, F1, F2}`: tensor-product space
- `el_id::Int`: ID of the element

# Returns
- `::Tuple{Tuple{Matrix{Float64},Vector{Int}},Tuple{Matrix{Float64},Vector{Int}}}`: extraction operators for each constituent space
"""
function _get_extraction_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int) where {n, F1, F2}
    max_ind_el = _get_num_elements_per_space(tp_space)
    ordered_index = linear_to_ordered_index(el_id, max_ind_el)
    return (get_extraction(tp_space.function_space_1,ordered_index[1]),
            get_extraction(tp_space.function_space_2, ordered_index[2]))
end

"""
    get_local_basis(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}

Compute the local basis functions and their derivatives for a tensor product space element.

# Arguments
- `tp_space`: The tensor product space
- `el_id`: Element ID
- `xi`: Tuple of vectors representing evaluation points in each dimension
- `nderivatives`: Number of derivatives to compute

# Returns
- `Dict{NTuple{n,Int}, Matrix{Float64}}`: Dictionary mapping derivative orders to basis function values
"""
function get_local_basis(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    # Get dimensions of constituent spaces
    n1 = get_n(tp_space.function_space_1)
    n2 = get_n(tp_space.function_space_2)

    # Compute local basis for each constituent space
    local_basis_per_dim = _get_local_basis_per_dim(tp_space, el_id, xi, nderivatives)

    # Generate keys for all possible derivative combinations
    der_keys = _integer_sums(nderivatives, n+1)
    
    # Initialize dictionary to store local basis functions and derivatives
    local_basis = Dict{NTuple{n,Int}, Matrix{Float64}}(i[1:n] => Matrix{Float64}(undef,1,1) for i in der_keys)

    # Compute tensor product of constituent basis functions for each derivative combination
    for key in keys(local_basis)
        tmp = kron(local_basis_per_dim[2][key[n1+1:n]...], local_basis_per_dim[1][key[1:n1]...])
        local_basis[key] = tmp
    end

    return local_basis
end

"""
    _get_local_basis_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}

Helper function to compute local basis functions for each constituent space of a tensor product space.

# Arguments
- `tp_space`: The tensor product space
- `el_id`: Element ID
- `xi`: Tuple of vectors representing evaluation points in each dimension
- `nderivatives`: Number of derivatives to compute

# Returns
- `Tuple`: Local basis functions for each constituent space
"""
function _get_local_basis_per_dim(tp_space::TensorProductSpace{n, F1, F2}, el_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, F1 <: AbstractFiniteElementSpace{n1} where {n1}, F2 <: AbstractFiniteElementSpace{n2} where {n2}}
    # Get number of elements in each constituent space
    max_ind_el = _get_num_elements_per_space(tp_space)

    # Get dimensions of constituent spaces
    n1 = get_n(tp_space.function_space_1)
    n2 = get_n(tp_space.function_space_2)

    # Convert linear element ID to ordered index
    ordered_index = linear_to_ordered_index(el_id, max_ind_el)

    # Split evaluation points for each constituent space
    xi_1 = xi[1:n1]
    xi_2 = xi[n1+1:n]

    # Compute local basis for each constituent space
    return get_local_basis(tp_space.function_space_1, ordered_index[1], xi_1, nderivatives), 
           get_local_basis(tp_space.function_space_2, ordered_index[2], xi_2, nderivatives)
end
