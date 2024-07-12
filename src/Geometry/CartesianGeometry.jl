import LinearAlgebra

"""
    CartesianGeometry{n,n} <: AbstractAnalGeometry{n, n}

A struct representing a Cartesian geometry in n-dimensional space.

# Fields
- `n_elements::NTuple{n,Int}`: Number of elements in each dimension.
- `breakpoints::NTuple{n,Vector{Float64}}`: Breakpoints defining the grid in each dimension.
- `cartesian_idxs::CartesianIndices{n, NTuple{n, Base.OneTo{Int}}}`: Cartesian indices for element indexing.

# Type parameters
- `n`: Dimension of the space.
"""
struct CartesianGeometry{n,n} <: AbstractAnalGeometry{n, n}
    n_elements::NTuple{n,Int}
    breakpoints::NTuple{n,Vector{Float64}}
    cartesian_idxs::CartesianIndices{n, NTuple{n, Base.OneTo{Int}}}

    """
        CartesianGeometry(breakpoints::NTuple{n,Vector{Float64}}) where {n}

    Construct a new CartesianGeometry instance.

    # Arguments
    - `breakpoints`: A tuple of vectors defining the grid points in each dimension.

    # Returns
    A new CartesianGeometry instance.
    """
    function CartesianGeometry(breakpoints::NTuple{n,Vector{Float64}}) where {n}
        n_elements = length.(breakpoints) .- 1  # Calculate number of elements in each dimension
        cartesian_idxs = CartesianIndices(n_elements)  # Create Cartesian indices for element indexing
        return new{n,n}(n_elements, breakpoints, cartesian_idxs)
    end
end

"""
    get_num_elements(geometry::CartesianGeometry{n,n}) where {n}

Get the total number of elements in the CartesianGeometry.

# Arguments
- `geometry`: The CartesianGeometry instance.

# Returns
The total number of elements as an integer.
"""
function get_num_elements(geometry::CartesianGeometry{n,n}) where {n}
    return prod(geometry.n_elements)
end

"""
    get_domain_dim(geometry::CartesianGeometry{n,n}) where {n}

Get the dimension of the domain for a CartesianGeometry.

# Arguments
- `geometry`: The CartesianGeometry instance.

# Returns
The domain dimension `n`.
"""
function get_domain_dim(geometry::CartesianGeometry{n,n}) where {n}
    return n
end

"""
    get_image_dim(geometry::CartesianGeometry{n,n}) where {n}

Get the dimension of the image (codomain) for a CartesianGeometry.

# Arguments
- `geometry`: The CartesianGeometry instance.

# Returns
The image dimension `n`.
"""
function get_image_dim(geometry::CartesianGeometry{n,n}) where {n}
    return n
end

"""
    evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}

Evaluate the CartesianGeometry at specific points in the reference element.

# Arguments
- `geometry`: The CartesianGeometry instance.
- `element_idx`: The index of the element to evaluate.
- `ξ`: A tuple of vectors representing the evaluation points in the reference element.

# Returns
A matrix of evaluated points in the physical space.
"""
function evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])  # Get the multi-index of the element
    
    # Compute univariate points for each dimension
    univariate_points = ntuple(k -> (1 .- ξ[k]) .* geometry.breakpoints[k][ordered_idx[k]] + ξ[k] .* geometry.breakpoints[k][ordered_idx[k]+1], n)
    
    points_tensor_product_idx = CartesianIndices(size.(univariate_points, 1))  # Get the multidimensional indices for the tensor product points

    # Compute coordinates as the tensor product of the unidimensional points
    n_points = prod(size.(ξ, 1))  # Total number of points to evaluate
    x = zeros(Float64, n_points, n)
    for (point_idx, point_cartesian_idx) in enumerate(points_tensor_product_idx)
        for component_idx in 1:n 
            x[point_idx, component_idx] = univariate_points[component_idx][point_cartesian_idx[component_idx]]
        end    
    end

    return x
end

"""
    jacobian(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}

Compute the Jacobian of the CartesianGeometry at specific points in the reference element.

# Arguments
- `geometry`: The CartesianGeometry instance.
- `element_idx`: The index of the element to evaluate.
- `ξ`: A tuple of vectors representing the evaluation points in the reference element.

# Returns
A 3D array representing the Jacobian matrices for each evaluation point.
"""
function jacobian(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])  # Get the multi-index of the element
    
    # Compute the spacing in every direction
    dx = zeros(Float64, n)
    for dim_idx in range(1, n)
        start_breakpoint_idx = ordered_idx[dim_idx]
        end_breakpoint_idx = ordered_idx[dim_idx] + 1
        dx[dim_idx] = geometry.breakpoints[dim_idx][end_breakpoint_idx] - geometry.breakpoints[dim_idx][start_breakpoint_idx]
    end
    
    # Generate the Jacobian for the Cartesian grid
    # For each point, it's a diagonal matrix multiplied by the cell spacings in each direction
    J = zeros(Float64, prod(length.(ξ)), n, n)
    for dim_idx in range(1, n)
        J[:, dim_idx, dim_idx] .= dx[dim_idx]
    end
    
    return J
end
