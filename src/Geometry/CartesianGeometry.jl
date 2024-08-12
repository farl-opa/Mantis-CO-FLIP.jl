import LinearAlgebra
struct CartesianGeometry{n} <: AbstractAnalyticalGeometry{n}
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
        n_elements = length.(breakpoints) .- 1
        cartesian_idxs = CartesianIndices(n_elements)
        return new{n}(n_elements, breakpoints, cartesian_idxs)
    end
end

function get_num_elements(geometry::CartesianGeometry{n}) where {n}
    return prod(geometry.n_elements)
end

# function get_boundary_indices(geometry::CartesianGeometry{n}) where {n}
#     return    
# end

# function get_num_boundary_elements(geometry::CartesianGeometry{n}) where {n}
#     if any(geometry.n_elements) == 1
#         return
#     else
#         return 2*sum(geometry.n_elements)
#     end
# end

# function get_num_boundary_elements(geometry::CartesianGeometry{1,1})
#     return minimum([2, geometry.n_elements])
# end

function get_domain_dim(::CartesianGeometry{n}) where {n}
    return n
end

function get_image_dim(::CartesianGeometry{n}) where {n}
    return n
end

function evaluate(geometry::CartesianGeometry{n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])
    univariate_points = ntuple( k -> (1 .- ξ[k]) .* geometry.breakpoints[k][ordered_idx[k]] + ξ[k] .* geometry.breakpoints[k][ordered_idx[k]+1], n)
    
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

function jacobian(geometry::CartesianGeometry{n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    # Get the multi-index of the element
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])
    
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
