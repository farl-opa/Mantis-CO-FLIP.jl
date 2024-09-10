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

# Methods for ease of geometry creation

@doc raw"""
    create_cartesian_geometry(starting_point::NTuple{n, Float64}, box_size::NTuple{n, Float64}, num_elements::NTuple{n, Int}) where {n}

Create a Cartesian geometry based on the provided starting point, box size, and number of elements. The geometry is created in `n`-dimensional space, where `n` is determined by the length of the input tuples.

# Arguments
- `starting_point::NTuple{n, Float64}`: The coordinates of the starting point of the geometry.
- `box_size::NTuple{n, Float64}`: The size of the bounding box in each dimension.
- `num_elements::NTuple{n, Int}`: The number of elements in each dimension.

# Returns
- `::CartesianGeometry{n}`: The generated Cartesian geometry structure.

# Examples
```julia
# Create a 3D Cartesian geometry
start_3d = (0.0, 0.0, 0.0)
size_3d = (10.0, 5.0, 2.0)
elements_3d = (100, 50, 20)
geometry_3d = create_cartesian_geometry(start_3d, size_3d, elements_3d)
```
"""
function create_cartesian_geometry(starting_point::NTuple{n, Float64}, box_size::NTuple{n, Float64}, num_elements::NTuple{n, Int}) where {n}

    all(box_size .> 0.0) ? nothing : throw(ArgumentError("The argumente 'box_size' must be greater than 0."))

    breakpoints = Tuple(collect(LinRange(starting_point[i], starting_point[i]+box_size[i], num_elements[i]+1)) for i ∈ 1:n)

    return CartesianGeometry(breakpoints)
end

@doc raw"""
    create_cartesian_geometry(starting_point::Float64, box_size::Float64, num_elements::Int)

Create a 1D Cartesian geometry based on the provided starting point, box size, and number of elements. It creates a linear distribution of points from the starting point, spaced evenly across the specified box size.

# Arguments
- `starting_point::Float64`: The coordinate of the starting point of the geometry.
- `box_size::Float64`: The size of the bounding box along the single dimension.
- `num_elements::Int`: The number of elements along the dimension.

# Returns
- `::CartesianGeometry{1}`: The generated 1D Cartesian geometry structure.

# Examples
```julia
# Create a 1D Cartesian geometry
start_1d = 0.0
size_1d = 10.0
elements_1d = 100
geometry_1d = create_cartesian_geometry(start_1d, size_1d, elements_1d)
```
"""
function create_cartesian_geometry(starting_point::Float64, box_size::Float64, num_elements::Int)
    return create_cartesian_geometry((starting_point,), (box_size,), (num_elements,))
end