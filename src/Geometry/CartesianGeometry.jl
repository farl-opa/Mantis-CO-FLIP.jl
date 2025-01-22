import LinearAlgebra
struct CartesianGeometry{manifold_dim} <: AbstractAnalyticalGeometry{manifold_dim}
    n_elements::NTuple{manifold_dim,Int}
    breakpoints::NTuple{manifold_dim,Vector{Float64}}
    cartesian_idxs::CartesianIndices{manifold_dim, NTuple{manifold_dim, Base.OneTo{Int}}}

    """
        CartesianGeometry(breakpoints::NTuple{manifold_dim,Vector{Float64}}) where {manifold_dim}

    Construct a new CartesianGeometry instance.

    # Arguments
    - `breakpoints`: A tuple of vectors defining the grid points in each dimension.

    # Returns
    A new CartesianGeometry instance.
    """
    function CartesianGeometry(breakpoints::NTuple{manifold_dim,T}) where {manifold_dim, N <: Number, T <: AbstractVector{N}}
        n_elements = length.(breakpoints) .- 1
        cartesian_idxs = CartesianIndices(n_elements)
        return new{manifold_dim}(n_elements, breakpoints, cartesian_idxs)
    end
end

function get_num_elements(geometry::CartesianGeometry{manifold_dim}) where {manifold_dim}
    return prod(geometry.n_elements)
end

function get_domain_dim(::CartesianGeometry{manifold_dim}) where {manifold_dim}
    return manifold_dim
end

function get_image_dim(::CartesianGeometry{manifold_dim}) where {manifold_dim}
    return manifold_dim
end

function evaluate(geometry::CartesianGeometry{manifold_dim}, element_idx::Int, ξ::NTuple{manifold_dim,Vector{Float64}}) where {manifold_dim}

    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])
    univariate_points = ntuple(manifold_dim) do k
        dim_points = zeros(length(ξ[k])) # initialize vector for current dimension

        for point in eachindex(dim_points)
            dim_points[point] = (1-ξ[k][point])*geometry.breakpoints[k][ordered_idx[k]] +
            ξ[k][point]*geometry.breakpoints[k][ordered_idx[k]+1]
        end

        return dim_points
    end

    points_tensor_product_idx = CartesianIndices(size.(univariate_points, 1))  # Get the multidimensional indices for the tensor product points

    # Compute coordinates as the tensor product of the unidimensional points
    n_points = prod(size.(ξ, 1))  # Total number of points to evaluate
    x = zeros(Float64, n_points, manifold_dim)
    for (point_idx, point_cartesian_idx) in enumerate(points_tensor_product_idx)
        for component_idx in 1:manifold_dim
            x[point_idx, component_idx] = univariate_points[component_idx][point_cartesian_idx[component_idx]]
        end
    end

    return x
end

function jacobian(geometry::CartesianGeometry{manifold_dim}, element_idx::Int, ξ::NTuple{manifold_dim,Vector{Float64}}) where {manifold_dim}
    # Get the multi-index of the element
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])

    # Compute the spacing in every direction
    dx = zeros(Float64, manifold_dim)
    for dim_idx in range(1, manifold_dim)
        start_breakpoint_idx = ordered_idx[dim_idx]
        end_breakpoint_idx = ordered_idx[dim_idx] + 1
        dx[dim_idx] = geometry.breakpoints[dim_idx][end_breakpoint_idx] - geometry.breakpoints[dim_idx][start_breakpoint_idx]
    end

    # Generate the Jacobian for the Cartesian grid
    # For each point, it's a diagonal matrix multiplied by the cell spacings in each direction
    J = zeros(Float64, prod(length.(ξ)), manifold_dim, manifold_dim)
    for dim_idx in range(1, manifold_dim)
        J[:, dim_idx, dim_idx] .= dx[dim_idx]
    end

    return J
end

function _get_element_size(geometry::CartesianGeometry{manifold_dim}, element_id::Int) where {manifold_dim}
    ordered_index = FunctionSpaces.linear_to_ordered_index(element_id, Vector(geometry.n_elements))

    element_measure = 1
    for k ∈ 1:1:manifold_dim
        element_measure *= abs(geometry.breakpoints[k][ordered_index[k]+1] - geometry.breakpoints[k][ordered_index[k]])
    end

    return element_measure
end

function _get_element_dimensions(geometry::CartesianGeometry{manifold_dim}, element_id::Int) where {manifold_dim}
    ordered_index = FunctionSpaces.linear_to_ordered_index(element_id, [geometry.n_elements...])

    return [abs(geometry.breakpoints[k][ordered_index[k]+1] - geometry.breakpoints[k][ordered_index[k]]) for k in 1:manifold_dim]
end
