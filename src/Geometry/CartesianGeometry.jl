
struct CartesianGeometry{manifold_dim} <: AbstractAnalyticalGeometry{manifold_dim}
    num_elements::Int
    breakpoints::NTuple{manifold_dim,Vector{Float64}}
    ordered_indices::CartesianIndices

    """
        CartesianGeometry(breakpoints::NTuple{manifold_dim,T}) where {
            manifold_dim, N <: Number, T <: AbstractVector{N}
        }

    Construct a new CartesianGeometry instance.

    # Arguments
    - `breakpoints`: A tuple of vectors defining the grid points in each dimension.

    # Returns
    A new CartesianGeometry instance.
    """
    function CartesianGeometry(
        breakpoints::NTuple{manifold_dim,AbstractVector{T}}
    ) where {manifold_dim,T<:Number}
        num_elements_per_dim = length.(breakpoints) .- 1
        num_elements = prod(num_elements_per_dim)
        ordered_indices = CartesianIndices(num_elements_per_dim)

        return new{manifold_dim}(num_elements, breakpoints, ordered_indices)
    end
end

function get_ordered_indices(geometry::CartesianGeometry)
    return geometry.ordered_indices
end

function get_ordered_indices(geometry::CartesianGeometry, element_id::Int)
    return get_ordered_indices(geometry)[element_id]
end

function get_breakpoints(geometry::CartesianGeometry)
    return geometry.breakpoints
end

function _get_num_elements_per_dim(
    geometry::CartesianGeometry{manifold_dim}
) where {manifold_dim}
    num_elements_per_dim = ntuple(manifold_dim) do k
        return length(geometry.breakpoints[k]) - 1
    end

    return num_elements_per_dim
end

function get_image_dim(::CartesianGeometry{manifold_dim}) where {manifold_dim}
    return manifold_dim
end

function evaluate(
    geometry::CartesianGeometry{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim}
    # Compute coordinates as the tensor product of the unidimensional points
    n_points_per_dim = size.(xi, 1)
    n_points = prod(n_points_per_dim)  # Total number of points to evaluate
    eval = zeros(Float64, n_points, manifold_dim)

    # Map the evaluation points to the geometry
    ordered_id = get_ordered_indices(geometry, element_id)
    breakpoints = get_breakpoints(geometry)
    univariate_points = ntuple(manifold_dim) do k
        dim_points = zeros(n_points_per_dim[k]) # initialize vector for current dimension

        for point in eachindex(dim_points)
            dim_points[point] =
                (1 - xi[k][point]) * breakpoints[k][ordered_id[k]] +
                xi[k][point] * breakpoints[k][ordered_id[k] + 1]
        end

        return dim_points
    end

    # Get the multidimensional indices for the tensor product points
    ordered_points = CartesianIndices(n_points_per_dim)
    linear_points = LinearIndices(ordered_points)
    for (linear_point_id, ordered_point_id) in zip(linear_points, ordered_points)
        for component_id in 1:manifold_dim
            eval[linear_point_id, component_id] = univariate_points[component_id][
                ordered_point_id[component_id]
            ]
        end
    end

    return eval
end

function jacobian(
    geometry::CartesianGeometry{manifold_dim},
    element_id::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim}
    ordered_id = get_ordered_indices(geometry, element_id)
    breakpoints = get_breakpoints(geometry)

    # Compute the spacing in every direction
    dx = zeros(Float64, manifold_dim)
    for dim_id in range(1, manifold_dim)
        start_id = ordered_id[dim_id]
        end_id = ordered_id[dim_id] + 1
        dx[dim_id] = breakpoints[dim_id][end_id] - breakpoints[dim_id][start_id]
    end

    # Generate the Jacobian for the Cartesian grid
    # Per point, it's a diagonal matrix multiplied by the cell spacings in each direction
    J = zeros(Float64, prod(length.(xi)), manifold_dim, manifold_dim)
    for dim_id in range(1, manifold_dim)
        J[:, dim_id, dim_id] .= dx[dim_id]
    end

    return J
end

function get_element_measure(
    geometry::CartesianGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    ordered_id = get_ordered_indices(geometry, element_id)
    breakpoints = get_breakpoints(geometry)

    element_measure = 1
    for k in 1:manifold_dim
        element_measure *= abs(
            breakpoints[k][ordered_id[k] + 1] - breakpoints[k][ordered_id[k]]
        )
    end

    return element_measure
end

function get_element_lengths(
    geometry::CartesianGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    ordered_id = get_ordered_indices(geometry, element_id)
    breakpoints = get_breakpoints(geometry)

    element_lengths = Vector{Float64}(undef, manifold_dim)
    for k in 1:manifold_dim
        element_lengths[k] = abs(
            breakpoints[k][ordered_id[k] + 1] - breakpoints[k][ordered_id[k]]
        )
    end

    return element_lengths
end
