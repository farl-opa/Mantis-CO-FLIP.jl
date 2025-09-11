"""
    CartesianGeometry{manifold_dim, T, CI} <: AbstractAnalyticalGeometry{manifold_dim}

A structure representing a Cartesian grid geometry in `manifold_dim` dimensions.

# Fields
- `breakpoints::NTuple{manifold_dim, AbstractVector{T}}`: A tuple of vectors defining the
    grid points in each dimension.
- `cart_num_elements::CI`: A `CartesianIndices` object representing the indices of elements
    in the grid. Used to convert from linear to cartesian indexing.
"""
struct CartesianGeometry{manifold_dim, T, CI} <: AbstractAnalyticalGeometry{manifold_dim}
    breakpoints::NTuple{manifold_dim, AbstractVector{T}}
    cart_num_elements::CI

    function CartesianGeometry(
        breakpoints::NTuple{manifold_dim, AbstractVector{T}}
    ) where {manifold_dim, T <: Number}
        const_num_elements = ntuple(dim -> length(breakpoints[dim]) - 1, manifold_dim)
        cart_num_elements = CartesianIndices(const_num_elements)

        return new{manifold_dim, T, typeof(cart_num_elements)}(
            breakpoints, cart_num_elements
        )
    end
end

get_cart_num_elements(geometry::CartesianGeometry) = geometry.cart_num_elements
get_breakpoints(geometry::CartesianGeometry) = geometry.breakpoints
get_image_dim(::CartesianGeometry{manifold_dim}) where {manifold_dim} = manifold_dim

function get_num_elements(geometry::CartesianGeometry)
    return prod(get_constituent_num_elements(geometry))
end

function get_constituent_element_id(geometry::CartesianGeometry, element_id::Int)
    return get_cart_num_elements(geometry)[element_id]
end

function get_constituent_num_elements(geometry::CartesianGeometry)
    return Tuple(maximum(get_cart_num_elements(geometry)))
end

function evaluate(
    geometry::CartesianGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    const_element_id = get_constituent_element_id(geometry, element_id)
    breakpoints = get_breakpoints(geometry)
    scaling = ntuple(
        dim ->
            breakpoints[dim][const_element_id[dim] + 1] -
            breakpoints[dim][const_element_id[dim]],
        manifold_dim,
    )
    offset = ntuple(dim -> breakpoints[dim][const_element_id[dim]], manifold_dim)
    num_points = Points.get_num_points(xi)
    eval = zeros(num_points, manifold_dim)
    for (i, point) in enumerate(xi)
        for dim in axes(eval, 2)
            eval[i, dim] += affine_map(point[dim], scaling[dim], offset[dim])
        end
    end

    return eval
end

function jacobian(
    geometry::CartesianGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    const_element_id = get_constituent_element_id(geometry, element_id)
    breakpoints = get_breakpoints(geometry)
    scaling = ntuple(
        dim ->
            breakpoints[dim][const_element_id[dim] + 1] -
            breakpoints[dim][const_element_id[dim]],
        manifold_dim,
    )
    # Generate the Jacobian for the Cartesian grid
    # Per point, it's a diagonal matrix multiplied by the cell spacings in each direction
    num_points = Points.get_num_points(xi)
    J = zeros(num_points, manifold_dim, manifold_dim)
    for dim in axes(J, 3)
        for point in axes(J, 1)
            J[point, dim, dim] = scaling[dim]
        end
    end

    return J
end

function get_element_vertices(
    geometry::CartesianGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    const_element_id = get_constituent_element_id(geometry, element_id)
    breakpoints = get_breakpoints(geometry)
    element_vertices = ntuple(manifold_dim) do dim
        vertex_1 = breakpoints[dim][const_element_id[dim]]
        vertex_2 = breakpoints[dim][const_element_id[dim] + 1]

        return [vertex_1, vertex_2]
    end

    return element_vertices
end

function get_element_lengths(
    geometry::CartesianGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    element_vertices = get_element_vertices(geometry, element_id)
    element_lengths = ntuple(manifold_dim) do dim
        return element_vertices[dim][2] - element_vertices[dim][1]
    end

    return element_lengths
end

function get_element_measure(geometry::CartesianGeometry, element_id::Int)
    element_lengths = get_element_lengths(geometry, element_id)

    return prod(element_lengths)
end
