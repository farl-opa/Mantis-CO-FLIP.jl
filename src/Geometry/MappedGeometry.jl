
struct Mapping{M, dM}
    dimensions::NTuple{2, Int}
    mapping::M
    dmapping::dM

    function Mapping(dimensions::NTuple{2, Int}, mapping::M, dmapping::dM) where {
        M <: Function, dM <: Function
    }
        return new{M, dM}(dimensions, mapping, dmapping)
    end
end

function get_manifold_dim(mapping::Mapping)
    return mapping.dimensions[1]
end

function get_image_dim(mapping::Mapping)
    return mapping.dimensions[2]
end

function evaluate(mapping::Mapping, x::Matrix{Float64})
    image_dim = get_image_dim(mapping)
    num_points = size(x, 1)
    eval = zeros(num_points, image_dim)

    for point in 1:num_points
        eval[point, :] .= mapping.mapping(view(x, point, :))
    end

    return eval
end

function jacobian(mapping::Mapping{M, dM}, x::Matrix{Float64}) where {
    M<:Function, dM<:Function
}
    manifold_dim = get_manifold_dim(mapping)
    image_dim = get_image_dim(mapping)
    num_points = size(x, 1)
    J = zeros(num_points, image_dim, manifold_dim)

    for i in 1:num_points
        J[i,:,:] .= mapping.dmapping(view(x, i, :))  # Compute Jacobian for each input point
    end

    return J
end

struct MappedGeometry{manifold_dim, G, Map} <: AbstractGeometry{manifold_dim}
    geometry::G
    mapping::Map
    num_elements::Int

    function MappedGeometry(geometry::G, mapping::Map) where {
        manifold_dim, G<:AbstractGeometry{manifold_dim}, Map<:Mapping
    }
        num_elements = get_num_elements(geometry)

        return new{manifold_dim, G, Map}(geometry, mapping, num_elements)
    end
end

function get_image_dim(geometry::MappedGeometry)
    return geometry.mapping.dimensions[2]
end

function get_element_lengths(geometry::MappedGeometry, element_id::Int)
    return get_element_lengths(geometry.geometry, element_id) 
end

function get_element_volume(geometry::MappedGeometry, element_id::Int)
    return get_element_volume(geometry.geometry, element_id) 
end

function evaluate(
    geometry::MappedGeometry, element_idx::Int, xi::NTuple{manifold_dim,Vector{Float64}}
) where {manifold_dim}
    x = evaluate(geometry.geometry, element_idx, xi)
    x_mapped = evaluate(geometry.mapping, x)

    return x_mapped
end

function jacobian(
    geometry::MappedGeometry,
    element_idx::Int,
    xi::NTuple{manifold_dim,Vector{Float64}},
) where {manifold_dim}
    # the Jacobian for the mapping from the elements to base geometry image
    J_1 = jacobian(geometry.geometry, element_idx, xi)
    x = evaluate(geometry.geometry, element_idx, xi)
    # the mapping from the image of the  base geometry to the image of the mapping
    J_2 = jacobian(geometry.mapping, x)  

    num_points = size(x,1)
    image_dim = get_image_dim(geometry)
    J_1_image_dim = get_image_dim(geometry.geometry)

    J = zeros(num_points, image_dim, manifold_dim)
    for cart_id in CartesianIndices(J)
        (point, k_im, k_mani) = Tuple(cart_id)
        for k_im_1 in 1:J_1_image_dim
            J[point, k_im, k_mani] += J_2[point, k_im, k_im_1] * J_1[point, k_im_1, k_mani]
        end
    end

    return J
end
