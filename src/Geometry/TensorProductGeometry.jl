
struct TensorProductGeometry{manifold_dim, T} <: AbstractGeometry{manifold_dim}
    geometries::T
    num_elements::Int
    ordered_indices::CartesianIndices

    function TensorProductGeometry(geometries::T) where {
        num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
    }
        num_elements_per_geometry = ntuple(num_geometries) do k
            return get_num_elements(geometries[k])
        end

        num_elements = prod(num_elements_per_geometry)
        ordered_indices = CartesianIndices(num_elements_per_geometry)
        manifold_dim_per_geometry = ntuple(num_geometries) do k
            return get_manifold_dim(geometries[k])
        end

        manifold_dim = sum(manifold_dim_per_geometry)

        return new{manifold_dim, T}(
            geometries,
            num_elements,
            ordered_indices
        )
    end
end

function _get_num_elements_per_geometry(
    geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    num_elements_per_geometry = ntuple(num_geometries) do k
        return get_num_elements(geometry.geometries[k])
    end

    return num_elements_per_geometry
end

function get_geometry(geometry::TensorProductGeometry, geometry_id::Int)
    return geometry.geometries[geometry_id]
end

function get_ordered_indices(geometry::TensorProductGeometry)
    return geometry.ordered_indices
end

function get_ordered_indices(geometry::TensorProductGeometry, element_id::Int)
    return get_ordered_indices(geometry)[element_id]
end

function get_element_lengths(
    geometry::TensorProductGeometry{manifold_dim, T}, element_id::Int
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    ordered_index = get_ordered_indices(geometry, element_id)
    element_lengths_per_geometry = ntuple(num_geometries) do i
        return get_element_lengths(geometry.geometries[i], ordered_index[i])
    end

    return collect(Iterators.flatten(element_lengths_per_geometry))
end

function get_element_measure(geometry::TensorProductGeometry, element_id::Int)
    return prod(get_element_lengths(geometry, element_id))
end

function _get_manifold_dim_per_geometry(geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    manifold_dim_per_geometry = ntuple(num_geometries) do k
        return get_manifold_dim(geometry.geometries[k])
    end

    return manifold_dim_per_geometry
end

function _get_cum_manifold_dim_per_geometry(geometry::TensorProductGeometry)
    cum_manifold_dim_per_geometry = cumsum(
        (0, _get_manifold_dim_per_geometry(geometry)...)
    )

    return cum_manifold_dim_per_geometry
end

function _get_manifold_ranges(geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    manifold_dim_per_geometry = _get_manifold_dim_per_geometry(geometry)
    cum_manifold_dim_per_geometry = _get_cum_manifold_dim_per_geometry(geometry)

    manifold_ranges = ntuple(num_geometries) do k
        beg_idx = cum_manifold_dim_per_geometry[k]+1
        end_idx = cum_manifold_dim_per_geometry[k]+manifold_dim_per_geometry[k]

        return beg_idx:end_idx
    end

    return manifold_ranges
end

function _get_image_dim_per_geometry(geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    image_dim_per_geometry = ntuple(num_geometries) do k
        return get_image_dim(geometry.geometries[k])
    end

    return image_dim_per_geometry
end

function _get_cum_image_dim_per_geometry(geometry::TensorProductGeometry)
    cum_image_dim_per_geometry = cumsum(
        (0, _get_image_dim_per_geometry(geometry)...)
    )

    return cum_image_dim_per_geometry
end

function _get_image_ranges(geometry::TensorProductGeometry{manifold_dim, T}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    image_dim_per_geometry = _get_image_dim_per_geometry(geometry)
    cum_image_dim_per_geometry = _get_cum_image_dim_per_geometry(geometry)

    image_ranges = ntuple(num_geometries) do k
        beg_idx = cum_image_dim_per_geometry[k]+1
        end_idx = cum_image_dim_per_geometry[k]+image_dim_per_geometry[k]

        return beg_idx:end_idx
    end

    return image_ranges
end

function get_image_dim(geometry::TensorProductGeometry{manifold_dim, T}) where {
    manifold_dim, T
}
    return sum(_get_image_dim_per_geometry(geometry))
end

"""
    evaluate(
        geometry::TensorProductGeometry{manifold_dim, T},
        element_idx::Int,
        ξ::NTuple{manifold_dim, Vector{Float64}}
    ) where {
        manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
    }

Go [here](Modules/Geometry.md) for more details.
"""
function evaluate(
    geometry::TensorProductGeometry{manifold_dim, T},
    element_idx::Int,
    ξ::NTuple{manifold_dim, Vector{Float64}}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    ordered_indices = get_ordered_indices(geometry, element_idx)
    manifold_ranges = _get_manifold_ranges(geometry)
    # Evaluate the subgeometries at their associated evaluation points
    eval_per_geometry = ntuple(num_geometries) do k
        return evaluate(
            geometry.geometries[k], ordered_indices[k], ξ[manifold_ranges[k]]
        )
    end

    image_ranges = _get_image_ranges(geometry)
    n_points_per_geometry = ntuple(num_geometries) do k
        return prod(size.(ξ[manifold_ranges[k]], 1))
    end
    n_points = prod(n_points_per_geometry) # total number of evaluation points
    image_dim = get_image_dim(geometry)
    eval = zeros(Float64, n_points, image_dim) # evaluation storage
    
    ordered_points = CartesianIndices(n_points_per_geometry)
    linear_points = LinearIndices(ordered_points)
    # loop over all points
    for k in 1:num_geometries # loop over geometries
        for (lin_point, ord_point) in zip(linear_points, ordered_points)
            eval[lin_point, image_ranges[k]] .= @view eval_per_geometry[k][ord_point[k], :]
        end
    end

    return eval
end

"""
    jacobian(
        geometry::TensorProductGeometry{manifold_dim, T},
        element_idx::Int,
        ξ::NTuple{manifold_dim, Vector{Float64}}
    ) where {
        manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
    }

Go [here](Modules/Geometry.md) for more details.
"""
function jacobian(
    geometry::TensorProductGeometry{manifold_dim, T},
    element_idx::Int,
    ξ::NTuple{manifold_dim, Vector{Float64}}
) where {
    manifold_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
}
    ordered_indices = get_ordered_indices(geometry, element_idx)
    manifold_ranges = _get_manifold_ranges(geometry)
    # Evaluate the jacobian for each subgeometry at their associated evaluation points
    jacobian_per_geometry = ntuple(num_geometries) do k
        return jacobian(
            geometry.geometries[k], ordered_indices[k], ξ[manifold_ranges[k]]
        )
    end

    image_ranges = _get_image_ranges(geometry)
    n_points_per_geometry = ntuple(num_geometries) do k
        return prod(size.(ξ[manifold_ranges[k]], 1))
    end
    n_points = prod(n_points_per_geometry) # total number of evaluation points
    image_dim = get_image_dim(geometry)
    jacobian_eval = zeros(Float64, n_points, image_dim, manifold_dim) # evaluation storage

    ordered_points = CartesianIndices(n_points_per_geometry)
    linear_points = LinearIndices(ordered_points)
    # loop over all points
    for k in 1:num_geometries # loop over geometries
        for (lin_point, ord_point) in zip(linear_points, ordered_points)
            jacobian_eval[lin_point, image_ranges[k], manifold_ranges[k]] .= view(
                jacobian_per_geometry[k], ord_point[k], :, :
            )
        end
    end

    return jacobian_eval
end
