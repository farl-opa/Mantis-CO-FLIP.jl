############################################################################################
#                                        Structure                                         #
############################################################################################

struct HierarchicalGeometry{manifold_dim, H} <: AbstractGeometry{manifold_dim}
    hier_space::H

    function HierarchicalGeometry(
        hier_space::FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim, S, T}
    ) where {manifold_dim, S, T}
        return new{
            manifold_dim, FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim, S, T}
        }(
            hier_space
        )
    end
end

############################################################################################
#                                      Basic Getters                                       #
############################################################################################

function get_space(hier_geo::HierarchicalGeometry)
    return hier_geo.hier_space
end

function get_num_elements(hier_geo::HierarchicalGeometry)
    return FunctionSpaces.get_num_elements(get_space(hier_geo))
end

function get_num_levels(hier_geo::HierarchicalGeometry)
    return FunctionSpaces.get_num_levels(get_space(hier_geo))
end

function get_domain_dim(::HierarchicalGeometry{manifold_dim, H}) where {manifold_dim, H}
    return manifold_dim
end

function get_image_dim(::HierarchicalGeometry{manifold_dim, H}) where {manifold_dim, H}
    return manifold_dim
end

function get_num_subdivisions(hier_geo::HierarchicalGeometry)
    return FunctionSpaces.get_num_subdivisions(get_space(hier_geo))
end

############################################################################################
#                                      Other methods                                       #
############################################################################################

function evaluate(
    hier_geo::HierarchicalGeometry{manifold_dim, H},
    hier_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, H}
    element_vertices = FunctionSpaces.get_element_vertices(get_space(hier_geo), hier_id)
    A = zeros(Float64, manifold_dim, manifold_dim)
    b = zeros(Float64, manifold_dim)
    for k in 1:manifold_dim
        A[k, k] = (element_vertices[k][2] - element_vertices[k][1])
        b[k] = element_vertices[k][1]
    end

    mapped_points = affine_map(xi, A, b)
    mapped_matrix = vec_tuple_to_matrix(mapped_points)

    return mapped_matrix
end

function jacobian(
    hier_geo::HierarchicalGeometry{manifold_dim, H},
    hier_id::Int,
    xi::NTuple{manifold_dim, Vector{Float64}},
) where {manifold_dim, H}
    element_vertices = FunctionSpaces.get_element_vertices(get_space(hier_geo), hier_id)
    delta = zeros(Float64, manifold_dim)
    for k in 1:manifold_dim
        delta[k] = (element_vertices[k][2] - element_vertices[k][1])
    end

    num_points = prod(length.(xi))
    J = zeros(Float64, num_points, manifold_dim, manifold_dim)
    for k in range(1, manifold_dim)
        for point in 1:num_points
            J[point, k, k] = delta[k]
        end
    end

    return J
end

function get_element_lengths(hier_geo::HierarchicalGeometry, hier_id::Int)
    level, element_level_id = FunctionSpaces.convert_to_element_level_and_level_id(
        get_space(hier_geo), hier_id
    )

    return FunctionSpaces.get_element_dimensions(
        FunctionSpaces.get_space(get_space(hier_geo), level), element_level_id
    )
end

function get_element_measure(hier_geo::HierarchicalGeometry, hier_id::Int)
    level, element_level_id = FunctionSpaces.convert_to_element_level_and_level_id(
        get_space(hier_geo), hier_id
    )

    return FunctionSpaces.get_element_measure(
        FunctionSpaces.get_space(get_space(hier_geo), level), element_level_id
    )
end
