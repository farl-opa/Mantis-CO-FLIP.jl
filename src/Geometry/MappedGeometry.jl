
struct Mapping{M, dM}
    dimensions::NTuple{2, Int}
    mapping::M
    dmapping::dM

    function Mapping(dimensions::NTuple{2, Int}, mapping::M, dmapping::dM) where {M<:Function, dM<:Function}
        return new{M, dM}(dimensions, mapping, dmapping)
    end
end

function get_domain_dim(mapping::Mapping{M, dM}) where {M<:Function, dM<:Function}
    return mapping.dimensions[1]
end

function get_image_dim(mapping::Mapping{M, dM}) where {M<:Function, dM<:Function}
    return mapping.dimensions[2]
end

function evaluate(mapping::Mapping{M, dM}, x::Matrix{Float64}) where {M<:Function, dM<:Function}
    m = get_image_dim(mapping)
    y = zeros(size(x,1),m)
    for i = 1:size(x,1)
        y[i,:] .= mapping.mapping(x[i,:])  # Apply mapping function to each input point
    end
    return y
end

function jacobian(mapping::Mapping{M, dM}, x::Matrix{Float64}) where {M<:Function, dM<:Function}
    n = get_domain_dim(mapping)
    m = get_image_dim(mapping)
    J = zeros(size(x,1),m,n)
    for i = 1:size(x,1)
        J[i,:,:] .= mapping.dmapping(x[i,:])  # Compute Jacobian for each input point
    end
    return J
end

struct MappedGeometry{n, G, Map} <: AbstractGeometry{n}
    geometry::G
    mapping::Map
    n_elements::Int

    function MappedGeometry(geometry::G, mapping::Map) where {G<:AbstractGeometry{n}, Map<:Mapping} where {n}
        n_elements = get_num_elements(geometry)
        return new{n, G, Map}(geometry, mapping, n_elements)
    end
end

function get_num_elements(geometry::MappedGeometry{n, G, Map} ) where {n, G<:AbstractGeometry{n}, Map<:Mapping}
    return geometry.n_elements
end

function get_domain_dim(_::MappedGeometry{n, G, Map} ) where {n, G<:AbstractGeometry{n}, Map<:Mapping}
    return n
end

function get_image_dim(geometry::MappedGeometry{n, G, Map} ) where {n, G<:AbstractGeometry{n}, Map<:Mapping}
    return geometry.mapping.dimensions[2]
end

function _get_element_dimensions(geometry::MappedGeometry{n, G, Map}, element_id::Int) where {n, G<:AbstractGeometry{n}, Map<:Mapping}
    return _get_element_dimensions(geometry.geometry, element_id)
end

function evaluate(geometry::MappedGeometry{n, G, Map}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, G<:AbstractGeometry{n}, Map<:Mapping}
    x = evaluate(geometry.geometry, element_idx, ξ)
    x_mapped = evaluate(geometry.mapping, x)
    return x_mapped
end

function jacobian(geometry::MappedGeometry{n, G, Map}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, G<:AbstractGeometry{n}, Map<:Mapping}
    J_1 = jacobian(geometry.geometry, element_idx, ξ)  # the Jacobian for the mapping from the elements to base geometry image
    x = evaluate(geometry.geometry, element_idx, ξ)
    J_2 = jacobian(geometry.mapping, x)  # the mapping from the image of the  base geometry to the image of the mapping
    n_eval = size(x,1)
    m = get_image_dim(geometry)
    J = zeros(n_eval, m, n)
    k = get_image_dim(geometry.geometry)
    for i = 1:n_eval
        # Compute combined Jacobian using chain rule: J = J_2 * J_1
        J[i,:,:] .= reshape(J_2[i,:,:], m, k) * reshape(J_1[i,:,:], k, n)
    end

    return J
end
