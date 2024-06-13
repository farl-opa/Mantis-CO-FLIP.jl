
struct Mapping{n, m}
    mapping::Function
    dmapping::Function

    function Mapping(dimension::NTuple{2, Int}, mapping::Function, dmapping::Function)
        return new{dimension[1], dimension[2]}(mapping, dmapping)
    end
end

function get_domain_dim(::Mapping{n, m}) where {n, m}
    return n
end

function get_image_dim(::Mapping{n, m}) where {n, m}
    return m
end

function evaluate(mapping::Mapping{n, m}, x::Matrix{Float64}) where {n, m}
    y = zeros(size(x,1),m)
    for i = 1:size(x,1)
        y[i,:] .= mapping.mapping(x[i,:])
    end
    return y
end

function jacobian(mapping::Mapping{n, m}, x::Matrix{Float64}) where {n, m}
    J = zeros(size(x,1),m,n)
    for i = 1:size(x,1)
        J[i,:,:] .= mapping.dmapping(x[i,:])
    end
    return J
end

struct MappedGeometry{n, m} <: AbstractGeometry{n, m}
    geometry::AbstractGeometry
    mapping::Mapping
    n_elements::Int

    function MappedGeometry{n, m}(geometry::AbstractGeometry{n, k}, mapping::Mapping{k, m}) where {n, k, m}
        n_elements = get_num_elements(geometry)
        return new{n, m}(geometry, mapping, n_elements)
    end
    
    function MappedGeometry(geometry::AbstractGeometry{n, k}, mapping::Mapping{k, m}) where {n, k, m}
        n_elements = get_num_elements(geometry)
        return new{n, m}(geometry, mapping, n_elements)
    end
end

function get_num_elements(geometry::MappedGeometry{n,m}) where {n,m}
    return geometry.n_elements
end

function get_domain_dim(::MappedGeometry{n,m}) where {n, m}
    return n
end

function get_image_dim(::MappedGeometry{n,m}) where {n, m}
    return m
end

function evaluate(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}
    x = evaluate(geometry.geometry, element_idx, ξ)
    x_mapped = evaluate(geometry.mapping, x)
    return x_mapped
end

function jacobian(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}
    J_1 = jacobian(geometry.geometry, element_idx, ξ)  # the Jacobian for the mapping from the elements to base geometry image
    x = evaluate(geometry.geometry, element_idx, ξ)
    J_2 = jacobian(geometry.mapping, x)  # the mapping from the image of the  base geometry to the image of the mapping
    n_eval = size(x,1)
    J = zeros(n_eval, m, n)
    k = get_image_dim(geometry.geometry)
    for i = 1:n_eval
        J[i,:,:] .= reshape(J_2[i,:,:],m,k) * reshape(J_1[i,:,:],k,n)
    end

    return J
end
