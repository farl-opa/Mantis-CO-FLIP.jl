"""
    Mapping{n, m}

A struct representing a mapping function from an n-dimensional domain to an m-dimensional codomain.

# Fields
- `mapping::Function`: The mapping function from ℝⁿ to ℝᵐ.
- `dmapping::Function`: The derivative (Jacobian) of the mapping function.

# Type parameters
- `n`: Dimension of the domain.
- `m`: Dimension of the codomain.
"""
struct Mapping{n, m}
    mapping::Function
    dmapping::Function

    """
        Mapping(dimension::NTuple{2, Int}, mapping::Function, dmapping::Function)

    Construct a new Mapping instance.

    # Arguments
    - `dimension`: A tuple (n, m) specifying the dimensions of the domain and codomain.
    - `mapping`: The mapping function from ℝⁿ to ℝᵐ.
    - `dmapping`: The derivative (Jacobian) of the mapping function.

    # Returns
    A new Mapping instance.
    """
    function Mapping(dimension::NTuple{2, Int}, mapping::Function, dmapping::Function)
        return new{dimension[1], dimension[2]}(mapping, dmapping)
    end
end

"""
    get_domain_dim(::Mapping{n, m}) where {n, m}

Get the dimension of the domain for a Mapping.

# Arguments
- `::Mapping{n, m}`: The Mapping instance (unused in the function body).

# Returns
The domain dimension `n`.
"""
function get_domain_dim(::Mapping{n, m}) where {n, m}
    return n
end

"""
    get_image_dim(::Mapping{n, m}) where {n, m}

Get the dimension of the codomain (image) for a Mapping.

# Arguments
- `::Mapping{n, m}`: The Mapping instance (unused in the function body).

# Returns
The codomain dimension `m`.
"""
function get_image_dim(::Mapping{n, m}) where {n, m}
    return m
end

"""
    evaluate(mapping::Mapping{n, m}, x::Matrix{Float64}) where {n, m}

Evaluate the mapping function for a set of input points.

# Arguments
- `mapping`: The Mapping instance.
- `x`: A matrix of input points, where each row is a point in ℝⁿ.

# Returns
A matrix of output points, where each row is a point in ℝᵐ.
"""
function evaluate(mapping::Mapping{n, m}, x::Matrix{Float64}) where {n, m}
    y = zeros(size(x,1), m)  # Initialize output matrix
    for i = 1:size(x,1)
        y[i,:] .= mapping.mapping(x[i,:])  # Apply mapping function to each input point
    end
    return y
end

"""
    jacobian(mapping::Mapping{n, m}, x::Matrix{Float64}) where {n, m}

Compute the Jacobian of the mapping function for a set of input points.

# Arguments
- `mapping`: The Mapping instance.
- `x`: A matrix of input points, where each row is a point in ℝⁿ.

# Returns
A 3D array representing the Jacobian matrices for each input point.
"""
function jacobian(mapping::Mapping{n, m}, x::Matrix{Float64}) where {n, m}
    J = zeros(size(x,1), m, n)  # Initialize Jacobian array
    for i = 1:size(x,1)
        J[i,:,:] .= mapping.dmapping(x[i,:])  # Compute Jacobian for each input point
    end
    return J
end

"""
    MappedGeometry{n, m} <: AbstractGeometry{n, m}

A struct representing a geometry that has been transformed by a mapping function.

# Fields
- `geometry::AbstractGeometry`: The original geometry.
- `mapping::Mapping`: The mapping function applied to the original geometry.
- `n_elements::Int`: Number of elements in the geometry.

# Type parameters
- `n`: Dimension of the domain.
- `m`: Dimension of the codomain after mapping.
"""
struct MappedGeometry{n, m} <: AbstractGeometry{n, m}
    geometry::AbstractGeometry
    mapping::Mapping
    n_elements::Int

    """
        MappedGeometry{n, m}(geometry::AbstractGeometry{n, k}, mapping::Mapping{k, m}) where {n, k, m}

    Construct a new MappedGeometry instance with explicit type parameters.

    # Arguments
    - `geometry`: The original geometry.
    - `mapping`: The mapping function to be applied.

    # Returns
    A new MappedGeometry instance.
    """
    function MappedGeometry{n, m}(geometry::AbstractGeometry{n, k}, mapping::Mapping{k, m}) where {n, k, m}
        n_elements = get_num_elements(geometry)
        return new{n, m}(geometry, mapping, n_elements)
    end
    
    """
        MappedGeometry(geometry::AbstractGeometry{n, k}, mapping::Mapping{k, m}) where {n, k, m}

    Construct a new MappedGeometry instance with inferred type parameters.

    # Arguments
    - `geometry`: The original geometry.
    - `mapping`: The mapping function to be applied.

    # Returns
    A new MappedGeometry instance.
    """
    function MappedGeometry(geometry::AbstractGeometry{n, k}, mapping::Mapping{k, m}) where {n, k, m}
        n_elements = get_num_elements(geometry)
        return new{n, m}(geometry, mapping, n_elements)
    end
end

"""
    get_num_elements(geometry::MappedGeometry{n,m}) where {n,m}

Get the number of elements in the MappedGeometry.

# Arguments
- `geometry`: The MappedGeometry instance.

# Returns
The number of elements as an integer.
"""
function get_num_elements(geometry::MappedGeometry{n,m}) where {n,m}
    return geometry.n_elements
end

"""
    get_domain_dim(::MappedGeometry{n,m}) where {n, m}

Get the dimension of the domain for a MappedGeometry.

# Arguments
- `::MappedGeometry{n,m}`: The MappedGeometry instance (unused in the function body).

# Returns
The domain dimension `n`.
"""
function get_domain_dim(::MappedGeometry{n,m}) where {n, m}
    return n
end

"""
    get_image_dim(::MappedGeometry{n,m}) where {n, m}

Get the dimension of the codomain (image) for a MappedGeometry.

# Arguments
- `::MappedGeometry{n,m}`: The MappedGeometry instance (unused in the function body).

# Returns
The codomain dimension `m`.
"""
function get_image_dim(::MappedGeometry{n,m}) where {n, m}
    return m
end

"""
    evaluate(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}

Evaluate the MappedGeometry at specific points in the reference element.

# Arguments
- `geometry`: The MappedGeometry instance.
- `element_idx`: The index of the element to evaluate.
- `ξ`: A tuple of vectors representing the evaluation points in the reference element.

# Returns
The evaluated points after applying both the original geometry and the mapping.
"""
function evaluate(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}
    x = evaluate(geometry.geometry, element_idx, ξ)  # Evaluate original geometry
    x_mapped = evaluate(geometry.mapping, x)  # Apply mapping to the result
    return x_mapped
end

"""
    jacobian(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}

Compute the Jacobian of the MappedGeometry at specific points in the reference element.

# Arguments
- `geometry`: The MappedGeometry instance.
- `element_idx`: The index of the element to evaluate.
- `ξ`: A tuple of vectors representing the evaluation points in the reference element.

# Returns
The Jacobian matrix at the specified points, combining the Jacobians of the original geometry and the mapping.
"""
function jacobian(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}
    J_1 = jacobian(geometry.geometry, element_idx, ξ)  # Jacobian of original geometry
    x = evaluate(geometry, element_idx, ξ)  # Evaluate mapped geometry
    J_2 = jacobian(geometry.mapping, x)  # Jacobian of mapping
    n_eval = size(x,1)
    J = zeros(n_eval, m, n)  # Initialize combined Jacobian
    k = get_image_dim(geometry.geometry)
    for i = 1:n_eval
        # Compute combined Jacobian using chain rule: J = J_2 * J_1
        J[i,:,:] .= reshape(J_2[i,:,:], m, k) * reshape(J_1[i,:,:], k, n)
    end

    return J
end
