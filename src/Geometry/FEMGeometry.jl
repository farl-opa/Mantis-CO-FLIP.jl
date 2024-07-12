import LinearAlgebra

"""
    FEMGeometry{n,m} <: AbstractGeometry{n, m}

A struct representing the geometry of a Finite Element Method (FEM) problem.

# Fields
- `geometry_coeffs::Array{Float64,2}`: Coefficients defining the geometry.
- `fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}`: The finite element space.
- `n_elements::Int`: Number of elements in the FEM mesh.

# Type parameters
- `n`: Dimension of the domain (e.g., 1D, 2D, 3D).
- `m`: Dimension of the image (codomain).
"""
struct FEMGeometry{n,m} <: AbstractGeometry{n, m}
    geometry_coeffs::Array{Float64,2}
    fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}
    n_elements::Int

    """
        FEMGeometry(fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}, geometry_coeffs::Array{Float64,2}) where {n}

    Construct a new FEMGeometry instance.

    # Arguments
    - `fem_space`: The finite element space.
    - `geometry_coeffs`: Coefficients defining the geometry.

    # Returns
    A new FEMGeometry instance.
    """
    function FEMGeometry(fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}, geometry_coeffs::Array{Float64,2}) where {n}
        m = size(geometry_coeffs, 2)  # Determine the image dimension from geometry coefficients
        n_elements = FunctionSpaces.get_num_elements(fem_space)  # Get number of elements from fem_space
        return new{n,m}(geometry_coeffs, fem_space, n_elements)
    end
end

"""
    get_num_elements(geometry::FEMGeometry{n,m}) where {n,m}

Get the number of elements in the FEM mesh.

# Arguments
- `geometry`: The FEMGeometry instance.

# Returns
The number of elements as an integer.
"""
function get_num_elements(geometry::FEMGeometry{n,m}) where {n,m}
    return geometry.n_elements
end

"""
    get_domain_dim(::FEMGeometry{n,m}) where {n, m}

Get the dimension of the domain.

# Arguments
- `::FEMGeometry{n,m}`: The FEMGeometry instance (unused in the function body).

# Returns
The domain dimension `n`.
"""
function get_domain_dim(::FEMGeometry{n,m}) where {n, m}
    return n
end

"""
    get_image_dim(::FEMGeometry{n,m}) where {n, m}

Get the dimension of the image (codomain).

# Arguments
- `::FEMGeometry{n,m}`: The FEMGeometry instance (unused in the function body).

# Returns
The image dimension `m`.
"""
function get_image_dim(::FEMGeometry{n,m}) where {n, m}
    return m
end

"""
    evaluate(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}

Evaluate the FEM geometry at specific points in the reference element.

# Arguments
- `geometry`: The FEMGeometry instance.
- `element_id`: The ID of the element to evaluate.
- `xi`: A tuple of vectors representing the evaluation points in the reference element.

# Returns
The evaluated geometry at the specified points.
"""
function evaluate(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # Evaluate the FEM space basis functions
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 0)
    
    # Combine basis functions with geometry coefficients and return
    return fem_basis[1][1] * geometry.geometry_coeffs[fem_basis_indices,:]
end

"""
    jacobian(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}

Compute the Jacobian of the FEM geometry at specific points in the reference element.

# Arguments
- `geometry`: The FEMGeometry instance.
- `element_id`: The ID of the element to evaluate.
- `xi`: A tuple of vectors representing the evaluation points in the reference element.

# Returns
The Jacobian matrix at the specified points.
"""
function jacobian(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # Jᵢⱼ = ∂Φⁱ/∂ξⱼ
    # Evaluate FEM space basis functions and their derivatives
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 1)
    
    # Initialize variables
    keys = Matrix{Int}(LinearAlgebra.I, n, n)  # Identity matrix for derivative indices
    n_eval_points = prod(length.(xi))  # Total number of evaluation points
    J = zeros(n_eval_points, m, n)  # Initialize Jacobian matrix
    
    # Compute Jacobian
    for k = 1:n 
        der_idx = FunctionSpaces._get_derivative_idx(keys[k,:])  # Get derivative index
        # Compute partial derivatives and store in Jacobian matrix
        J[:, :, k] .= fem_basis[2][der_idx] * geometry.geometry_coeffs[fem_basis_indices,:]
    end
    
    return J
end
