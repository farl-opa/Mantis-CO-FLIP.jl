
@doc raw"""
    evaluate_exterior_derivative(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}

Evaluate the exterior derivative of a FormField at given points.

# Arguments
- `form::FormField`: The form field to evaluate
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `d_form_eval::Vector{Vector{Float64}}`: Evaluated exterior derivative of the form field.
        Follows the same data format as evaluate: `d_form_eval[i][j]` is the component `i` 
        of the exterior derivative evaluated at the tensor product `j`.

# Sizes
- `d_form_eval`: Vector of length `n_derivative_components`, where each element is a Vector{Float64} of length `n_evaluation_points`
        `n_evaluation_points = n_1 * ... * n_n`, where `n_i` is the number of points in the component `i` of the tensor product Tuple.
- `d_form_indices`: Vector of length `n_derivative_components`, where each element is a Vector{Int} of length 1 of value 1, since for a 
        `FormField` there is only one `basis`. This is done for consistency with `FormBasis`.
"""
function evaluate_exterior_derivative(form::FormField{manifold_dim, form_rank, G, FS}, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}
    n_form_components = binomial(manifold_dim, form_rank)
    d_form_basis_eval, form_basis_indices = evaluate_exterior_derivative(form.form_space, element_idx, xi)
    n_derivative_components = size(d_form_basis_eval, 1)
    
    d_form_eval = Vector{Vector{Float64}}(undef, n_derivative_components)

    # This algorithm works for all exterior derivatives, as long as the 
    # exterior derivative of the basis of the form space are correctly 
    # implemented. The algorithm works in the following way.
    # 
    # The number of components, n_form_components, of the original k-form, 
    # αᵏ, is given by
    #
    #   n_form_components = binomial(manifold_dim, form_rank)
    #
    # This means that αᵏ is given by n_form_components sets of coefficients, aᵢ,
    # one set per component, and an equal number of sets of basis functions, Nⱼ,
    # 
    #   aᵢ = [aᵢ₁, ... , aᵢᵣ] with i = 1, ..., n_form_components and r = Mᵢ the number of coefficients (basis) in this component
    #   Nᵢ = [Nᵢ₁, ..., Nᵢᵣ], with i and r as above
    # 
    # The computation (evaluation) of the k-form is to loop over the components and multiply
    # the basis by the coefficients (the basis as evaluated at the points of interest)
    # 
    #  αᵏᵢ = ∑ʳⱼ₌₁ aᵢⱼNᵢⱼ
    #
    # The computation of the derivative is the same, as long as there is
    # consistency on the computation of the derivatives of the basis. The derivative of
    # the k-form can be directly computed as
    #
    #   d αᵏᵢ = d ∑ʳⱼ₌₁ aᵢⱼNᵢⱼ = ∑ʳⱼ₌₁ aᵢⱼ d Nᵢⱼ
    #
    # The components of the derivatives of the basis need to be compatible, i.e., we need to use canonical basis:
    #       1-forms: dξ₁, dξ₂, dξ₃                    (in this order)
    #       2-forms: dξ₂ ∧ dξ₃, dξ₃ ∧ dξ₁, dξ₁ ∧ dξ₂  (in this order)
    #       3-forms: dξ₁ ∧ dξ₂ ∧ dξ₃
    # 
    for derivative_form_component_idx in 1:n_derivative_components
        d_form_eval[derivative_form_component_idx] = d_form_basis_eval[derivative_form_component_idx] * form.coefficients[form_basis_indices[1]]
    end

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return d_form_eval, [[1]]
end

@doc raw"""
    evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0, G}} where {G <: Geometry.AbstractGeometry}

Evaluate the exterior derivative of a 0-form (scalar function) at given points.

# Arguments
- `form_space::FS`: The 0-form space.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_d_form_basis_eval`: Vector of arrays containing evaluated exterior derivative basis functions
        `local_d_form_basis_eval[i][j, k]` is the component `i` of the exterior derivative of global basis 
        `form_basis_indices[k]` evaluated at the tensor product point `j`. 
        See [`(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}`](@ref)
        for more details: the formats are identical.
- `form_basis_indices`: Vector indices of the basis functions
        `form_basis_indices[k]` is the global index of the  exterior derivative of the 
        returned basis `k` of this element.

# Sizes
- `local_d_form_basis_eval`: Vector of length `manifold_dim`, where each element is an Array{Float64, 2} of size (n_evaluation_points, n_basis_functions).
- `form_basis_indices`: Vector{Int} of length n_basis_functions.
"""
function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, 0, G}  where {G <: Geometry.AbstractGeometry{manifold_dim}}}
    # Preallocate memory for output array
    n_derivative_form_components = manifold_dim
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_idx)
    n_evaluation_points = prod(size.(xi, 1))
    
    # We can avoid this if we change the output format of evaluation of directsum spaces
    # flip the second with the third index there...
    local_d_form_basis_eval = [zeros(Float64, n_evaluation_points, n_basis_functions) for _ = 1:n_derivative_form_components]

    # Evaluate derivatives
    d_local_fem_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(form_space, element_idx, xi, 1)

    # Store the required values
    key = zeros(Int, manifold_dim)
    for coordinate_idx = 1:manifold_dim
        key[coordinate_idx] = 1
        der_idx = FunctionSpaces.get_derivative_idx(key)
        key[coordinate_idx] = 0
        @. local_d_form_basis_eval[coordinate_idx] = d_local_fem_basis[2][der_idx][1]
    end

    return local_d_form_basis_eval, form_basis_indices
end

function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{n, n, G}} where {n, G <: Geometry.AbstractGeometry{n}}
    throw(ArgumentError("Mantis.Forms.evaluate_exterior_derivative: Manifold dim == Form rank: Unable to compute exterior derivative of volume forms."))
end

@doc raw"""
    evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{2, Vector{Float64}}) where {FS <: AbstractFormSpace{2, 1, G}} where {G <: Geometry.AbstractGeometry}

Evaluate the exterior derivative of a 1-form in 2D.

# Arguments
- `form_space::FS`: The 1-form space in 2D.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{2, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_d_form_basis_eval`: Vector of arrays containing evaluated exterior derivative basis functions
        `local_d_form_basis_eval[i][j, k]` is the component `i` of the exterior derivative of global basis 
        `form_basis_indices[k]` evaluated at the tensor product point `j`. 
        See [`(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}`](@ref)
        for more details: the formats are identical.
- `form_basis_indices`: Vector of indices of the active basis functions
        `form_basis_indices[k]` is the global index of the  exterior derivative of the 
        returned basis `k` of this element.

# Sizes
- `local_d_form_basis_eval`: Vector of length `n_manifold_dim`, where each element is an `Array{Float64, 2}` of size `(n_evaluation_points, n_basis_functions)`.
- `form_basis_indices`: Vector{Int} of length `n_basis_functions`.
"""
function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{2, Vector{Float64}}) where {FS <: AbstractFormSpace{2, 1, G}} where {G <: Geometry.AbstractGeometry{2}}
    # manifold_dim = 2
    n_derivative_form_components = 1 # binomial(manifold_dim, 2)
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_idx)
    n_evaluation_points = prod(size.(xi, 1))

    # Preallocate memory for output array
    local_d_form_basis_eval = [zeros(Float64, n_evaluation_points, n_basis_functions) for _ = 1:n_derivative_form_components]
    
    # Evaluate derivatives
    d_local_fem_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(form_space, element_idx, xi, 1)
    
    # The exterior derivative is 
    # (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    # Store the required values
    der_idx_1 = FunctionSpaces.get_derivative_idx([1, 0])
    der_idx_2 = FunctionSpaces.get_derivative_idx([0, 1])
    @. local_d_form_basis_eval[1] = d_local_fem_basis[2][der_idx_1][2] - d_local_fem_basis[2][der_idx_2][1]
    
    return local_d_form_basis_eval, form_basis_indices
end

@doc raw"""
    evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 1, G}} where {G <: Geometry.AbstractGeometry}

Evaluate the exterior derivative of a 1-form in 3D.

# Arguments
- `form_space::FS`: The 1-form space in 3D
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_d_form_basis_eval`: Vector of arrays containing evaluated exterior derivative basis functions
        `local_d_form_basis_eval[i][j, k]` is the component `i` of the exterior derivative of global basis 
        `form_basis_indices[k]` evaluated at the tensor product point `j`. 
        See [`(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}`](@ref)
        for more details: the formats are identical.
- `form_basis_indices`: Vector of indices of the active basis functions
        `form_basis_indices[k]` is the global index of the  exterior derivative of the 
        returned basis `k` of this element.

# Sizes
- `local_d_form_basis_eval`: Vector of length `n_manifold_dim`, where each element is an `Array{Float64, 2}` of size `(n_evaluation_points, n_basis_functions)`.
- `form_basis_indices`: Vector{Int} of length `n_basis_functions`.
"""
function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 1, G}} where {G <: Geometry.AbstractGeometry{3}}
    # manifold_dim = 3
    n_derivative_form_components = 3 # binomial(manifold_dim, 2)
    
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_idx)
    n_evaluation_points = prod(size.(xi, 1))

    # Preallocate memory for output array
    local_d_form_basis_eval = [zeros(Float64, n_evaluation_points, n_basis_functions) for _ = 1:n_derivative_form_components]
    
    # Evaluate the underlying FEM space and its first order derivatives (all derivatives for each component)
    d_local_fem_basis, form_basis_indices = FunctionSpaces.evaluate(form_space.fem_space, element_idx, xi, 1)
    
    # The exterior derivative is 
    # (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃ + (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁ + (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    der_idx_1 = FunctionSpaces.get_derivative_idx([1, 0, 0])
    der_idx_2 = FunctionSpaces.get_derivative_idx([0, 1, 0])
    der_idx_3 = FunctionSpaces.get_derivative_idx([0, 0, 1])
    # First: (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃
    @. local_d_form_basis_eval[1] = d_local_fem_basis[2][der_idx_2][3] - d_local_fem_basis[2][der_idx_3][2]
    # Second: (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁
    @. local_d_form_basis_eval[2] = d_local_fem_basis[2][der_idx_3][1] - d_local_fem_basis[2][der_idx_1][3]
    # Third: (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    @. local_d_form_basis_eval[3] = d_local_fem_basis[2][der_idx_1][2] - d_local_fem_basis[2][der_idx_2][1]

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return local_d_form_basis_eval, [form_basis_indices]
end

@doc raw"""
    evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 2, G}} where {G <: Geometry.AbstractGeometry}

Evaluate the exterior derivative of a 2-form in 3D.

# Arguments
- `form_space::FS`: The 2-form space in 3D
- `element_idx::Int`: Index of the element to evaluate
- `xi::NTuple{3, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_d_form_basis_eval`: Vector of arrays containing evaluated exterior derivative basis functions
        `local_d_form_basis_eval[i][j, k]` is the component `i` of the exterior derivative of global basis 
        `form_basis_indices[i][k]` evaluated at the tensor product point `j`. 
        See [`(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}`](@ref)
        for more details: the formats are identical.
- `form_basis_indices`: Vector of vectors containing indices of the basis functions
        `form_basis_indices[i][k]` is the global index of the  exterior derivative of the 
        returned basis `k` of this element from component `i`.

# Sizes
- `local_d_form_basis_eval`: Vector of length 1, where the element is an Array{Float64, 2} of size (n_evaluation_points, n_basis_functions_dξ_2_dξ_3 + n_basis_functions_dξ_3_dξ_1 + n_basis_functions_dξ_1_dξ_2)
- `form_basis_indices`: Vector of length 1, where the element is a Vector{Int} of length (n_basis_functions_dξ_2_dξ_3 + n_basis_functions_dξ_3_dξ_1 + n_basis_functions_dξ_1_dξ_2)
"""
function evaluate_exterior_derivative(form_space::FS, element_idx::Int, xi::NTuple{3, Vector{Float64}}) where {FS <: AbstractFormSpace{3, 2, G}} where {G <: Geometry.AbstractGeometry{3}}
    # manifold_dim = 3
    n_derivative_form_components = 1 # binomial(manifold_dim, 2)
    
    n_basis_functions = FunctionSpaces.get_num_basis(form_space.fem_space, element_idx)
    n_evaluation_points = prod(size.(xi, 1))

    # Preallocate memory for output array
    local_d_form_basis_eval = [zeros(Float64, n_evaluation_points, n_basis_functions) for _ = 1:n_derivative_form_components]
    
    # Evaluate the underlying FEM space and its first order derivatives (all derivatives for each component)
    d_local_fem_basis, form_basis_indices = FunctionSpaces.evaluate(form_space.fem_space, element_idx, xi, 1)
    
    # The form is 
    # α₁ dξ₂∧dξ₃ + α₂ dξ₃∧dξ₁ + α₃ dξ₁∧dξ₂
    # The exterior derivative is 
    # (∂α₁/∂ξ₁ + ∂α₂/∂ξ₂ + ∂α₃/∂ξ₃) dξ₁∧dξ₂∧dξ₃
    der_idx_1 = FunctionSpaces.get_derivative_idx([1, 0, 0])
    der_idx_2 = FunctionSpaces.get_derivative_idx([0, 1, 0])
    der_idx_3 = FunctionSpaces.get_derivative_idx([0, 0, 1])
    @. local_d_form_basis_eval[1] = d_local_fem_basis[2][der_idx_1][1] + d_local_fem_basis[2][der_idx_2][2] + d_local_fem_basis[2][der_idx_3][3]
    
    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return local_d_form_basis_eval, [form_basis_indices]
end
