

@doc raw"""
    FormSpace{manifold_dim, form_rank, G, F} <: AbstractFormSpace{manifold_dim, form_rank, G}

Concrete implementation of a function space for differential forms.

# Fields
- `geometry::G`: The geometry of the manifold
- `fem_space::F`: The finite element space(s) used for the form components
- `label::String`: Label for the form space

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the geometry
- `F`: Type of the finite element space
"""
struct FormSpace{manifold_dim, form_rank, G, F} <: AbstractFormSpace{manifold_dim, form_rank, G}
    geometry::G
    fem_space::F
    label::String

    # NOTE: FunctionSpaces.AbstractFunctionSpace does not have a parameter of dimension of manifold,
    # we need to add this, but it implies several changes (I started, but postponed it!)

    @doc raw"""
        FormSpace(form_rank::Int, geometry::G, fem_space::Tuple{F}) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, F <: FunctionSpaces.AbstractFunctionSpace}

    Constructor for 0-forms and n-forms.

    # Arguments
    - `form_rank::Int`: Rank of the differential form (should be `0` or `manifold_dim``).
    - `geometry::G`: The geometry of the manifold.
    - `fem_space::Tuple{F}`: A tuple containing a single finite element space, since 0- and n-forms 
            contain only one component.
    - `label::String`: Label for the form space.
    """
    function FormSpace(form_rank::Int, geometry::G, fem_space::F, label::String) where {manifold_dim, num_components, G <: Geometry.AbstractGeometry{manifold_dim}, F <: FunctionSpaces.AbstractMultiValuedFiniteElementSpace{manifold_dim, num_components}}
        if (form_rank ∈ Set([0, manifold_dim])) && (num_components > 1)
            throw(ArgumentError("Mantis.Forms.FormSpace: form_rank = $form_rank with manifold_dim = $manifold_dim requires only one component multivalued FEM space (got num_compoents = $num_components)."))
        elseif (form_rank ∉ Set([0, manifold_dim])) && (num_components != manifold_dim)
            throw(ArgumentError("Mantis.Forms.FormSpace: form_rank = $form_rank with manifold_dim = $manifold_dim requires a multivalued FEM space with num_components = $manifold_dim (got num_components = $num_components)."))
        end

        new{manifold_dim, form_rank, G, F}(geometry, fem_space, label)
    end
end

@doc raw"""
    evaluate(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {form_rank, G <: Geometry.AbstractGeometry}

Evaluate the basis functions of a differential form space at given points `xi`.

# Arguments
- `form_space::FS`: The differential form space.
- `element_idx::Int`: Index of the element to evaluate.
- `xi::NTuple{manifold_dim, Vector{Float64}}`: Tuple of tensor-product coordinate vectors for evaluation points. 
        Points are the tensor product of the coordinates per dimension, therefore there will be
        ``n_{1} \times \dots \times n_{\texttt{manifold_dim}}`` points where to evaluate.

# Returns
- `local_form_basis`: Vector of arrays containing evaluated basis functions for each component evaluated at each point.
        `local_form_basis[i][j,k]` is the evaluation at the tensor product point `j` of the component `i` of global basis
        `form_basis_indices[i][k]` of component `i` evaluated. Note that basis do not have values in all components. 
        Typically each basis is associated only to one component, e.g., ``\mathrm{d}\xi_{1}``. To avoid populating 
        the evaluation with zeros on the other components, we only return, per component, the basis that have nonzero values 
        on that component.
- `form_basis_indices`: Vector containing the global indices of the basis forms.

# Sizes
- `local_form_basis`: Vector of length `n_form_components`, where each element is an Array{Float64, 2} of size (n_evaluation_points, n_basis_functions)
- `form_basis_indices`: Vector{Int} of length n_basis_functions
"""
function evaluate(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}
    # The form space is made up of components 
    # e.g, 
    #   0-forms: single component
    #   1-forms:(dξ₁, dξ₂) (2D)
    #   1-forms:(dξ₁, dξ₂, dξ₃) (3D)
    #   2-forms:(dξ₁dξ₂) (2D)
    #   2-forms:(dξ₂dξ₃, dξ₃dξ₁, dξ₁dξ₂) (3D)
    #   3-forms: single component
    # We use the numbering of the function space.
    
    # Evaluate the form spaces
    local_form_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(form_space, element_idx, xi, 0)  # (only evaluate the basis (0-th order derivative))
    
    return local_form_basis[1][1], form_basis_indices
end

"""
    _evaluate_form_in_canonical_coordinates(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}, nderivatives::Int) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}

Evaluate the form basis functions and their arbitrary derivatives in canonical coordinates.

All of our function spaces are built in the parameteric domain (e.g., B-splines on a parametric patch with breakpoints [0.0, 0.5, 0.6, 2.0]). However, when we evaluate the (values/derivatives of) the function spaces, what is actually returned are the evaluations for the functions pulled-back to the canonical coordinates [0, 1].
    
For example, in 1D, let the mapping from canonical coordinates to the `i`-th parametric element be `Φᵢ: [0, 1] -> [aᵢ, bᵢ]` then, for a multi-valued function `f` defined on the parametric patch, the evaluate method actually returns the values/derivatives of `f∘Φᵢ`. This is similar to (implicitly) treating `f` as a zero form in the FunctionSpaces module and obviously does not hold for differential forms in general, and that's why the proper pullback is needed corresponding to the map `Φᵢ`.

Consider the multi-valued function `f₀ := f∘Φᵢ`. Then, `FunctionSpaces.evaluate(f)` returns `evaluations::Vector{Vector{Vector{Matrix{Float64}}}` where `evaluations[i][j][k][a,b]` is the evaluation of:
- the `j`-th mixed derivative ...
- of order `i-1` ...
- for the `b`-th basis function ...
- of the `k`-th component ...
- at the `a`-th evaluation point ...
- for `f₀`.
See [`FunctionSpaces.evaluate`] for more details on the order in which all the mixed derivatives of order `i-1` are stored.

Therefore, if `f` is meant to denote a differential form, then all of these evaluations need to be pulled back to the canonical coordinates `[0,1]^manifold_dim`. Note, if `f` is a `k`-form, then all these evaluations are pulled-back as `k`-forms. That's because these evaluations are just partial derivatives of the `k`-form components, they do not represent a differential form of higher rank.
 
# Arguments
- `local_form_basis::Matrix{Matrix{Float64}}`: The basis functions evaluated at the parametric coordinates.
- `element_idx::Int`: Index of the element to evaluate.
- `form_rank::Int`: Rank of the form.
- `manifold_dim::Int`: Dimension of the manifold.

# Returns
- `local_form_basis`: The basis functions evaluated at the canonical coordinates of the element.
"""
function _evaluate_form_in_canonical_coordinates(form_space::FS, element_idx::Int, xi::NTuple{manifold_dim, Vector{Float64}}, nderivatives::Int) where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry{manifold_dim}, FS <: AbstractFormSpace{manifold_dim, form_rank, G}}
    # Evaluate the form spaces on parametric domain ...
    local_form_basis, form_basis_indices = FunctionSpaces.evaluate(form_space.fem_space, element_idx, xi, nderivatives)  # (only evaluate the basis (0-th order derivative))
    # ... and account for the transformation from a parametric mesh element to the canonical mesh element
    local_form_basis = _pullback_to_canonical_coordinates(get_geometry(form_space), local_form_basis, element_idx, form_rank)

    return local_form_basis, form_basis_indices
end

"""
    _pullback_to_canonical_coordinates(local_form_basis::Matrix{Matrix{Float64}}, element_idx::Int, form_rank::Int, manifold_dim::Int)

Pullback the basis functions to the canonical coordinates of the element. See the method `_evaluate_form_in_canonical_coordinates` for more details.
 
# Arguments
- `local_form_basis::Matrix{Matrix{Float64}}`: The basis functions evaluated at the parametric coordinates.
- `element_idx::Int`: Index of the element to evaluate.
- `form_rank::Int`: Rank of the form.
- `manifold_dim::Int`: Dimension of the manifold.

# Returns
- `local_form_basis`: The basis functions evaluated at the canonical coordinates of the element.
"""
function _pullback_to_canonical_coordinates(geometry::Geometry.AbstractGeometry{manifold_dim}, form_evaluations::Vector{Vector{Vector{Matrix{Float64}}}}, element_idx::Int, form_rank::Int) where {manifold_dim}
    
    # Pullback the evaluations to the canonical coordinates of the element
    if form_rank > 0
        # Get the element dimensions
        element_dimensions = Geometry.get_element_dimensions(geometry, element_idx)
        for i ∈ eachindex(form_evaluations)
            for j ∈ eachindex(form_evaluations[i])
                if form_rank == manifold_dim
                    form_evaluations[i][j][1] .*= prod(element_dimensions) # only 1 component
                elseif form_rank == 1
                    for k ∈ 1:manifold_dim
                        form_evaluations[i][j][k] .*= element_dimensions[k] # manifold_dim many components
                    end
                elseif manifold_dim == 3
                    form_evaluations[i][j][1] .*= prod(element_dimensions[2:3])
                    form_evaluations[i][j][2] .*= prod(element_dimensions[1:2:3])
                    form_evaluations[i][j][3] .*= prod(element_dimensions[1:2])
                else
                    throw(ArgumentError("Mantis.Forms.evaluate: combination of (form rank, manifold dim) = ($form_rank, $manifold_dim) is not supported."))
                end
            end
        end
    end

    return form_evaluations
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
    for coordinate_idx = 1:manifold_dim
        key = zeros(Int, manifold_dim)
        key[coordinate_idx] = 1
        der_idx = FunctionSpaces._get_derivative_idx(key)
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
    der_idx_1 = FunctionSpaces._get_derivative_idx([1, 0])
    der_idx_2 = FunctionSpaces._get_derivative_idx([0, 1])
    local_d_form_basis_eval[1] = d_local_fem_basis[2][der_idx_1][2] - d_local_fem_basis[2][der_idx_2][1]
    
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
    
    der_idx_1 = FunctionSpaces._get_derivative_idx([1, 0, 0])
    der_idx_2 = FunctionSpaces._get_derivative_idx([0, 1, 0])
    der_idx_3 = FunctionSpaces._get_derivative_idx([0, 0, 1])
    # First: (∂α₃/∂ξ₂ - ∂α₂/∂ξ₃) dξ₂∧dξ₃
    @. local_d_form_basis_eval[1] = d_local_fem_basis[2][der_idx_2][3] - d_local_fem_basis[2][der_idx_3][2]
    # Second: (∂α₁/∂ξ₃ - ∂α₃/∂ξ₁) dξ₃∧dξ₁
    @. local_d_form_basis_eval[2] = d_local_fem_basis[2][der_idx_3][1] - d_local_fem_basis[2][der_idx_1][3]
    # Third: (∂α₂/∂ξ₁ - ∂α₁/∂ξ₂) dξ₁∧dξ₂
    @. local_d_form_basis_eval[3] = d_local_fem_basis[2][der_idx_1][2] - d_local_fem_basis[2][der_idx_2][1]

    return local_d_form_basis_eval, form_basis_indices
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
    der_idx_1 = FunctionSpaces._get_derivative_idx([1, 0, 0])
    der_idx_2 = FunctionSpaces._get_derivative_idx([0, 1, 0])
    der_idx_3 = FunctionSpaces._get_derivative_idx([0, 0, 1])
    @. local_d_form_basis_eval[1] = d_local_fem_basis[2][der_idx_1][1] + d_local_fem_basis[2][der_idx_2][2] + d_local_fem_basis[2][der_idx_3][3]

    return local_d_form_basis_eval, form_basis_indices
end

@doc raw"""
    get_num_basis(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry}

Returns the number of degrees of freedom of the FormSpace `form_space`.

# Arguments
- `form_space::AbstractFormSpace`: The FormSpace to compute the number of degrees of freedom.

# Returns
- `::Int`: The total number of degrees of freedom of the space.
"""
function get_num_basis(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry}
    return FunctionSpaces.get_num_basis(form_space.fem_space)
end

function get_max_local_dim(form_space::FS) where {FS <: AbstractFormSpace{manifold_dim, form_rank, G}} where {manifold_dim, form_rank, G <: Geometry.AbstractGeometry}
    return FunctionSpaces.get_max_local_dim(form_space.fem_space)
end
